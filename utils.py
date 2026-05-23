"""
utils.py - reusable correctness + profiling helpers for GPU learning.

Write this once, import it in every project. The whole point is that the
measurement harness never changes, so your numbers are comparable across
weeks and you spend attention interpreting profiles, not rebuilding them.

Requires a CUDA-capable GPU (i.e. run it on your RunPod box, not your laptop).
"""

import torch
from torch.profiler import profile, schedule, ProfilerActivity


def benchmark(fn, *args, warmup=15, iters=100, **kwargs):
    """Mean milliseconds per call, measured correctly for GPU work.

    Handles the two classic traps:
      - warmup: throw away one-time costs (kernel compile, allocation, cold caches)
      - sync:   GPU calls are async, so we synchronize before stopping the clock
    Uses CUDA events, which timestamp on the GPU itself (more accurate than
    time.time() for GPU work).
    """
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()  # wait for the GPU to actually finish

    return start.elapsed_time(end) / iters  # ms per call


def check_correct(mine, reference, atol=1e-2, rtol=1e-2, name="output"):
    """Compare your output against a trusted reference.

    Prints a verdict and, on failure, the max absolute error and where it
    occurs (so you can debug the *pattern* of the bug). Returns True/False.

    Note: never use ==. GPU math reorders operations, so a correct kernel
    still differs in the last few digits. allclose checks "close enough".
    Loosen atol/rtol for fp16/bf16 (~1e-2), tighten for fp32 (~1e-4).
    """
    ok = torch.allclose(mine, reference, atol=atol, rtol=rtol)
    if ok:
        print(f"[correct] {name} matches reference (atol={atol}, rtol={rtol})")
    else:
        diff = (mine - reference).abs()
        print(f"[WRONG]   {name} does NOT match reference")
        print(f"          max abs error: {diff.max().item():.3e}")
        print(f"          worst element (flattened index): {diff.argmax().item()}")
    return ok


def compare(my_fn, ref_fn, *args, atol=1e-2, rtol=1e-2, **kwargs):
    """One-call correctness + speed comparison against a reference function.

    Prints the correctness verdict and the speed ratio (mine / reference).
    Returns (t_mine_ms, t_ref_ms) so the caller can compute GFLOPS, etc.
    """
    mine = my_fn(*args, **kwargs)
    reference = ref_fn(*args, **kwargs)
    check_correct(mine, reference, atol=atol, rtol=rtol)

    t_mine = benchmark(my_fn, *args, **kwargs)
    t_ref = benchmark(ref_fn, *args, **kwargs)
    verdict = "slower" if t_mine > t_ref else "faster"
    print(
        f"[speed]   mine: {t_mine:.4f} ms   reference: {t_ref:.4f} ms   "
        f"ratio: {t_mine / t_ref:.2f}x ({verdict})"
    )
    return t_mine, t_ref


def gflops(flop_count, ms):
    """Convert a known FLOP count + time-in-ms to GFLOPS."""
    return flop_count / (ms / 1000) / 1e9


def matmul_flops(m, n, k):
    """FLOPs for an (m x k) @ (k x n) matmul = 2 * m * n * k."""
    return 2 * m * n * k


def profile_timeline(fn, *args, active=10, row_limit=12,
                     trace_path="trace.json", **kwargs):
    """Timeline pass with the torch profiler. Answers: where is the time going?

    Uses a schedule (wait/warmup/active) so warmup is handled formally instead
    of by hand. Prints the top ops by CUDA time and writes a trace you can open
    at ui.perfetto.dev (or chrome://tracing).
    """
    wait, warmup = 1, 5
    total_steps = wait + warmup + active
    sched = schedule(wait=wait, warmup=warmup, active=active, repeat=1)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=sched,
    ) as prof:
        for _ in range(total_steps):
            fn(*args, **kwargs)
            prof.step()
            torch.cuda.synchronize()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=row_limit))
    prof.export_chrome_trace(trace_path)
    print(f"[trace]   wrote {trace_path} - open at ui.perfetto.dev")

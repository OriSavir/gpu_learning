"""
bench.py - per-project benchmark entry point. Copy this into each week's
project folder and edit the three sections marked EDIT below.

Two ways to run it:

  # Full pass: correctness vs reference, speed ratio, GFLOPS, timeline
  python bench.py

  # Single-kernel microscope (skips 20 warmup launches, profiles launch #21):
  ncu --set full --launch-skip 20 --launch-count 1 -o compute_report \\
      python bench.py --kernel-only

Then on your laptop:
  scp gpu:/workspace/gpu-learning/<folder>/compute_report.ncu-rep .   # open in NSight Compute
  scp gpu:/workspace/gpu-learning/<folder>/trace.json .               # open at ui.perfetto.dev
"""

import sys
import torch

from utils import compare, profile_timeline, gflops, matmul_flops

torch.manual_seed(0)  # fixed seed -> same input every run -> comparable numbers

# ---- EDIT 1: the fixed input -------------------------------------------------
N = 4096
x = torch.randn(N, N, device="cuda")


# ---- EDIT 2: the function under test -----------------------------------------
# Replace the body with your kernel call. For now it's a placeholder that uses
# PyTorch so the harness runs end to end before you've written anything.
def my_fn(a):
    return a @ a


# ---- EDIT 3: the trusted reference -------------------------------------------
# PyTorch's known-correct version of the same operation.
def ref_fn(a):
    return a @ a


if __name__ == "__main__":
    if "--kernel-only" in sys.argv:
        # Lean loop for ncu to profile: just the kernel, launched many times.
        # ncu's --launch-skip walks past these warmups to a steady-state launch.
        for _ in range(40):
            my_fn(x)
        torch.cuda.synchronize()
    else:
        # Full pass.
        t_mine, t_ref = compare(my_fn, ref_fn, x)

        # GFLOPS (edit matmul_flops if your op isn't a square matmul).
        flops = matmul_flops(N, N, N)
        print(
            f"[gflops]  mine: {gflops(flops, t_mine):.0f} GFLOPS   "
            f"reference: {gflops(flops, t_ref):.0f} GFLOPS"
        )

        # Timeline: where is the time going?
        profile_timeline(my_fn, x)

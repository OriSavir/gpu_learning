#include <stdio.h>
#include <cuda_runtime.h>

__global__
void addVecKernel(float *A, float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}


void vecAdd(float *A, float *B, float *C, int n) {
    // part 1: allocate memory on the device using cudaMalloc
    int size = n * sizeof(float);
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // part 3a: copy data from host to device using cudaMemcpy
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // part 2: launch kernel on device, grid of threads to perform addition
    int threads = 256;
    int blocks = ceil(n / (float)threads);
    addVecKernel<<<blocks, threads>>>(d_A, d_B, d_C, n);

    // part 3b: copy data from device to host using cudaMemcpy
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main () {
    int n = 100;
    float *A = (float*)malloc(n * sizeof(float));
    float *B = (float*)malloc(n * sizeof(float));
    float *C = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        A[i] = i;
        B[i] = i * 2;
    }

    vecAdd(A, B, C, n);
    for (int i = 0; i < n; i++) {
        printf("%f + %f = %f\n", A[i], B[i], C[i]);
    }
    free(A);
    free(B);
    free(C);
    return 0;
}
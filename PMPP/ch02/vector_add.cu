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
    vecAddKernel<<ceil(n/256.0), 256>>>(d_A, d_B, d_C, n);

    // part 3b: copy data from device to host using cudaMemcpy
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
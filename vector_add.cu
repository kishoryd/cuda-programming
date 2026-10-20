#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); return 1; } } while (0)

__global__ void vectorAdd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int n = 1 << 20;
    size_t bytes = n * sizeof(float);
    float *a, *b, *c;
    CUDA_CHECK(cudaMallocManaged(&a, bytes));
    CUDA_CHECK(cudaMallocManaged(&b, bytes));
    CUDA_CHECK(cudaMallocManaged(&c, bytes));
    for (int i = 0; i < n; i++) { a[i] = 1.0f; b[i] = 2.0f; }

    int threads = 256, blocks = (n + threads - 1) / threads;
    vectorAdd<<<blocks, threads>>>(a, b, c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < n; i++) if (c[i] != 3.0f) { printf("FAIL at %d\n", i); return 1; }
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    printf("PASS: %d elements on %s (sm_%d%d)\n", n, p.name, p.major, p.minor);
    cudaFree(a); cudaFree(b); cudaFree(c);
}

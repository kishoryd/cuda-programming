#include <cuda_runtime.h>
#include <cstdio>

#define BLOCK 256

// every thread updates the same counters: count of even values, max value, and a bitwise OR
__global__ void AtomicOps(const int* in, int* count, int* maxVal, unsigned int* orAll, int N){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N){
        int v = in[i];
        if (v % 2 == 0){
            atomicAdd(count, 1);
        }
        atomicMax(maxVal, v);
        atomicOr(orAll, (unsigned int)v);
    }
}

// atomicCAS loop: build a float atomic min out of compare-and-swap
__device__ void AtomicMinFloat(float* addr, float val){
    int* a = (int*)addr;
    int old = *a;
    while (val < __int_as_float(old)){
        int assumed = old;
        old = atomicCAS(a, assumed, __float_as_int(val));
        if (old == assumed) break;
    }
}

__global__ void MinKernel(const float* in, float* out, int N){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N){
        AtomicMinFloat(out, in[i]);
    }
}

int main(){
    const int N = 1 << 20;

    int* h_in = nullptr;
    float* h_f = nullptr;
    cudaMallocHost(&h_in, N * sizeof(int));
    cudaMallocHost(&h_f, N * sizeof(float));
    for (int i = 0; i < N; i++){
        h_in[i] = i;
        h_f[i] = (float)(i + 5);   // smallest value is 5
    }

    int *d_in, *d_count, *d_max;
    unsigned int* d_or;
    float *d_f, *d_min;
    cudaMalloc(&d_in, N * sizeof(int));
    cudaMalloc(&d_count, sizeof(int));
    cudaMalloc(&d_max, sizeof(int));
    cudaMalloc(&d_or, sizeof(unsigned int));
    cudaMalloc(&d_f, N * sizeof(float));
    cudaMalloc(&d_min, sizeof(float));

    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f, h_f, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_count, 0, sizeof(int));
    cudaMemset(d_max, 0, sizeof(int));
    cudaMemset(d_or, 0, sizeof(unsigned int));
    float big = 1e30f;
    cudaMemcpy(d_min, &big, sizeof(float), cudaMemcpyHostToDevice);

    int grid = (N + BLOCK - 1) / BLOCK;
    AtomicOps<<<grid, BLOCK>>>(d_in, d_count, d_max, d_or, N);
    MinKernel<<<grid, BLOCK>>>(d_f, d_min, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("Kernel launch error: %s \n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    int h_count = 0, h_max = 0;
    unsigned int h_or = 0;
    float h_min = 0.0f;
    cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_max, d_max, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_or, d_or, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_min, d_min, sizeof(float), cudaMemcpyDeviceToHost);

    printf("even count = %d (expected %d)\n", h_count, N / 2);
    printf("max = %d (expected %d)\n", h_max, N - 1);
    printf("or = %u (expected %u)\n", h_or, (unsigned int)(N - 1));
    printf("min = %.1f (expected %.1f)\n", h_min, 5.0f);

    cudaFreeHost(h_in);
    cudaFreeHost(h_f);
    cudaFree(d_in);
    cudaFree(d_count);
    cudaFree(d_max);
    cudaFree(d_or);
    cudaFree(d_f);
    cudaFree(d_min);
    return 0;
}

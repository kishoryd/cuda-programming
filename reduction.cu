#include <cuda_runtime.h>
#include <cstdio>

#define BLOCK 256

// each block adds up BLOCK elements and atomically adds its partial sum to *out
__global__ void SumReduce(const float* in, float* out, int N){
    __shared__ float s[BLOCK];

    int tid = threadIdx.x;
    int i = tid + blockDim.x * blockIdx.x;

    s[tid] = (i < N) ? in[i] : 0.0f;
    __syncthreads();

    // tree reduction: halve the number of active threads each step
    for (int stride = BLOCK / 2; stride > 0; stride >>= 1){
        if (tid < stride){
            s[tid] += s[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0){
        atomicAdd(out, s[0]);
    }
}

int main(){
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float* h_in = nullptr;
    cudaMallocHost(&h_in, bytes);
    for (int i = 0; i < N; i++){
        h_in[i] = 1.0f;
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, sizeof(float));
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, sizeof(float));

    int grid = (N + BLOCK - 1) / BLOCK;
    SumReduce<<<grid, BLOCK>>>(d_in, d_out, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("Kernel launch error: %s \n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    float h_out = 0.0f;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("sum = %.1f (expected %d)\n", h_out, N);

    cudaFreeHost(h_in);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

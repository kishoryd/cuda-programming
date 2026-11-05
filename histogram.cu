#include <cuda_runtime.h>
#include <cstdio>

#define BINS 256
#define BLOCK 256

// each block builds a private histogram in shared memory, then merges it into the global one
__global__ void Histogram(const unsigned char* data, unsigned int* hist, int N){
    __shared__ unsigned int local[BINS];

    int tid = threadIdx.x;
    local[tid] = 0;                     // BLOCK == BINS, so one bin per thread
    __syncthreads();

    // grid-stride loop: each thread handles several elements
    int stride = blockDim.x * gridDim.x;
    for (int i = tid + blockDim.x * blockIdx.x; i < N; i += stride){
        atomicAdd(&local[data[i]], 1u);
    }
    __syncthreads();

    atomicAdd(&hist[tid], local[tid]);
}

int main(){
    const int N = 1 << 24;
    unsigned char* h_data = nullptr;
    unsigned int h_hist[BINS];

    cudaMallocHost(&h_data, N);
    for (int i = 0; i < N; i++){
        h_data[i] = i % BINS;           // every value appears exactly N / BINS times
    }

    unsigned char* d_data;
    unsigned int* d_hist;
    cudaMalloc(&d_data, N);
    cudaMalloc(&d_hist, BINS * sizeof(unsigned int));
    cudaMemcpy(d_data, h_data, N, cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, BINS * sizeof(unsigned int));

    Histogram<<<64, BLOCK>>>(d_data, d_hist, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("Kernel launch error: %s \n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_hist, d_hist, BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    int bad = 0;
    for (int b = 0; b < BINS; b++){
        if (h_hist[b] != (unsigned int)(N / BINS)) bad++;
    }
    printf("hist[0] = %u (expected %d), %d of %d bins wrong\n", h_hist[0], N / BINS, bad, BINS);

    cudaFreeHost(h_data);
    cudaFree(d_data);
    cudaFree(d_hist);
    return 0;
}

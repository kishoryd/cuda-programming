#include <cuda_runtime.h>
#include <cstdio>

#define BLOCK 1024

// inclusive scan of BLOCK elements in one block (Hillis-Steele)
__global__ void PrefixScan(const int* in, int* out, int N){
    __shared__ int s[BLOCK];

    int tid = threadIdx.x;
    s[tid] = (tid < N) ? in[tid] : 0;
    __syncthreads();

    // each step adds the value `offset` positions to the left
    for (int offset = 1; offset < BLOCK; offset <<= 1){
        int val = 0;
        if (tid >= offset){
            val = s[tid - offset];
        }
        __syncthreads();
        s[tid] += val;
        __syncthreads();
    }

    if (tid < N){
        out[tid] = s[tid];
    }
}

int main(){
    const int N = BLOCK;
    size_t bytes = N * sizeof(int);

    int* h_in = nullptr;
    int* h_out = nullptr;
    cudaMallocHost(&h_in, bytes);
    cudaMallocHost(&h_out, bytes);
    for (int i = 0; i < N; i++){
        h_in[i] = 1;
    }

    int *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    PrefixScan<<<1, BLOCK>>>(d_in, d_out, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("Kernel launch error: %s \n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    printf("scan[%d] = %d (expected %d)\n", N - 1, h_out[N - 1], N);

    cudaFreeHost(h_in);
    cudaFreeHost(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

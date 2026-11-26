#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define N 1024

// single-block LSD radix sort, 1 bit per pass, using a scan to find output positions
__global__ void RadixSort(unsigned int* data){
    __shared__ unsigned int keys[N];
    __shared__ unsigned int tmp[N];
    __shared__ unsigned int scan[N];

    int tid = threadIdx.x;
    keys[tid] = data[tid];
    __syncthreads();

    for (int bit = 0; bit < 32; bit++){
        unsigned int k = keys[tid];
        unsigned int zero = ((k >> bit) & 1u) ? 0u : 1u;

        // inclusive scan (Hillis-Steele) of the "bit is zero" flags
        scan[tid] = zero;
        __syncthreads();
        for (int offset = 1; offset < N; offset <<= 1){
            unsigned int v = (tid >= offset) ? scan[tid - offset] : 0u;
            __syncthreads();
            scan[tid] += v;
            __syncthreads();
        }

        unsigned int totalZeros = scan[N - 1];
        // zeros go first (stable), ones go after them
        unsigned int pos = zero ? (scan[tid] - 1) : (totalZeros + tid - scan[tid]);
        tmp[pos] = k;
        __syncthreads();

        keys[tid] = tmp[tid];
        __syncthreads();
    }

    data[tid] = keys[tid];
}

int main(){
    size_t bytes = N * sizeof(unsigned int);

    unsigned int* h = (unsigned int*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; i++){
        h[i] = ((unsigned int)rand() << 16) ^ (unsigned int)rand();
    }

    unsigned int* d;
    cudaMalloc(&d, bytes);
    cudaMemcpy(d, h, bytes, cudaMemcpyHostToDevice);

    RadixSort<<<1, N>>>(d);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("Kernel launch error: %s \n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h, d, bytes, cudaMemcpyDeviceToHost);

    // check the output is in non-decreasing order
    int unsortedPairs = 0;
    for (int i = 1; i < N; i++){
        if (h[i - 1] > h[i]) unsortedPairs++;
    }
    printf("unsorted pairs = %d (expected 0)\n", unsortedPairs);

    free(h);
    cudaFree(d);
    return 0;
}

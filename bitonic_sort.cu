#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// one compare-and-swap step of the bitonic network
// k = size of the bitonic sequence being merged, j = distance between partners
__global__ void BitonicStep(int* data, int j, int k){
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int partner = i ^ j;

    // only the lower index of each pair does the swap
    if (partner > i){
        bool ascending = ((i & k) == 0);
        int a = data[i];
        int b = data[partner];
        if ((a > b) == ascending){
            data[i] = b;
            data[partner] = a;
        }
    }
}

int main(){
    const int N = 1 << 20;   // must be a power of two
    size_t bytes = N * sizeof(int);

    int* h_data = (int*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; i++){
        h_data[i] = rand() % 1000000;
    }

    int* d_data;
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice);

    int block = 256;
    int grid = N / block;

    // build bigger and bigger bitonic sequences, then merge them
    for (int k = 2; k <= N; k <<= 1){
        for (int j = k >> 1; j > 0; j >>= 1){
            BitonicStep<<<grid, block>>>(d_data, j, k);
        }
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("Kernel launch error: %s \n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);

    // count adjacent pairs that are out of order
    int bad = 0;
    for (int i = 1; i < N; i++){
        if (h_data[i - 1] > h_data[i]) bad++;
    }
    printf("out-of-order pairs = %d (expected 0)\n", bad);

    free(h_data);
    cudaFree(d_data);
    return 0;
}

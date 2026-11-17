#include <cuda_runtime.h>
#include <cstdio>

#define NSTREAMS 4
#define BLOCK 256

__global__ void VectorAdd(const float* A, const float* B, float* C, int n){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n){
        C[i] = A[i] + B[i];
    }
}

int main(){
    const int N = 1 << 20;
    const int chunk = N / NSTREAMS;
    size_t bytes = N * sizeof(float);
    size_t chunkBytes = chunk * sizeof(float);

    // pinned host memory is required for async copies to overlap
    float *h_A, *h_B, *h_C;
    cudaMallocHost(&h_A, bytes);
    cudaMallocHost(&h_B, bytes);
    cudaMallocHost(&h_C, bytes);
    for (int i = 0; i < N; i++){
        h_A[i] = float(i);
        h_B[i] = float(2 * i);
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaStream_t streams[NSTREAMS];
    for (int s = 0; s < NSTREAMS; s++){
        cudaStreamCreate(&streams[s]);
    }

    // each stream copies in, computes, and copies out its own chunk,
    // so one chunk's copy can overlap with another chunk's kernel
    int grid = (chunk + BLOCK - 1) / BLOCK;
    for (int s = 0; s < NSTREAMS; s++){
        int off = s * chunk;
        cudaMemcpyAsync(d_A + off, h_A + off, chunkBytes, cudaMemcpyHostToDevice, streams[s]);
        cudaMemcpyAsync(d_B + off, h_B + off, chunkBytes, cudaMemcpyHostToDevice, streams[s]);
        VectorAdd<<<grid, BLOCK, 0, streams[s]>>>(d_A + off, d_B + off, d_C + off, chunk);
        cudaMemcpyAsync(h_C + off, d_C + off, chunkBytes, cudaMemcpyDeviceToHost, streams[s]);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("Kernel launch error: %s \n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    // C[i] should be i + 2i = 3i
    int wrong = 0;
    for (int i = 0; i < N; i++){
        if (h_C[i] != float(3 * i)) wrong++;
    }
    printf("C[%d] = %.1f (expected %.1f)\n", N - 1, h_C[N - 1], float(3 * (N - 1)));
    printf("mismatches = %d (expected 0)\n", wrong);

    for (int s = 0; s < NSTREAMS; s++){
        cudaStreamDestroy(streams[s]);
    }
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}

#include <cuda_runtime.h>
#include <cstdio>

#define BLOCK 256
#define WARP 32

// sum a value across the 32 lanes of a warp using register shuffles
__device__ float warpReduce(float v){
    for (int offset = WARP / 2; offset > 0; offset >>= 1){
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

// each warp reduces with shuffles, warp sums go through shared memory,
// then the first warp reduces them and the block atomically adds to *out
__global__ void SumReduce(const float* in, float* out, int N){
    __shared__ float warpSums[BLOCK / WARP];

    int tid = threadIdx.x;
    int i = tid + blockDim.x * blockIdx.x;
    int lane = tid % WARP;
    int warp = tid / WARP;

    float v = (i < N) ? in[i] : 0.0f;
    v = warpReduce(v);

    if (lane == 0){
        warpSums[warp] = v;
    }
    __syncthreads();

    if (warp == 0){
        v = (lane < BLOCK / WARP) ? warpSums[lane] : 0.0f;
        v = warpReduce(v);
        if (lane == 0){
            atomicAdd(out, v);
        }
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

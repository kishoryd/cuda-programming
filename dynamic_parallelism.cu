#include <cuda_runtime.h>
#include <cstdio>

#define NSEG 16
#define SEG_UNIT 1024
#define CHILD_BLOCK 256

// child grid: fill one segment with its label
__global__ void FillSegment(int* out, int offset, int len, int label){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < len){
        out[offset + i] = label;
    }
}

// parent grid: one thread per segment, each segment has a different length,
// so each thread sizes and launches its own child grid from the device
__global__ void Parent(int* out){
    int s = threadIdx.x;
    if (s >= NSEG) return;

    int len = (s + 1) * SEG_UNIT;
    int offset = SEG_UNIT * s * (s + 1) / 2;
    int blocks = (len + CHILD_BLOCK - 1) / CHILD_BLOCK;

    // own stream per thread so the children can run concurrently
    // (launches into the default stream from one block are serialized)
    cudaStream_t st;
    cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
    FillSegment<<<blocks, CHILD_BLOCK, 0, st>>>(out, offset, len, s + 1);
    cudaStreamDestroy(st);
    // the parent grid is not complete until all its children finish,
    // so the host sync below waits for them too
}

int main(){
    const int N = SEG_UNIT * NSEG * (NSEG + 1) / 2;
    size_t bytes = N * sizeof(int);

    int* h_out = nullptr;
    cudaMallocHost(&h_out, bytes);

    int* d_out;
    cudaMalloc(&d_out, bytes);
    cudaMemset(d_out, 0, bytes);

    Parent<<<1, NSEG>>>(d_out);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("Kernel launch error: %s \n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // segment s holds (s+1)*SEG_UNIT copies of the value s+1
    long long sum = 0, expected = 0;
    int mismatches = 0, idx = 0;
    for (int s = 0; s < NSEG; s++){
        int len = (s + 1) * SEG_UNIT;
        expected += (long long)len * (s + 1);
        for (int i = 0; i < len; i++, idx++){
            sum += h_out[idx];
            if (h_out[idx] != s + 1) mismatches++;
        }
    }
    printf("elements = %d, sum = %lld (expected %lld), mismatches = %d (expected 0)\n",
           N, sum, expected, mismatches);

    cudaFreeHost(h_out);
    cudaFree(d_out);
    return 0;
}

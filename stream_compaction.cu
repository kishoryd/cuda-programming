#include <cuda_runtime.h>
#include <cstdio>

#define BLOCK 256

// step 1: flag the elements to keep (even values) and scan the flags inside each block
__global__ void BlockScan(const int* in, int* scan, int* sums, int N){
    __shared__ int s[BLOCK];

    int tid = threadIdx.x;
    int i = tid + blockDim.x * blockIdx.x;

    s[tid] = (i < N && in[i] % 2 == 0) ? 1 : 0;
    __syncthreads();

    // inclusive scan (Hillis-Steele)
    for (int offset = 1; offset < BLOCK; offset <<= 1){
        int v = (tid >= offset) ? s[tid - offset] : 0;
        __syncthreads();
        s[tid] += v;
        __syncthreads();
    }

    if (i < N) scan[i] = s[tid];
    if (tid == BLOCK - 1) sums[blockIdx.x] = s[tid];
}

// step 2: one block turns the per-block counts into per-block start offsets
__global__ void ScanSums(int* sums, int numBlocks){
    __shared__ int s[BLOCK];

    int tid = threadIdx.x;
    int orig = (tid < numBlocks) ? sums[tid] : 0;
    s[tid] = orig;
    __syncthreads();

    for (int offset = 1; offset < BLOCK; offset <<= 1){
        int v = (tid >= offset) ? s[tid - offset] : 0;
        __syncthreads();
        s[tid] += v;
        __syncthreads();
    }

    // inclusive -> exclusive
    if (tid < numBlocks) sums[tid] = s[tid] - orig;
}

// step 3: scatter each kept element to its compacted position
__global__ void Scatter(const int* in, const int* scan, const int* sums, int* out, int N){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N && in[i] % 2 == 0){
        out[sums[blockIdx.x] + scan[i] - 1] = in[i];
    }
}

int main(){
    const int N = 1 << 16;           // 256 blocks, fits the single-block sums scan
    size_t bytes = N * sizeof(int);
    int grid = (N + BLOCK - 1) / BLOCK;

    int* h_in = (int*)malloc(bytes);
    int* h_out = (int*)malloc(bytes);
    int expected = 0;
    for (int i = 0; i < N; i++){
        h_in[i] = i;
        if (h_in[i] % 2 == 0) expected++;
    }

    int *d_in, *d_scan, *d_sums, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_scan, bytes);
    cudaMalloc(&d_sums, grid * sizeof(int));
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    BlockScan<<<grid, BLOCK>>>(d_in, d_scan, d_sums, N);
    ScanSums<<<1, BLOCK>>>(d_sums, grid);
    Scatter<<<grid, BLOCK>>>(d_in, d_scan, d_sums, d_out, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("Kernel launch error: %s \n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    // total kept = start offset of the last block + its inclusive scan at the last element
    int lastOffset = 0, lastScan = 0;
    cudaMemcpy(&lastOffset, d_sums + grid - 1, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&lastScan, d_scan + N - 1, sizeof(int), cudaMemcpyDeviceToHost);
    int kept = lastOffset + lastScan;

    cudaMemcpy(h_out, d_out, expected * sizeof(int), cudaMemcpyDeviceToHost);

    // output must be 0, 2, 4, ... in order
    int bad = 0;
    for (int i = 0; i < expected; i++){
        if (h_out[i] != 2 * i) bad++;
    }
    printf("kept = %d (expected %d), out-of-place = %d (expected 0)\n", kept, expected, bad);

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_scan);
    cudaFree(d_sums);
    cudaFree(d_out);
    return 0;
}

#include <cuda_runtime.h>
#include <cstdio>

#define TILE 32
#define REPS 256

// a warp is one row of the block (same threadIdx.y, threadIdx.x = 0..31).
// reading tile[row][threadIdx.y] with a different row per thread walks down a column.
// PAD = 0: row stride is 32 floats, so every thread hits the same bank (32-way conflict)
// PAD = 1: row stride is 33 floats, so the 32 threads land in 32 different banks
template <int PAD>
__global__ void ColumnRead(const float* in, float* out, int N){
    __shared__ float tile[TILE][TILE + PAD];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    // row-wise store: consecutive threads, consecutive banks, no conflict
    tile[threadIdx.y][threadIdx.x] = in[y * N + x];
    __syncthreads();

    // column-wise reads, repeated so the conflict cost shows up in the timing
    float sum = 0.0f;
    for (int r = 0; r < REPS; r++){
        int row = (threadIdx.x + r) % TILE;
        sum += tile[row][threadIdx.y];
    }
    out[y * N + x] = sum;
}

// every input is 1.0f, so every output should be REPS
int CountMismatches(const float* h, int count){
    int errors = 0;
    for (int i = 0; i < count; i++){
        if (h[i] != (float)REPS){
            errors++;
        }
    }
    return errors;
}

int main(){
    const int N = 2048;
    size_t bytes = N * N * sizeof(float);

    float* h_in = nullptr;
    float* h_out = nullptr;
    cudaMallocHost(&h_in, bytes);
    cudaMallocHost(&h_out, bytes);
    for (int i = 0; i < N * N; i++){
        h_in[i] = 1.0f;
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid(N / TILE, N / TILE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms_conflict = 0.0f, ms_padded = 0.0f;

    // warm-up launch so the first timed kernel doesn't pay startup cost
    ColumnRead<1><<<grid, block>>>(d_in, d_out, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("Kernel launch error: %s \n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    // unpadded tile: 32-way bank conflicts
    cudaMemset(d_out, 0, bytes);
    cudaEventRecord(start);
    ColumnRead<0><<<grid, block>>>(d_in, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_conflict, start, stop);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    int err_conflict = CountMismatches(h_out, N * N);

    // padded tile: conflict-free
    cudaMemset(d_out, 0, bytes);
    cudaEventRecord(start);
    ColumnRead<1><<<grid, block>>>(d_in, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_padded, start, stop);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    int err_padded = CountMismatches(h_out, N * N);

    printf("out[0] = %.1f (expected %d)\n", h_out[0], REPS);
    printf("conflict kernel mismatches = %d (expected 0)\n", err_conflict);
    printf("padded kernel mismatches   = %d (expected 0)\n", err_padded);
    printf("tile[32][32] (conflicts): %.3f ms\n", ms_conflict);
    printf("tile[32][33] (padded):    %.3f ms\n", ms_padded);
    printf("speedup from padding = %.2fx (expected > 1)\n", ms_conflict / ms_padded);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

#include <cuda_runtime.h>
#include <cstdio>

#define BLOCK 256
#define MAX_ROW 64

// sparse matrix-vector multiply y = A * x, with A stored in CSR form:
//   rowPtr[r] .. rowPtr[r+1]-1 are the entries of row r
//   colIdx[k] is the column of entry k, val[k] is its value

// scalar kernel: one thread per row.
// neighbouring threads read from different rows, so the loads are scattered.
__global__ void SpmvScalar(const int* rowPtr, const int* colIdx, const float* val,
                           const float* x, float* y, int N){
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    if (row < N){
        float sum = 0.0f;
        for (int k = rowPtr[row]; k < rowPtr[row + 1]; k++){
            sum += val[k] * x[colIdx[k]];
        }
        y[row] = sum;
    }
}

// vector kernel: one warp per row.
// the 32 lanes walk the row together (coalesced loads), then a shuffle reduction
// adds the 32 partial sums without touching shared memory.
__global__ void SpmvVector(const int* rowPtr, const int* colIdx, const float* val,
                           const float* x, float* y, int N){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int row = tid / 32;
    int lane = threadIdx.x & 31;

    // row is the same for the whole warp, so the warp stays converged
    if (row < N){
        float sum = 0.0f;
        for (int k = rowPtr[row] + lane; k < rowPtr[row + 1]; k += 32){
            sum += val[k] * x[colIdx[k]];
        }

        for (int offset = 16; offset > 0; offset >>= 1){
            sum += __shfl_down_sync(0xffffffff, sum, offset);
        }

        if (lane == 0){
            y[row] = sum;
        }
    }
}

int CountMismatches(const float* got, const float* expected, int N){
    int errors = 0;
    for (int i = 0; i < N; i++){
        if (got[i] != expected[i]){
            errors++;
        }
    }
    return errors;
}

int main(){
    const int N = 1 << 16;

    // row r gets (r % MAX_ROW) + 1 entries, so row lengths vary from 1 to MAX_ROW
    int* h_rowPtr = nullptr;
    cudaMallocHost(&h_rowPtr, (N + 1) * sizeof(int));
    h_rowPtr[0] = 0;
    for (int r = 0; r < N; r++){
        h_rowPtr[r + 1] = h_rowPtr[r] + (r % MAX_ROW) + 1;
    }
    int nnz = h_rowPtr[N];

    int* h_colIdx = nullptr;
    float* h_val = nullptr;
    float* h_x = nullptr;
    float* h_y = nullptr;
    float* h_ref = nullptr;
    cudaMallocHost(&h_colIdx, nnz * sizeof(int));
    cudaMallocHost(&h_val, nnz * sizeof(float));
    cudaMallocHost(&h_x, N * sizeof(float));
    cudaMallocHost(&h_y, N * sizeof(float));
    cudaMallocHost(&h_ref, N * sizeof(float));

    // columns spaced 7 apart so each row's columns are distinct and spread out.
    // small integer values keep every sum exact in float.
    for (int r = 0; r < N; r++){
        for (int k = h_rowPtr[r]; k < h_rowPtr[r + 1]; k++){
            int j = k - h_rowPtr[r];
            h_colIdx[k] = (r + j * 7) % N;
            h_val[k] = (float)(1 + (r + j) % 3);
        }
    }
    for (int i = 0; i < N; i++){
        h_x[i] = (float)(1 + i % 5);
    }

    // CPU reference
    for (int r = 0; r < N; r++){
        float sum = 0.0f;
        for (int k = h_rowPtr[r]; k < h_rowPtr[r + 1]; k++){
            sum += h_val[k] * h_x[h_colIdx[k]];
        }
        h_ref[r] = sum;
    }

    int *d_rowPtr, *d_colIdx;
    float *d_val, *d_x, *d_y;
    cudaMalloc(&d_rowPtr, (N + 1) * sizeof(int));
    cudaMalloc(&d_colIdx, nnz * sizeof(int));
    cudaMalloc(&d_val, nnz * sizeof(float));
    cudaMalloc(&d_x, N * sizeof(float));
    cudaMalloc(&d_y, N * sizeof(float));
    cudaMemcpy(d_rowPtr, h_rowPtr, (N + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, h_colIdx, nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_val, h_val, nnz * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice);

    int gridScalar = (N + BLOCK - 1) / BLOCK;
    int gridVector = (N * 32 + BLOCK - 1) / BLOCK;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms_scalar = 0.0f, ms_vector = 0.0f;

    // warm-up launch so the first timed kernel doesn't pay startup cost
    SpmvScalar<<<gridScalar, BLOCK>>>(d_rowPtr, d_colIdx, d_val, d_x, d_y, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("Kernel launch error: %s \n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    // thread per row
    cudaMemset(d_y, 0, N * sizeof(float));
    cudaEventRecord(start);
    SpmvScalar<<<gridScalar, BLOCK>>>(d_rowPtr, d_colIdx, d_val, d_x, d_y, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_scalar, start, stop);

    cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    int err_scalar = CountMismatches(h_y, h_ref, N);

    // warp per row
    cudaMemset(d_y, 0, N * sizeof(float));
    cudaEventRecord(start);
    SpmvVector<<<gridVector, BLOCK>>>(d_rowPtr, d_colIdx, d_val, d_x, d_y, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_vector, start, stop);

    cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost);
    int err_vector = CountMismatches(h_y, h_ref, N);

    int last = N - 1;
    printf("rows = %d, nonzeros = %d\n", N, nnz);
    printf("y[%d] = %.1f (expected %.1f)\n", last, h_y[last], h_ref[last]);
    printf("scalar kernel mismatches = %d (expected 0)\n", err_scalar);
    printf("vector kernel mismatches = %d (expected 0)\n", err_vector);
    printf("thread per row: %.3f ms\n", ms_scalar);
    printf("warp per row:   %.3f ms\n", ms_vector);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFreeHost(h_rowPtr);
    cudaFreeHost(h_colIdx);
    cudaFreeHost(h_val);
    cudaFreeHost(h_x);
    cudaFreeHost(h_y);
    cudaFreeHost(h_ref);
    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_val);
    cudaFree(d_x);
    cudaFree(d_y);
    return 0;
}

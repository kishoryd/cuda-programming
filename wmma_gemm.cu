#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>

using namespace nvcuda;

#define M 64
#define N 64
#define K 64

// each warp computes one 16x16 tile of C using tensor cores (half in, float accumulate)
__global__ void WmmaGemm(const half* A, const half* B, float* C){
    int tileRow = blockIdx.y;
    int tileCol = blockIdx.x;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);

    // walk along K one 16-wide slice at a time
    for (int k = 0; k < K; k += 16){
        wmma::load_matrix_sync(a, A + tileRow * 16 * K + k, K);
        wmma::load_matrix_sync(b, B + k * N + tileCol * 16, N);
        wmma::mma_sync(acc, a, b, acc);
    }

    wmma::store_matrix_sync(C + tileRow * 16 * N + tileCol * 16, acc, N, wmma::mem_row_major);
}

int main(){
    half* h_A = new half[M * K];
    half* h_B = new half[K * N];
    float* h_C = new float[M * N];

    // A = 1.0, B = 2.0 so every C element is K * 1 * 2
    for (int i = 0; i < M * K; i++) h_A[i] = __float2half(1.0f);
    for (int i = 0; i < K * N; i++) h_B[i] = __float2half(2.0f);

    half *d_A, *d_B;
    float* d_C;
    cudaMalloc(&d_A, M * K * sizeof(half));
    cudaMalloc(&d_B, K * N * sizeof(half));
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMemcpy(d_A, h_A, M * K * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(half), cudaMemcpyHostToDevice);

    // one warp (32 threads) per block, one block per output tile
    dim3 grid(N / 16, M / 16);
    WmmaGemm<<<grid, 32>>>(d_A, d_B, d_C);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("Kernel launch error: %s \n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    float expected = 2.0f * K;
    int bad = 0;
    for (int i = 0; i < M * N; i++){
        if (h_C[i] != expected) bad++;
    }
    printf("C[0] = %.1f, C[last] = %.1f (expected %.1f), mismatches = %d (expected 0)\n",
           h_C[0], h_C[M * N - 1], expected, bad);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}

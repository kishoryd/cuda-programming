#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cmath>

int main(){
    const int N = 256;
    size_t bytes = N * N * sizeof(float);

    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);
    for (int i = 0; i < N * N; i++){
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    // cuBLAS is column-major, so compute C^T = B^T * A^T to get row-major C = A * B
    const float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                &alpha, d_B, N, d_A, N, &beta, d_C, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // every entry is sum over N of 1 * 2
    float expected = 2.0f * N;
    float maxErr = 0.0f;
    for (int i = 0; i < N * N; i++){
        maxErr = fmaxf(maxErr, fabsf(h_C[i] - expected));
    }
    printf("C[0] = %.1f (expected %.1f), max error = %f\n", h_C[0], expected, maxErr);

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    return 0;
}

#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>
#define TILE 32

__global__ void TiledMul(const float* A, const float* B, float* C, int N){
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    float sum = 0.0f;

    for (int t=0; t < N; t += TILE){
        if (row < N && (t + threadIdx.x) < N){
            sA[threadIdx.y][threadIdx.x] = A [row * N + t + threadIdx.x];
        }else{
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }
        if ((t + threadIdx.y) < N && col < N){
            sB[threadIdx.y][threadIdx.x] = B [(threadIdx.y + t) * N + col];
        }else{
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        for (int k=0; k < TILE; k++){
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        __syncthreads();
    }

    if (row < N && col < N){
        C[row * N + col] = sum;
    }
}

int main(){
    const int N = 1024;
    float* h_A=nullptr;
    float* h_B=nullptr;
    float* h_C=nullptr;
    size_t bytes = N*N * sizeof(float);

    cudaMallocHost(&h_A, bytes);
    cudaMallocHost(&h_B, bytes);
    cudaMallocHost(&h_C, bytes);

    for(int i=0; i < N*N; i++){
        h_A[i]=1.0f;
        h_B[i]=2.0f;
    }
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cuaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A,h_A,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,bytes,cudaMemcpyHostToDevice);

    dim3 block(TILE,TILE);
    dim3 grid((N+TILE-1)/TILE, (N+TILE-1)/TILE);
    TiledMul<<<grid, block>>>(d_A,d_B,d_C,N);

    cudaError_t err = cudaGetLastError();

    if(err != cudaSuccess){
        printf("Kernel launch error: %s \n",cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_C,d_C,bytes,cudaMemcpyDeviceToHost);

    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    cudaFree(d_dA);
    cudaFree(dB);
    cudaFree(d_C);
    return 0;
}
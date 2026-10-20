#include <cuda_runtime.h>
#include <cstdio>

__global__ void VectorMul(const float* A, const float* B, float* C, int N){
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;

    if( row < N && col < N){
        float sum=0.0f;
        for(int i=0; i<N; i++){
            sum += A[row * N + i] * B[i * N + col];
        }
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
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A,h_A,bytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,bytes,cudaMemcpyHostToDevice);

    dim3 block(32,32);
    dim3 grid((N+31)/32, (N+31)/32);
    VectorMul<<<grid, block>>>(d_A,d_B,d_C,N);

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

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;



}

#include <cuda_runtime_api.h>
#include <cuda/cmath>


__global__ void vecAdd(float* A, float* B, float* C, int vectorlength){
    int workindex = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (workindex < vectorlength){
        A[workindex] = 1.0f;
        B[workindex] = 2.0f;
        C[workindex] = A[workindex] + B[workindex];
    }
}

void unified_mem(int vectorlength)
{
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    
    cudaMallocManaged(&A, vectorlength*sizeof(float));
    cudaMallocManaged(&B, vectorlength*sizeof(float));
    cudaMallocManaged(&C, vectorlength*sizeof(float));

    int threads = 256;
    int blocks = cuda::ceil_div(vectorlength, threads);
    vecAdd<<<blocks, threads>>>(A,B,C,vectorlength);
    cudaDeviceSynchronize();

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);


}
int main(){
    unified_mem(1000);
    return 0;
}
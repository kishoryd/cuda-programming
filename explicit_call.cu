#include <cuda_runtime.h>
#include <cstdio>
#include <cuda/cmath>


__global__ void vecAdd(int vectorlength, float* A, float* B, float* C)
{
    int index= threadIdx.x + blockDim.x * blockIdx.x;
    if (index < vectorlength){
        C[index] = A[index] + B[index];
    }
}

// void SerialAdd(float* A, float* B, float* C, int vectorlength){
//     for (int i=0; i < vectorlength; i++){
//         C[i] = A[i] + B[i];
//     }
// }
#define CUDA_CHECK(expr_check) do {
    cudaError_t result = expr_check;
    if(result != cudaSuccess){
        fprintf(stderr,"CUDA runtine error %s:%i,%d = %s\n", __FILE__,__LINE__,result,cudaGetErrorString(result));
    }
}while(0)

void explicit_call(int vectorlength)
{
    //pointers for host memory
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;
    
    //allocating memory for host (best practice)
    cudaMallocHost(&A, vectorlength*sizeof(float));
    cudaMallocHost(&B, vectorlength*sizeof(float));
    cudaMallocHost(&C, vectorlength*sizeof(float));

    for(int i=0; i < vectorlength; i++){
        A[i]=1.0f;
        B[i]=2.0f;
    }

    //pointers for device memory
    float* devA = nullptr;
    float* devB = nullptr;
    float* devC = nullptr;

    //allocate memory for device
    cudaMalloc(&devA, vectorlength*sizeof(float));
    cudaMalloc(&devB, vectorlength*sizeof(float));
    cudaMalloc(&devC, vectorlength*sizeof(float));

    cudaMemcpy(devA, A, vectorlength*sizeof(float), cudaMemcpyDefault);
    cudaMemcpy(devB, B, vectorlength*sizeof(float), cudaMemcpyDefault);

    int threads=256;
    int blocks = cuda::ceil_div(vectorlength,threads);
    vecAdd<<<blocks,threads>>>(vectorlength, devA, devB, devC);

    cudaDeviceSynchronize();

    cudaMemcpy(C,devC, vectorlength*sizeof(float), cudaMemcpyDefault);

    //SerialAdd(A, B, vectorlength);

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);
    cudaFreeHost(A);
    cudaFreeHost(B);
    cudaFreeHost(C);
}
int main(){
    explicit_call(1000);
}

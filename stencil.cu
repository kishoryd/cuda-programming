#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define BLOCK 256
#define RADIUS 3

// 1D stencil: out[i] = sum of in[i-RADIUS .. i+RADIUS], out-of-range counts as 0
__global__ void Stencil1D(const float* in, float* out, int N){
    __shared__ float s[BLOCK + 2 * RADIUS];

    int tid = threadIdx.x;
    int i = tid + blockDim.x * blockIdx.x;
    int ls = tid + RADIUS;

    // load the block's elements plus a halo on each side
    s[ls] = (i < N) ? in[i] : 0.0f;
    if (tid < RADIUS){
        int left = i - RADIUS;
        int right = i + BLOCK;
        s[ls - RADIUS] = (left >= 0) ? in[left] : 0.0f;
        s[ls + BLOCK] = (right < N) ? in[right] : 0.0f;
    }
    __syncthreads();

    if (i < N){
        float sum = 0.0f;
        for (int k = -RADIUS; k <= RADIUS; k++){
            sum += s[ls + k];
        }
        out[i] = sum;
    }
}

int main(){
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float* h_in = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    for (int i = 0; i < N; i++){
        h_in[i] = 1.0f;
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    int grid = (N + BLOCK - 1) / BLOCK;
    Stencil1D<<<grid, BLOCK>>>(d_in, d_out, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("Kernel launch error: %s \n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // each end loses (3+2+1) = 6 from the missing neighbours
    double sum = 0.0;
    for (int i = 0; i < N; i++){
        sum += h_out[i];
    }
    printf("out[0] = %.1f (expected 4.0)\n", h_out[0]);
    printf("out[N/2] = %.1f (expected 7.0)\n", h_out[N / 2]);
    printf("sum = %.1f (expected %.1f)\n", sum, 7.0 * N - 12.0);

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

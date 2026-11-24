#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define BLOCK 256

// max over a warp using shuffles
__device__ float warpMax(float v){
    for (int off = 16; off > 0; off >>= 1){
        v = fmaxf(v, __shfl_down_sync(0xffffffff, v, off));
    }
    return v;
}

// sum over a warp using shuffles
__device__ float warpSum(float v){
    for (int off = 16; off > 0; off >>= 1){
        v += __shfl_down_sync(0xffffffff, v, off);
    }
    return v;
}

// one block per row: numerically stable softmax (subtract row max)
__global__ void Softmax(const float* in, float* out, int cols){
    __shared__ float warp_part[BLOCK / 32];
    __shared__ float row_val;

    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp = tid >> 5;
    const float* x = in + (size_t)blockIdx.x * cols;
    float* y = out + (size_t)blockIdx.x * cols;

    // row max
    float m = -INFINITY;
    for (int j = tid; j < cols; j += BLOCK){
        m = fmaxf(m, x[j]);
    }
    m = warpMax(m);
    if (lane == 0) warp_part[warp] = m;
    __syncthreads();
    if (warp == 0){
        float v = (lane < BLOCK / 32) ? warp_part[lane] : -INFINITY;
        v = warpMax(v);
        if (lane == 0) row_val = v;
    }
    __syncthreads();
    m = row_val;
    __syncthreads();

    // row sum of exp(x - max)
    float s = 0.0f;
    for (int j = tid; j < cols; j += BLOCK){
        s += expf(x[j] - m);
    }
    s = warpSum(s);
    if (lane == 0) warp_part[warp] = s;
    __syncthreads();
    if (warp == 0){
        float v = (lane < BLOCK / 32) ? warp_part[lane] : 0.0f;
        v = warpSum(v);
        if (lane == 0) row_val = v;
    }
    __syncthreads();
    s = row_val;

    // normalize
    for (int j = tid; j < cols; j += BLOCK){
        y[j] = expf(x[j] - m) / s;
    }
}

int main(){
    const int rows = 1024;
    const int cols = 1000;
    size_t bytes = (size_t)rows * cols * sizeof(float);

    float* h_in = nullptr;
    float* h_out = nullptr;
    cudaMallocHost(&h_in, bytes);
    cudaMallocHost(&h_out, bytes);
    // large values would overflow a naive softmax
    for (int i = 0; i < rows * cols; i++){
        h_in[i] = 1000.0f + (i % 7);
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    Softmax<<<rows, BLOCK>>>(d_in, d_out, cols);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("Kernel launch error: %s \n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // row 0 should sum to 1; compare out[0] against a CPU result
    double sum0 = 0.0;
    for (int j = 0; j < cols; j++) sum0 += h_out[j];

    double ref_sum = 0.0;
    for (int j = 0; j < cols; j++) ref_sum += exp((double)(j % 7) - 6.0);
    double ref0 = exp(-6.0) / ref_sum;

    printf("row 0 sum = %.5f (expected 1.00000)\n", sum0);
    printf("out[0] = %.6e (expected %.6e)\n", h_out[0], ref0);

    cudaFreeHost(h_in);
    cudaFreeHost(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

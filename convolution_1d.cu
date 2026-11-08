#include <cuda_runtime.h>
#include <cstdio>

#define BLOCK 256
#define RADIUS 2
#define KSIZE (2 * RADIUS + 1)

// filter weights live in constant memory
__constant__ float c_filter[KSIZE];

// each thread computes one output; the input tile (plus halo) is staged in shared memory
__global__ void Conv1D(const float* in, float* out, int N){
    __shared__ float s[BLOCK + 2 * RADIUS];

    int tid = threadIdx.x;
    int i = tid + blockDim.x * blockIdx.x;

    // center element
    s[tid + RADIUS] = (i < N) ? in[i] : 0.0f;

    // left and right halo, zero padded outside the array
    if (tid < RADIUS){
        int left = i - RADIUS;
        s[tid] = (left >= 0) ? in[left] : 0.0f;

        int right = i + BLOCK;
        s[tid + BLOCK + RADIUS] = (right < N) ? in[right] : 0.0f;
    }
    __syncthreads();

    if (i < N){
        float sum = 0.0f;
        for (int k = 0; k < KSIZE; k++){
            sum += s[tid + k] * c_filter[k];
        }
        out[i] = sum;
    }
}

int main(){
    const int N = 1 << 20;
    size_t bytes = N * sizeof(float);

    float* h_in = nullptr;
    float* h_out = nullptr;
    cudaMallocHost(&h_in, bytes);
    cudaMallocHost(&h_out, bytes);
    for (int i = 0; i < N; i++){
        h_in[i] = 1.0f;
    }

    // box filter: every weight is 1, so interior outputs equal KSIZE
    float h_filter[KSIZE];
    for (int k = 0; k < KSIZE; k++){
        h_filter[k] = 1.0f;
    }
    cudaMemcpyToSymbol(c_filter, h_filter, sizeof(h_filter));

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    int grid = (N + BLOCK - 1) / BLOCK;
    Conv1D<<<grid, BLOCK>>>(d_in, d_out, N);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("Kernel launch error: %s \n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // interior = KSIZE, edges lose RADIUS taps to zero padding
    printf("out[0] = %.1f (expected %d)\n", h_out[0], RADIUS + 1);
    printf("out[N/2] = %.1f (expected %d)\n", h_out[N / 2], KSIZE);
    printf("out[N-1] = %.1f (expected %d)\n", h_out[N - 1], RADIUS + 1);

    cudaFreeHost(h_in);
    cudaFreeHost(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

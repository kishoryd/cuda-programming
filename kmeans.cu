#include <cuda_runtime.h>
#include <cstdio>

#define K 4
#define BLOCK 256

// assign each point to its nearest centroid, then accumulate sums with atomics
__global__ void AssignAndAccumulate(const float* x, const float* y, const float* cx, const float* cy,
                                    float* sumx, float* sumy, int* count, int N){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= N) return;

    int best = 0;
    float bestDist = 1e30f;
    for (int k = 0; k < K; k++){
        float dx = x[i] - cx[k];
        float dy = y[i] - cy[k];
        float d = dx * dx + dy * dy;
        if (d < bestDist){
            bestDist = d;
            best = k;
        }
    }

    atomicAdd(&sumx[best], x[i]);
    atomicAdd(&sumy[best], y[i]);
    atomicAdd(&count[best], 1);
}

// new centroid = mean of the points assigned to it
__global__ void UpdateCentroids(float* cx, float* cy, const float* sumx, const float* sumy, const int* count){
    int k = threadIdx.x;
    if (k < K && count[k] > 0){
        cx[k] = sumx[k] / count[k];
        cy[k] = sumy[k] / count[k];
    }
}

int main(){
    const int N = 1 << 16;
    size_t bytes = N * sizeof(float);

    float* h_x = new float[N];
    float* h_y = new float[N];
    // point i belongs to cluster i % K, centered at (10k, 10k) with a small symmetric offset
    for (int i = 0; i < N; i++){
        int k = i % K;
        float off = ((i / K) % 4 - 1.5f) * 0.1f;
        h_x[i] = 10.0f * k + off;
        h_y[i] = 10.0f * k - off;
    }

    // initial guess is the true center shifted by 1
    float h_cx[K], h_cy[K];
    for (int k = 0; k < K; k++){
        h_cx[k] = 10.0f * k + 1.0f;
        h_cy[k] = 10.0f * k + 1.0f;
    }

    float *d_x, *d_y, *d_cx, *d_cy, *d_sumx, *d_sumy;
    int* d_count;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMalloc(&d_cx, K * sizeof(float));
    cudaMalloc(&d_cy, K * sizeof(float));
    cudaMalloc(&d_sumx, K * sizeof(float));
    cudaMalloc(&d_sumy, K * sizeof(float));
    cudaMalloc(&d_count, K * sizeof(int));
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cx, h_cx, K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cy, h_cy, K * sizeof(float), cudaMemcpyHostToDevice);

    // a few iterations of Lloyd's algorithm
    int grid = (N + BLOCK - 1) / BLOCK;
    for (int iter = 0; iter < 3; iter++){
        cudaMemset(d_sumx, 0, K * sizeof(float));
        cudaMemset(d_sumy, 0, K * sizeof(float));
        cudaMemset(d_count, 0, K * sizeof(int));
        AssignAndAccumulate<<<grid, BLOCK>>>(d_x, d_y, d_cx, d_cy, d_sumx, d_sumy, d_count, N);
        UpdateCentroids<<<1, K>>>(d_cx, d_cy, d_sumx, d_sumy, d_count);
    }
    cudaDeviceSynchronize();

    int h_count[K];
    cudaMemcpy(h_cx, d_cx, K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cy, d_cy, K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_count, d_count, K * sizeof(int), cudaMemcpyDeviceToHost);

    for (int k = 0; k < K; k++){
        printf("cluster %d: centroid = (%.3f, %.3f) count = %d (expected (%.3f, %.3f) count %d)\n",
               k, h_cx[k], h_cy[k], h_count[k], 10.0f * k, 10.0f * k, N / K);
    }

    delete[] h_x;
    delete[] h_y;
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_cx);
    cudaFree(d_cy);
    cudaFree(d_sumx);
    cudaFree(d_sumy);
    cudaFree(d_count);
    return 0;
}

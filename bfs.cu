#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

#define BLOCK 256

// one thread per vertex: if it sits on the current level, label its unvisited neighbors
__global__ void BfsLevel(const int* rowPtr, const int* colIdx, int* dist, int level, int* changed, int V){
    int v = threadIdx.x + blockDim.x * blockIdx.x;
    if (v >= V || dist[v] != level) return;

    for (int e = rowPtr[v]; e < rowPtr[v + 1]; e++){
        int u = colIdx[e];
        // atomicCAS makes sure only one thread claims u
        if (atomicCAS(&dist[u], -1, level + 1) == -1){
            *changed = 1;
        }
    }
}

int main(){
    const int W = 512, H = 512;
    const int V = W * H;

    // build a W x H grid graph in CSR form
    std::vector<int> rowPtr(V + 1, 0), colIdx;
    for (int y = 0; y < H; y++){
        for (int x = 0; x < W; x++){
            int v = y * W + x;
            if (x > 0)     colIdx.push_back(v - 1);
            if (x < W - 1) colIdx.push_back(v + 1);
            if (y > 0)     colIdx.push_back(v - W);
            if (y < H - 1) colIdx.push_back(v + W);
            rowPtr[v + 1] = colIdx.size();
        }
    }
    int E = colIdx.size();

    std::vector<int> h_dist(V, -1);
    h_dist[0] = 0;  // source is vertex 0

    int *d_rowPtr, *d_colIdx, *d_dist, *d_changed;
    cudaMalloc(&d_rowPtr, (V + 1) * sizeof(int));
    cudaMalloc(&d_colIdx, E * sizeof(int));
    cudaMalloc(&d_dist, V * sizeof(int));
    cudaMalloc(&d_changed, sizeof(int));
    cudaMemcpy(d_rowPtr, rowPtr.data(), (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_colIdx, colIdx.data(), E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dist, h_dist.data(), V * sizeof(int), cudaMemcpyHostToDevice);

    int grid = (V + BLOCK - 1) / BLOCK;
    int level = 0;
    int changed = 1;

    // launch one kernel per level until no new vertex is found
    while (changed){
        changed = 0;
        cudaMemcpy(d_changed, &changed, sizeof(int), cudaMemcpyHostToDevice);
        BfsLevel<<<grid, BLOCK>>>(d_rowPtr, d_colIdx, d_dist, level, d_changed, V);
        cudaMemcpy(&changed, d_changed, sizeof(int), cudaMemcpyDeviceToHost);
        level++;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("Kernel launch error: %s \n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(h_dist.data(), d_dist, V * sizeof(int), cudaMemcpyDeviceToHost);
    printf("dist to far corner = %d (expected %d)\n", h_dist[V - 1], (W - 1) + (H - 1));

    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_dist);
    cudaFree(d_changed);
    return 0;
}

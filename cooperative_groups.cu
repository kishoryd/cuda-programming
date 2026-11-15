#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>

namespace cg = cooperative_groups;

#define BLOCK 256

// add up the 32 values held by a warp-sized tile using shuffles
__device__ int TileSum(cg::thread_block_tile<32> tile, int v){
    for (int offset = tile.size() / 2; offset > 0; offset >>= 1){
        v += tile.shfl_down(v, offset);
    }
    return v;
}

// full reduction in one kernel: grid.sync() replaces a second kernel launch
__global__ void GridSum(const int* in, int* partial, int* out, int N){
    cg::grid_group grid = cg::this_grid();
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);

    __shared__ int warpSums[BLOCK / 32];

    // grid-stride loop, so any grid size the device can co-schedule works
    int start = grid.thread_rank();
    int stride = grid.size();
    int v = 0;
    for (int i = start; i < N; i += stride){
        v += in[i];
    }

    // reduce inside each warp, then the first warp combines the warp sums
    v = TileSum(tile, v);
    if (tile.thread_rank() == 0){
        warpSums[tile.meta_group_rank()] = v;
    }
    block.sync();

    if (tile.meta_group_rank() == 0){
        v = (tile.thread_rank() < tile.meta_group_size()) ? warpSums[tile.thread_rank()] : 0;
        v = TileSum(tile, v);
        if (tile.thread_rank() == 0){
            partial[blockIdx.x] = v;
        }
    }

    // every block waits here until all partial sums are written
    grid.sync();

    // first warp of block 0 adds up the per-block partial sums
    if (blockIdx.x == 0 && tile.meta_group_rank() == 0){
        int s = 0;
        for (int b = tile.thread_rank(); b < (int)gridDim.x; b += 32){
            s += partial[b];
        }
        s = TileSum(tile, s);
        if (tile.thread_rank() == 0){
            *out = s;
        }
    }
}

int main(){
    int N = 1 << 20;
    size_t bytes = N * sizeof(int);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (!prop.cooperativeLaunch){
        printf("device does not support cooperative launch\n");
        return 1;
    }

    // grid.sync() needs all blocks resident at once, so size the grid by occupancy
    int blocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, GridSum, BLOCK, 0);
    int grid = blocksPerSM * prop.multiProcessorCount;

    int* h_in = nullptr;
    cudaMallocHost(&h_in, bytes);
    long long expected = 0;
    for (int i = 0; i < N; i++){
        h_in[i] = i % 10;
        expected += h_in[i];
    }

    int *d_in, *d_partial, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_partial, grid * sizeof(int));
    cudaMalloc(&d_out, sizeof(int));
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // cooperative kernels take their arguments as an array of pointers
    void* args[] = { &d_in, &d_partial, &d_out, &N };
    cudaError_t err = cudaLaunchCooperativeKernel((void*)GridSum, grid, BLOCK, args);
    if (err != cudaSuccess){
        printf("Kernel launch error: %s \n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    int h_out = 0;
    cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    printf("blocks = %d, sum = %d (expected %lld)\n", grid, h_out, expected);

    cudaFreeHost(h_in);
    cudaFree(d_in);
    cudaFree(d_partial);
    cudaFree(d_out);
    return 0;
}

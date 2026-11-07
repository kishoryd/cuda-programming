#include <cuda_runtime.h>
#include <cstdio>

#define TILE 32

// transpose through shared memory so both global reads and writes are coalesced
__global__ void Transpose(const float* in, float* out, int width, int height){
    // +1 padding avoids shared memory bank conflicts
    __shared__ float tile[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;

    if (x < width && y < height){
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }
    __syncthreads();

    // swap block indices so the output write is also row-contiguous
    int ox = blockIdx.y * TILE + threadIdx.x;
    int oy = blockIdx.x * TILE + threadIdx.y;

    if (ox < height && oy < width){
        out[oy * height + ox] = tile[threadIdx.x][threadIdx.y];
    }
}

int main(){
    const int width = 1024;
    const int height = 512;
    size_t bytes = width * height * sizeof(float);

    float* h_in = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);
    for (int i = 0; i < width * height; i++){
        h_in[i] = (float)i;
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE, TILE);
    dim3 grid((width + TILE - 1) / TILE, (height + TILE - 1) / TILE);
    Transpose<<<grid, block>>>(d_in, d_out, width, height);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("Kernel launch error: %s \n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // out[c][r] must equal in[r][c]; count mismatches
    int errors = 0;
    for (int r = 0; r < height; r++){
        for (int c = 0; c < width; c++){
            if (h_out[c * height + r] != h_in[r * width + c]){
                errors++;
            }
        }
    }
    printf("mismatches = %d (expected 0)\n", errors);

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}

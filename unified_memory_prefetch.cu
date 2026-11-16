#include <cuda_runtime.h>
#include <cstdio>

#define BLOCK 256

// c = a + b on managed memory
__global__ void VecAdd(const float* a, const float* b, float* c, int N){
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < N){
        c[i] = a[i] + b[i];
    }
}

// prefetch helper: the API takes a cudaMemLocation from CUDA 13 on
void Prefetch(const void* ptr, size_t bytes, int device, cudaStream_t s){
#if CUDART_VERSION >= 13000
    cudaMemLocation loc;
    loc.type = (device == cudaCpuDeviceId) ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice;
    loc.id = (device == cudaCpuDeviceId) ? 0 : device;
    cudaMemPrefetchAsync(ptr, bytes, loc, 0, s);
#else
    cudaMemPrefetchAsync(ptr, bytes, device, s);
#endif
}

int main(){
    const int N = 1 << 24;
    size_t bytes = N * sizeof(float);

    int device = 0;
    cudaGetDevice(&device);

    float *a, *b, *c;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // first touch on the host, so pages start in CPU memory
    for (int i = 0; i < N; i++){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    cudaStream_t s;
    cudaStreamCreate(&s);

    // move the pages to the GPU before the kernel instead of faulting them in
    Prefetch(a, bytes, device, s);
    Prefetch(b, bytes, device, s);
    Prefetch(c, bytes, device, s);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int grid = (N + BLOCK - 1) / BLOCK;
    cudaEventRecord(start, s);
    VecAdd<<<grid, BLOCK, 0, s>>>(a, b, c, N);
    cudaEventRecord(stop, s);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("Kernel launch error: %s \n", cudaGetErrorString(err));
        return 1;
    }

    // bring the result back to the host before reading it
    Prefetch(c, bytes, cudaCpuDeviceId, s);
    cudaStreamSynchronize(s);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    double sum = 0.0;
    for (int i = 0; i < N; i++){
        sum += c[i];
    }
    printf("kernel time = %.3f ms\n", ms);
    printf("sum = %.1f (expected %.1f)\n", sum, 3.0 * N);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(s);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    return 0;
}

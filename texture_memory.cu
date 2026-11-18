#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

#define W 512
#define H 512

// 3x3 box blur; reads go through the texture cache, clamp mode handles the borders
__global__ void BoxBlur(cudaTextureObject_t tex, float* out, int w, int h){
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    if (x < w && y < h){
        float sum = 0.0f;
        for (int dy = -1; dy <= 1; dy++){
            for (int dx = -1; dx <= 1; dx++){
                // +0.5f samples the texel center
                sum += tex2D<float>(tex, x + dx + 0.5f, y + dy + 0.5f);
            }
        }
        out[y * w + x] = sum / 9.0f;
    }
}

int clampi(int v, int lo, int hi){
    return v < lo ? lo : (v > hi ? hi : v);
}

int main(){
    size_t bytes = W * H * sizeof(float);

    float *h_img = nullptr, *h_out = nullptr, *h_ref = nullptr;
    cudaMallocHost(&h_img, bytes);
    cudaMallocHost(&h_out, bytes);
    cudaMallocHost(&h_ref, bytes);

    for (int y = 0; y < H; y++){
        for (int x = 0; x < W; x++){
            h_img[y * W + x] = (float)((x * 3 + y * 5) % 17);
        }
    }

    // copy the image into a 2D CUDA array (texture memory layout)
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaArray_t arr;
    cudaMallocArray(&arr, &desc, W, H);
    cudaMemcpy2DToArray(arr, 0, 0, h_img, W * sizeof(float), W * sizeof(float), H, cudaMemcpyHostToDevice);

    // describe what the texture points at
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    // describe how it is sampled: clamp at edges, no interpolation, pixel coords
    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    cudaTextureObject_t tex = 0;
    cudaCreateTextureObject(&tex, &resDesc, &texDesc, nullptr);

    float* d_out;
    cudaMalloc(&d_out, bytes);

    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    BoxBlur<<<grid, block>>>(tex, d_out, W, H);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("Kernel launch error: %s \n", cudaGetErrorString(err));
        return 1;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // CPU reference with the same clamped borders
    for (int y = 0; y < H; y++){
        for (int x = 0; x < W; x++){
            float sum = 0.0f;
            for (int dy = -1; dy <= 1; dy++){
                for (int dx = -1; dx <= 1; dx++){
                    int sx = clampi(x + dx, 0, W - 1);
                    int sy = clampi(y + dy, 0, H - 1);
                    sum += h_img[sy * W + sx];
                }
            }
            h_ref[y * W + x] = sum / 9.0f;
        }
    }

    int mismatches = 0;
    double total = 0.0, expected = 0.0;
    for (int i = 0; i < W * H; i++){
        total += h_out[i];
        expected += h_ref[i];
        if (fabsf(h_out[i] - h_ref[i]) > 1e-4f) mismatches++;
    }
    printf("blur sum = %.2f (expected %.2f), mismatches = %d (expected 0)\n", total, expected, mismatches);

    cudaDestroyTextureObject(tex);
    cudaFreeArray(arr);
    cudaFree(d_out);
    cudaFreeHost(h_img);
    cudaFreeHost(h_out);
    cudaFreeHost(h_ref);
    return 0;
}

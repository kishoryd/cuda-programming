#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/functional.h>
#include <cstdio>

int main(){
    const int N = 1 << 20;

    // (i * 7919) % N is a shuffled permutation of 0..N-1 since 7919 is odd and N is a power of 2
    thrust::host_vector<int> h_keys(N);
    for (int i = 0; i < N; i++){
        h_keys[i] = (int)(((long long)i * 7919) % N);
    }

    // copying into a device_vector does the cudaMalloc + cudaMemcpy for us
    thrust::device_vector<int> d_keys = h_keys;

    // ascending sort on the GPU (radix sort under the hood for int keys)
    thrust::sort(d_keys.begin(), d_keys.end());
    cudaDeviceSynchronize();

    thrust::host_vector<int> h_sorted = d_keys;

    // a sorted permutation of 0..N-1 must have value i at index i
    int mismatches = 0;
    for (int i = 0; i < N; i++){
        if (h_sorted[i] != i) mismatches++;
    }
    printf("ascending: first = %d, last = %d (expected 0 and %d), mismatches = %d (expected 0)\n",
           (int)h_sorted[0], (int)h_sorted[N - 1], N - 1, mismatches);

    // sort by key: values travel with their keys, here value = 2 * key
    thrust::device_vector<int> d_k = h_keys;
    thrust::device_vector<int> d_v(N);
    for (int i = 0; i < N; i++){
        h_sorted[i] = 2 * h_keys[i];
    }
    d_v = h_sorted;
    thrust::sort_by_key(d_k.begin(), d_k.end(), d_v.begin(), thrust::greater<int>());
    cudaDeviceSynchronize();

    thrust::host_vector<int> h_k = d_k;
    thrust::host_vector<int> h_v = d_v;

    // descending: index i holds key N-1-i and value 2*(N-1-i)
    int badPairs = 0;
    for (int i = 0; i < N; i++){
        int k = N - 1 - i;
        if (h_k[i] != k || h_v[i] != 2 * k) badPairs++;
    }
    printf("sort_by_key desc: key[0] = %d, val[0] = %d (expected %d and %d), bad pairs = %d (expected 0)\n",
           (int)h_k[0], (int)h_v[0], N - 1, 2 * (N - 1), badPairs);

    return 0;
}

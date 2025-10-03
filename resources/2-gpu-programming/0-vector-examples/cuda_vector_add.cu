#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel for vector addition
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(int argc, char** argv) {
    // Set default values and parse command line arguments
    int n = 10000000;  // default array size
    
    if (argc == 2) {
        n = atoi(argv[1]);
    } else if (argc != 1) {
        printf("Usage: %s [array_size]\n", argv[0]);
        printf("Default: array_size=10000000\n");
        return 1;
    }
    size_t size = n * sizeof(float);
    
    // Host memory
    float *h_a, *h_b, *h_c;
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    // Initialize host data
    for(int i = 0; i < n; i++){
        h_a[i] = 1.0f; h_b[i] = 2.0f;
    }

    // Device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    vector_add<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);

    cudaDeviceSynchronize();
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // print first 10 elements of h_c
    for(int i = 0; i < 10; i++){
        printf("%f ", h_c[i]);
    }
    printf("\n");

    // Clean up
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}


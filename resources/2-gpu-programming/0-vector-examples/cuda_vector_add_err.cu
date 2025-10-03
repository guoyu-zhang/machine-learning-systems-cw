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
    float *a, *b, *c;
    a = (float*)malloc(size);
    b = (float*)malloc(size);
    c = (float*)malloc(size);

    // Initialize host data
    for(int i = 0; i < n; i++){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    
    vector_add<<<gridSize, blockSize>>>(a, b, c, n);
    
    // print first 10 elements of c
    for(int i = 0; i < 10; i++){
        printf("%f ", c[i]);
    }
    printf("\n");

    // Clean up
    free(a);
    free(b);
    free(c);

    return 0;
}


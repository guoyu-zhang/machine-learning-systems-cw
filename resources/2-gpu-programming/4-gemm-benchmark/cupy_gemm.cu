/*
Original works by:
--------------------------------------------------------
MAGMA
Copyright (c) 2017 The University of Tennessee. All rights reserved.
Licensed under modified BSD license
*/
// Based on the origin version of cupy gemm.cu, made some specific optimizations for A5000 and A6000 GPU on my test platform.


#define fetch(arr, col, m, n, bound) arr[min(n*col + m, bound)]

extern "C" __global__
void gemm_naive(
        int M, int N, int K,  // Matrix dimensions: A is MxK, B is KxN, C is MxN
        const float* A,        // Input matrix A
        const float* B,        // Input matrix B
        float* C)             // Output matrix C
{
    // Thread indices within block
    int thread_idx_x = threadIdx.x;
    int thread_idx_y = threadIdx.y;

    // Linear thread index within block
    int thread_id = DIM_X * thread_idx_y + thread_idx_x;

    // Thread indices for loading matrix A
    int thread_idx_a_x = thread_id % DIM_XA;
    int thread_idx_a_y = thread_id / DIM_XA;

    // Thread indices for loading matrix B
    int thread_idx_b_x = thread_id % DIM_XB;
    int thread_idx_b_y = thread_id / DIM_XB;

    // Block indices
    int block_idx_x = blockIdx.x;
    int block_idx_y = blockIdx.y;

    // Shared memory for block-level matrix multiplication
    __shared__ float shared_a[BLK_K][BLK_M + 1];  // +1 for bank conflict avoidance
    __shared__ float shared_b[BLK_N][BLK_K + 1];  // +1 for bank conflict avoidance

    // Register arrays for computation
    float reg_c[THR_N][THR_M];  // Accumulator for output values
    float reg_a[THR_M];         // Cache for A values in inner loop
    float reg_b[THR_N];         // Cache for B values in inner loop

    // Register arrays for loading data
    float reg_load_a[BLK_K / DIM_YA][BLK_M / DIM_XA];
    float reg_load_b[BLK_N / DIM_YB][BLK_K / DIM_XB];

    // Calculate starting positions and bounds for matrix loading
    const float* offset_a = A + block_idx_x * BLK_M + thread_idx_a_y * M + thread_idx_a_x;
    int bound_a = (M * (K - 1) + M) - (block_idx_x * BLK_M + thread_idx_a_y * M + thread_idx_a_x) - 1;
    const float* offset_b = B + block_idx_y * BLK_N * K + thread_idx_b_y * K + thread_idx_b_x;
    int bound_b = (K * (N - 1) + K) - (block_idx_y * BLK_N * K + thread_idx_b_y * K + thread_idx_b_x) - 1;

    int m, n, k, k_block;
    
    // Initialize output registers to zero
    #pragma unroll
    for (n = 0; n < THR_N; n++) {
        #pragma unroll
        for (m = 0; m < THR_M; m++) {
            reg_c[n][m] = 0;
        }
    }

    // Initial load of A into shared memory with transpose
    #pragma unroll
    for (n = 0; n < BLK_K; n += DIM_YA) {
        #pragma unroll
        for (m = 0; m < BLK_M; m += DIM_XA) {
            shared_a[n + thread_idx_a_y][m + thread_idx_a_x] = fetch(offset_a, M, m, n, bound_a);
        }
    }

    // Initial load of B into shared memory with transpose
    #pragma unroll
    for (n = 0; n < BLK_N; n += DIM_YB) {
        #pragma unroll
        for (m = 0; m < BLK_K; m += DIM_XB) {
            shared_b[n + thread_idx_b_y][m + thread_idx_b_x] = fetch(offset_b, K, m, n, bound_b);
        }
    }
    __syncthreads();

    // Main loop over K dimension
    for (k_block = 0; k_block < K - BLK_K; k_block += BLK_K)
    {
        // Update pointers and bounds for next block
        offset_a += BLK_K * M;
        bound_a -= BLK_K * M;
        offset_b += BLK_K;
        bound_b -= BLK_K;
        
        // Load next block of A into registers
        #pragma unroll
        for (n = 0; n < BLK_K / DIM_YA; n++) {
            #pragma unroll
            for (m = 0; m < BLK_M / DIM_XA; m++) {
                reg_load_a[n][m] = fetch(offset_a, M, m * DIM_XA, n * DIM_YA, bound_a);
            }
        }

        // Load next block of B into registers
        #pragma unroll
        for (n = 0; n < BLK_N / DIM_YB; n++) {
            #pragma unroll
            for (m = 0; m < BLK_K / DIM_XB; m++) {
                reg_load_b[n][m] = fetch(offset_b, K, m * DIM_XB, n * DIM_YB, bound_b);
            }
        }

        // Compute current block multiplication
        #pragma unroll
        for (k = 0; k < BLK_K; k++)
        {
            // Load values from shared memory into registers
            #pragma unroll
            for (m = 0; m < THR_M; m++) {
                reg_a[m] = shared_a[k][m * DIM_X + thread_idx_x];
            }
            
            #pragma unroll
            for (n = 0; n < THR_N; n++) {
                reg_b[n] = shared_b[n * DIM_Y + thread_idx_y][k];
            }

            // Perform matrix multiplication for this k
            #pragma unroll
            for (n = 0; n < THR_N; n++) {
                #pragma unroll
                for (m = 0; m < THR_M; m++) {
                    reg_c[n][m] += reg_a[m] * reg_b[n];
                }
            }
        }
        __syncthreads();

        // Store next A block from registers to shared memory
        #pragma unroll
        for (n = 0; n < BLK_K / DIM_YA; n++) {
            #pragma unroll
            for (m = 0; m < BLK_M / DIM_XA; m++) {
                shared_a[n * DIM_YA + thread_idx_a_y][m * DIM_XA + thread_idx_a_x] = reg_load_a[n][m];
            }
        }

        // Store next B block from registers to shared memory
        #pragma unroll
        for (n = 0; n < BLK_N / DIM_YB; n++) {
            #pragma unroll
            for (m = 0; m < BLK_K / DIM_XB; m++) {
                shared_b[n * DIM_YB + thread_idx_b_y][m * DIM_XB + thread_idx_b_x] = reg_load_b[n][m];
            }
        }
        __syncthreads();
    }

    // Process final block (may be partial)
    k_block = K - k_block;
    #pragma unroll
    for (k = 0; k < k_block; k++)
    {
        // Load values for final block computation
        #pragma unroll
        for (m = 0; m < THR_M; m++) {
            reg_a[m] = shared_a[k][m * DIM_X + thread_idx_x];
        }

        #pragma unroll
        for (n = 0; n < THR_N; n++) {
            reg_b[n] = shared_b[n * DIM_Y + thread_idx_y][k];
        }
        
        // Final block multiplication
        #pragma unroll
        for (n = 0; n < THR_N; n++) {
            #pragma unroll
            for (m = 0; m < THR_M; m++) {
                reg_c[n][m] += reg_a[m] * reg_b[n];
            }
        }
    }

    // Store results back to global memory
    #pragma unroll
    for (n = 0; n < THR_N; n++) {
        int coord_c_n = block_idx_y * BLK_N + n * DIM_Y + thread_idx_y;
        #pragma unroll
        for (m = 0; m < THR_M; m++) {
            int coord_c_m = block_idx_x * BLK_M + m * DIM_X + thread_idx_x;
            if (coord_c_m < M && coord_c_n < N) {
                C[coord_c_n * M + coord_c_m] = reg_c[n][m];
            }
        }
    }
}
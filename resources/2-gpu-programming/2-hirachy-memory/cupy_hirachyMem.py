# Implementing shared memory in CUDA kernels using CuPy: Hadamard product
# Input: Vector A[N], Matrix B[M][N]
# Output: Matrix C[M][N]
# C[m][n] = A[n] * B[m][n]

import numpy as np
import cupy as cp
from cupyx import jit

import time
       
@jit.rawkernel() 
def my_multiply(A_cu, B_cu, C_cu):
    """
    Basic kernel function to compute Hadamard product without shared memory
    """
    # Calculate global thread indices
    global_idx = jit.threadIdx.x + jit.blockIdx.x * jit.blockDim.x
    global_idy = jit.threadIdx.y + jit.blockIdx.y * jit.blockDim.y
    
    # Compute element-wise multiplication
    C_cu[global_idy, global_idx] = A_cu[global_idx] * B_cu[global_idy, global_idx]
    
@jit.rawkernel()
def my_multiply_sharedMem(A_cu, B_cu, C_cu, bulk_x, bulk_y):
    """
    Kernel function using shared memory to improve memory access efficiency
    """
    # Calculate global thread indices
    global_idx = jit.threadIdx.x + jit.blockIdx.x * jit.blockDim.x
    global_idy = jit.threadIdx.y + jit.blockIdx.y * jit.blockDim.y
    
    # Allocate shared memory buffer
    shared_mem = jit.shared_memory(cp.float32, 512)
    
    # Convert bulk sizes to uint32
    bulk_x_u32 = cp.uint32(bulk_x)
    bulk_y_u32 = cp.uint32(bulk_y)
    
    # Calculate memory offsets
    offset_x = global_idx * bulk_x_u32
    offset_y = global_idy * bulk_y_u32
    
    # Calculate offset in shared memory for this thread
    block_offset = jit.threadIdx.x * bulk_x_u32

    # Load data into shared memory
    for i in range(bulk_x_u32):
        shared_mem[block_offset+i] = A_cu[offset_x+i]
        jit.syncthreads()  # Ensure all threads finish loading before computation
    
    # Compute Hadamard product using data from shared memory
    for i in range(bulk_y_u32):
        for j in range(bulk_x_u32):
            C_cu[offset_y+i, offset_x+j] = shared_mem[block_offset+j] * B_cu[offset_y+i, offset_x+j]
    

if __name__ == '__main__':
    
    # Prepare input data
    N = 2560000  # Vector size
    M = 512  # Batch size
    repeat = 10  # Number of repetitions
    
    # Print device capabilities
    device = cp.cuda.Device()
    print(f"Max threads per block: {device.attributes['MaxThreadsPerBlock']}")
    print(f"Max threads per multiprocessor: {device.attributes['MaxThreadsPerMultiProcessor']}")
    print(f"Max grid dimensions: {device.attributes['MaxGridDimX']}, {device.attributes['MaxGridDimY']}, {device.attributes['MaxGridDimZ']}")
    
    # Initialize random input data
    A = np.random.rand(N).astype(np.float32)  # Input vector A
    B = np.random.rand(M, N).astype(np.float32)  # Input matrix B

    # CPU test code (commented out)
    # C_cpu = np.multiply(A, B)
    # start_cpu = time.time()
    # for _ in range(repeat):
    #     C_cpu = np.multiply(A, B) # Hadamard product on CPU
    # cpu_time = (time.time() - start_cpu) / repeat
    # print(f"CPU time: {cpu_time*1000:.6f} ms")
    
    # Transfer data to GPU
    A_cu = cp.asarray(A)  # Transfer A to GPU
    B_cu = cp.asarray(B)  # Transfer B to GPU
    C_cu = cp.empty_like(B)  # Allocate memory for the output C
    
    # Test 1: Using CuPy's built-in multiply function
    cp.multiply(A_cu, B_cu, out=C_cu)
    start_gpu = time.time()
    for _ in range(repeat):
        cp.multiply(A_cu, B_cu, out=C_cu)
    device.synchronize()  # Ensure all GPU computations finish
    end_gpu = time.time()
    print(f"GPU time: {(end_gpu-start_gpu)*1000 / repeat:.6f} ms")
    # print(C_cu)
    
    # Test 2: Using custom kernel without shared memory
    Db = (128, 1, 1)  # Thread block dimensions
    Dg = (N//Db[0], M//Db[1], 1)  # Grid dimensions
    my_multiply[Dg,Db](A_cu, B_cu, C_cu)
    start_gpu = time.time()
    for _ in range(repeat):
        my_multiply[Dg,Db](A_cu, B_cu, C_cu)
    device.synchronize()  # Ensure all GPU computations finish
    end_gpu = time.time()
    print(f"GPU time with kernel func: {(end_gpu-start_gpu)*1000 / repeat:.6f} ms")
    # print(C_cu)
    
    # Test 3: Using custom kernel with shared memory
    bulk_x = 1  # Number of elements processed by each thread in x dimension
    bulk_y = 128  # Number of elements processed by each thread in y dimension
    Db = (256, 2,)  # Thread block dimensions
    Dg = (N//bulk_x//Db[0], M//bulk_y//Db[1],)  # Grid dimensions
    my_multiply_sharedMem[Dg,Db](A_cu, B_cu, C_cu, bulk_x, bulk_y)
    start_gpu = time.time()
    for _ in range(repeat):
        my_multiply_sharedMem[Dg,Db](A_cu, B_cu, C_cu, bulk_x, bulk_y)
    device.synchronize()  # Ensure all GPU computations finish
    end_gpu = time.time()
    print(f"GPU time with shared memory: {(end_gpu-start_gpu)*1000 / repeat:.6f} ms")
    # print(C_cu)

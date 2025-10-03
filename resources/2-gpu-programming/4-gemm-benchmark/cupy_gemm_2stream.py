# Import required libraries
import cupy as cp  # GPU array operations
import numpy as np # CPU array operations
import time       # For timing measurements
import math       # For math operations


# File containing the CUDA C++ kernel code
gemm_code_file = "cupy_gemm.cu"

def read_code(code_filename, params):
    """
    Read and preprocess CUDA kernel code by adding macro definitions.
    
    Args:
        code_filename: Path to the CUDA source file
        params: Dictionary of parameters to define as macros
    Returns:
        Preprocessed CUDA code as string
    """
    with open(code_filename, 'r') as f:
        code = f.read()
    for k, v in params.items():
        code = '#define ' + k + ' ' + str(v) + '\n' + code
    return code

def benchmark(func, args, n_run):
    """
    Benchmark a CUDA function using CUDA events for accurate GPU timing.
    
    Args:
        func: Function to benchmark
        args: Arguments to pass to the function
        n_run: Number of benchmark iterations
    Returns:
        List of execution times in milliseconds
    """
    times = []
    for _ in range(n_run):
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        func(*args)
        end.record()
        end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end))  # milliseconds
    return times

def gemm_naive(A, B,
          dim_x=16, dim_y=32, blk_m=128, blk_n=128, blk_k=8,
          dim_xa=128, dim_ya=4, dim_xb=4, dim_yb=128):
    """
    Matrix multiplication implementation using a custom CUDA kernel with dual streams.
    Splits the computation across two CUDA streams for potential performance gain.
    
    Args:
        A, B: Input matrices
        dim_x, dim_y: Thread block dimensions
        blk_m, blk_n, blk_k: Block sizes for tiling
        dim_xa, dim_ya, dim_xb, dim_yb: Additional block dimensions
    Returns:
        Result of matrix multiplication
    """
    assert A.dtype == cp.float32
    assert B.dtype == cp.float32
    assert (dim_x * dim_y == dim_xa * dim_ya == dim_xb * dim_yb)

    m, k = A.shape
    k, n = B.shape

    # Create two CUDA streams for parallel processing
    stream1 = cp.cuda.Stream()
    stream2 = cp.cuda.Stream()

    # Split matrices horizontally for parallel processing
    split_point = m // 2

    with stream1:
        # Process first half of matrix A
        A1 = cp.asfortranarray(A[:split_point])
        C1 = cp.empty((split_point, n), dtype=cp.float32, order='F')
        
        # Configure and launch kernel for first half
        config = {'DIM_X': dim_x, 'DIM_Y': dim_y,
                'BLK_M': blk_m, 'BLK_N': blk_n, 'BLK_K': blk_k,
                'DIM_XA': dim_xa, 'DIM_YA': dim_ya,
                'DIM_XB': dim_xb, 'DIM_YB': dim_yb,
                'THR_M': blk_m // dim_x, 'THR_N': blk_n // dim_y}
        code = read_code(gemm_code_file, params=config)
        kern = cp.RawKernel(code, 'gemm_naive')

        grid = (int(math.ceil(split_point / blk_m)), int(math.ceil(n / blk_n)), 1)
        block = (dim_x, dim_y / 2, 1)
        args = (split_point, n, k, A1, cp.asfortranarray(B), C1)
        shared_mem = blk_k * (blk_m + 1) * 4 + blk_n * (blk_k + 1) * 4
        kern(grid, block, args=args, shared_mem=shared_mem)

    with stream2:
        # Process second half of matrix A
        A2 = cp.asfortranarray(A[split_point:])
        C2 = cp.empty((m - split_point, n), dtype=cp.float32, order='F')

        # Launch kernel for second half
        grid = (int(math.ceil((m - split_point) / blk_m)), int(math.ceil(n / blk_n)), 1)
        block = (dim_x, dim_y / 2, 1)
        args = (m - split_point, n, k, A2, cp.asfortranarray(B), C2)
        kern(grid, block, args=args, shared_mem=shared_mem)

    # Wait for both streams to complete
    stream1.synchronize()
    stream2.synchronize()
    
    # Combine results from both streams
    return cp.vstack((C1, C2))

def gemm(a, b, block_size=32):
    """
    Efficient tiled implementation of matrix multiplication: C = A * B.
    Uses manual tiling with shared memory and two CUDA streams.

    Parameters:
        a: Input matrix A (M x K).
        b: Input matrix B (K x N).
        block_size: Size of tile (sub-matrix block) for GPU threads.
    Returns:
        c: Resultant matrix C (M x N).
    """
    M, K_a = a.shape
    K_b, N = b.shape
    assert K_a == K_b, "Incompatible dimensions for matrix multiplication!"

    c = cp.matmul(a, b)
    
    return c

def gemm_benchmark(size=1000, repeat=1):
    """
    Compares the performance of a naive GEMM implementation
    to a tiled GEMM implementation using CuPy with two streams.
    
    Args:
        size: Size of square matrices to test
        repeat: Number of benchmark iterations
    """
    # Generate random matrices of given size allocated on the GPU
    random_matrix_a = cp.random.rand(size, size).astype(cp.float32)
    random_matrix_b = cp.random.rand(size, size).astype(cp.float32)

    # Warmup runs to ensure GPU is ready
    for _ in range(3):
        gemm_naive(random_matrix_a, random_matrix_b)
    naive_time = benchmark(gemm_naive, (random_matrix_a, random_matrix_b), n_run=repeat)

    for _ in range(3):
        gemm(random_matrix_a, random_matrix_b)
    stream_times = benchmark(gemm, (random_matrix_a, random_matrix_b), n_run=repeat)

    # Print benchmark results
    print(f"Matrix size: {size}x{size}")
    print(f"Naive GEMM: {np.mean(naive_time)} ms")
    print(f"2-Stream GEMM: {np.mean(stream_times)} ms")
    print(f"Speedup (2-Stream over Naive): {np.mean(naive_time) / np.mean(stream_times):.2f}x")


if __name__ == "__main__":
    # Run benchmark for large matrix multiplication
    gemm_benchmark(10240, repeat=10)  # You can increase 'repeat' for robustness to noise

import cupy as cp
import numpy as np
import time
import math


gemm_code_file = "cupy_gemm.cu"

def read_code(code_filename, params):
    with open(code_filename, 'r') as f:
        code = f.read()
    for k, v in params.items():
        code = '#define ' + k + ' ' + str(v) + '\n' + code
    return code

def benchmark(func, args, n_run):
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

def gemm_cupy(A, B,
          dim_x=32, dim_y=16, blk_m=192, blk_n=256, blk_k=8,
          dim_xa=128, dim_ya=4, dim_xb=4, dim_yb=128):
    assert A.dtype == cp.float32
    assert B.dtype == cp.float32
    assert (dim_x * dim_y == dim_xa * dim_ya == dim_xb * dim_yb)

    m, k = A.shape
    k, n = B.shape

    # Inputs matrices need to be in Fortran order.
    A = cp.asfortranarray(A)
    B = cp.asfortranarray(B)

    C = cp.empty((m, n), dtype=cp.float32, order='F')

    config = {'DIM_X': dim_x, 'DIM_Y': dim_y,
              'BLK_M': blk_m, 'BLK_N': blk_n, 'BLK_K': blk_k,
              'DIM_XA': dim_xa, 'DIM_YA': dim_ya,
              'DIM_XB': dim_xb, 'DIM_YB': dim_yb,
              'THR_M': blk_m // dim_x, 'THR_N': blk_n // dim_y}
    code = read_code(gemm_code_file, params=config)
    kern = cp.RawKernel(code, 'gemm_naive')

    grid = (int(math.ceil(m / blk_m)), int(math.ceil(n / blk_n)), 1)
    block = (dim_x, dim_y, 1)
    args = (m, n, k, A, B, C)
    shared_mem = blk_k * (blk_m + 1) * 4 + blk_n * (blk_k + 1) * 4
    kern(grid, block, args=args, shared_mem=shared_mem)
    return C



def gemm(a, b, block_size=32):
    """
    Efficient tiled implementation of matrix multiplication: C = A * B.
    Uses manual tiling with shared memory.

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
    to a tiled GEMM implementation using CuPy.
    """
    # Generate random matrices of given size allcoated on the GPU
    random_matrix_a = cp.random.rand(size, size).astype(cp.float32)
    random_matrix_b = cp.random.rand(size, size).astype(cp.float32)

    for _ in range(3):
        gemm_cupy(random_matrix_a, random_matrix_b)
    naive_time = benchmark(gemm_cupy, (random_matrix_a, random_matrix_b), n_run=repeat)

    for _ in range(3):
        cp.dot(random_matrix_a, random_matrix_b)
    cublas_times = benchmark(cp.dot, (random_matrix_a, random_matrix_b), n_run=repeat)

    # Final Output
    print(f"Matrix size: {size}x{size}")
    print(f"CuBlas GEMM: {np.mean(cublas_times)} ms")
    print(f"Our Cupy Kernel GEMM: {np.mean(naive_time)} ms")
    


if __name__ == "__main__":
    # Run benchmark for 1000x1000 matrix multiplication
    gemm_benchmark(10240, repeat=10)  # You can increase 'repeat' for robustness to noise

import cupy as cp
import numpy as np
import time
import math

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

def gemm(a, b):
    n, m = a.shape
    k = b.shape[1]
    c = cp.zeros((n, k), dtype=cp.float32)
    for i in range(n):
        for j in range(k):
            c[i, j] = cp.sum(a[i, :] * b[:, j])
    return c

def gemm_benchmark(size=1000, repeat=1):
    # Generate random matrices of given size allocated on the GPU
    random_matrix_a = cp.random.rand(size, size).astype(cp.float32)
    random_matrix_b = cp.random.rand(size, size).astype(cp.float32)

    for _ in range(3):
        gemm(random_matrix_a, random_matrix_b)
    time_cost = benchmark(gemm, (random_matrix_a, random_matrix_b), n_run=repeat)

    # Final Output
    print(f"Matrix size: {size}x{size}")
    print(f"Naive GEMM: {np.mean(time_cost)} ms")


if __name__ == "__main__":
    # Run benchmark for 1000x1000 matrix multiplication
    gemm_benchmark(10240, repeat=10)  # You can increase 'repeat' for robustness to noise

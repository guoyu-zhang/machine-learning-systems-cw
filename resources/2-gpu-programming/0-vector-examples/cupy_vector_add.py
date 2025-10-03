import cupy as cp
import time
import sys

def cupy_vector_add(a, b):
    return a + b  # Perform vector addition

def main():
    # Parse command-line arguments
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000  # Default: 10M
    repeat = int(sys.argv[2]) if len(sys.argv) > 2 else 100    # Default: 100

    # Create vectors on GPU
    a = cp.random.rand(n, dtype=cp.float32)
    b = cp.random.rand(n, dtype=cp.float32)

    # Warm-up
    c = cupy_vector_add(a, b)

    # Benchmark
    start = time.time()
    for _ in range(repeat):
        c = cupy_vector_add(a, b)
    cp.cuda.Stream.null.synchronize()  # Ensure all GPU computations finish
    end = time.time()

    avg_time = (end - start) / repeat
    print(f"Vector Add (CuPy) - Vector size: {n}, Repeat: {repeat}, Avg Time: {avg_time:.6f} seconds.")

if __name__ == "__main__":
    main()


# Implementation of the stream example
# Step 1: x = x1 * x1
# Step 2: y = x2 * x2
# Step 3: z = x * sin(x1)
# Step 4: w = y * cos(x2)
# Step 5: z = z + w

import cupy as cp
import time

# Implementation of the kernels
def square_kernel(x):
    return x * x

def sin_kernel(x):
    return cp.sin(x)

def cos_kernel(x):
    return cp.cos(x)

def compute_with_streams(x1, x2):
    # Create CUDA streams
    stream1 = cp.cuda.Stream()
    stream2 = cp.cuda.Stream()

    # Step 1 & 2: x = x1 * x1 and y = x2 * x2 (run in separate streams)
    # No need to immediately synchronize
    with stream1:
        x = square_kernel(x1)
        # Step 3: z = x * sin(x1) (stream3, depends on `x`)
        z = x * sin_kernel(x1)
    with stream2:
        y = square_kernel(x2)
        # Step 4: w = y * cos(x2) (stream4, depends on `y`)
        w = y * cos_kernel(x2)

    # Step 5: z = z + w (sync all before this step)
    cp.cuda.Stream(null=True).synchronize()  # Synchronize all streams
    z = z + w

    return z


def compute_without_streams(x1, x2):
    # Steps are executed serially
    # Step 1: x = x1 * x1
    x = square_kernel(x1)

    # Step 2: y = x2 * x2
    y = square_kernel(x2)

    # Step 3: z = x * sin(x1)
    z = x * sin_kernel(x1)

    # Step 4: w = y * cos(x2)
    w = y * cos_kernel(x2)

    # Step 5: z = z + w
    z = z + w

    return z

def run_with_streams(n=1_000_000):
    # Create arrays
    x1 = cp.random.rand(n, dtype=cp.float32)
    x2 = cp.random.rand(n, dtype=cp.float32)

    # Test with streams
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    z_streams = compute_with_streams(x1, x2)
    end.record()
    end.synchronize()
    print(f"With streams example took {cp.cuda.get_elapsed_time(start, end):.6f} ms.")
    
    # Test without streams
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    z_no_streams = compute_without_streams(x1, x2)
    end.record()
    end.synchronize()
    print(f"Without streams example took {cp.cuda.get_elapsed_time(start, end):.6f} ms.")

    # Validate the results
    assert cp.allclose(z_streams, z_no_streams), "Results don't match!"

if __name__ == "__main__":
    run_with_streams(n=1_000_000_000)
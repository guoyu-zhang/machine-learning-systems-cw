# stream 1:
# stage 1: copy array x from host to device
# stage 2: copy array y from host to device
# stream 2:
# stage 3: compute z = x * y
# stage 4: compute w = y + sin(x)
# stage 5: z = z * w
# stream 3:
# stage 6: copy z from device to host


import cupy as cp
import numpy as np
import time

# Implementation of the kernels
def square_kernel(x):
    return x * x

def sin_kernel(x):
    return cp.sin(x)

def cos_kernel(x):
    return cp.cos(x)

def compute_with_streams(x, y):
    # Create CUDA streams
    stream1 = cp.cuda.Stream()
    stream2 = cp.cuda.Stream()
    stream3 = cp.cuda.Stream()

    batch_num = 8

    # Split arrays into batches
    batch_size = len(x) // batch_num
    batches = [(i * batch_size, min((i + 1) * batch_size, len(x))) for i in range(batch_num)]
    
    # Initialize output array on device instead of host
    z_host = np.empty_like(x)
    
    for start, end in batches:
        with stream1:
            x_d = cp.asarray(x[start:end])
            y_d = cp.asarray(y[start:end])
            
        with stream2:
            # Need to wait for stream1 to complete
            stream2.wait_event(stream1.record())
            z = x_d * y_d
            w = y_d + sin_kernel(x_d)
            z = z * w
            
        with stream3:
            # Need to wait for stream2 to complete
            stream3.wait_event(stream2.record())
            z_host[start:end] = cp.asnumpy(z)
    
    return z_host


def compute_without_streams(x, y):
    x_d = cp.asarray(x)
    y_d = cp.asarray(y)
    z = x_d * y_d
    w = y_d + sin_kernel(x_d)
    z = z * w
    z = cp.asnumpy(z)
    return z

def run_with_streams(n=1_000_000):
    # Create arrays
    x = np.random.rand(n).astype(np.float32)
    y = np.random.rand(n).astype(np.float32)

    # Test with streams
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    z_streams = compute_with_streams(x, y)
    end.record()
    end.synchronize()
    print(f"With streams example took {cp.cuda.get_elapsed_time(start, end):.6f} seconds.")
    # print last 10 elements of z_streams
    print(z_streams[-10:])
    
    # Test without streams
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    start.record()
    z_no_streams = compute_without_streams(x, y)
    end.record()
    end.synchronize()
    print(f"Without streams example took {cp.cuda.get_elapsed_time(start, end):.6f} seconds.")
    # print last 10 elements of z_no_streams
    print(z_no_streams[-10:])

    # Validate the results
    assert cp.allclose(z_streams, z_no_streams), "Results don't match!"

if __name__ == "__main__":
    run_with_streams(n=1_000_000_000)
import torch
import time

# Implementation of the kernels
def square_kernel(x):
    return x * x

def sin_kernel(x):
    return torch.sin(x)

def cos_kernel(x):
    return torch.cos(x)

def compute_with_streams(x1, x2):
    # Create CUDA streams
    stream1 = torch.cuda.Stream()
    stream2 = torch.cuda.Stream()

    # Initialize tensors with proper size on GPU
    z = torch.empty_like(x1, device='cuda')
    w = torch.empty_like(x1, device='cuda')

    # Stream1 operations
    with torch.cuda.stream(stream1):
        x = square_kernel(x1)  # Step 1
        z = x * sin_kernel(x1)  # Step 3: z = x * sin(x1)
    
    # Stream2 operations
    with torch.cuda.stream(stream2):
        y = square_kernel(x2)  # Step 2
        w = y * cos_kernel(x2)  # Step 4: w = y * cos(x2)

    # Wait for all streams to finish before proceeding
    torch.cuda.synchronize()

    # Step 5: z = z + w (requires results from both streams)
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
    # Allocate random input data on GPU
    x1 = torch.rand(n, device="cuda", dtype=torch.float32)
    x2 = torch.rand(n, device="cuda", dtype=torch.float32)

    # Test with streams
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    z_streams = compute_with_streams(x1, x2)
    end.record()
    torch.cuda.synchronize()  # Wait for all kernels to finish
    print(f"With streams example took {start.elapsed_time(end):.6f} ms.")  # Convert milliseconds to seconds

    # Test without streams
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    z_no_streams = compute_without_streams(x1, x2)
    end.record()
    torch.cuda.synchronize()  # Wait for all kernels to finish
    print(f"Without streams example took {start.elapsed_time(end):.6f} ms.")  # Convert milliseconds to seconds

    # Validate the results
    assert torch.allclose(z_streams, z_no_streams), "Results don't match!"

if __name__ == "__main__":
    run_with_streams(n=1_000_000_000)
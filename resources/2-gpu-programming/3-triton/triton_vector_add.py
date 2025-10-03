import triton
import triton.language as tl
import torch
import time
import sys

@triton.jit
def vec_add_kernel(X_ptr, Y_ptr, Z_ptr, N, BLOCK: tl.constexpr):
    """
    Vector addition kernel using Triton.
    Args:
      X_ptr, Y_ptr, Z_ptr: Pointers to input/output tensors (GPU).
      N: Total size of the vector.
      BLOCK: Block size (must be compile-time constant).
    """
    # Get the program ID (range depends on grid size)
    pid = tl.program_id(0)
    # Compute the block of indices handled by this program
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    # Mask to handle out-of-bounds memory accesses
    mask = offsets < N

    # Load inputs from device memory
    x = tl.load(X_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(Y_ptr + offsets, mask=mask, other=0.0)
    # Perform the vector addition
    z = x + y
    # Store the result
    tl.store(Z_ptr + offsets, z, mask=mask)

def triton_vector_add(n, repeat):
    # Create inputs on GPU (using PyTorch tensors for convenience)
    x = torch.rand(n, device="cuda", dtype=torch.float32)
    y = torch.rand(n, device="cuda", dtype=torch.float32)
    z = torch.empty_like(x)

    # Grid configuration: divide the work over blocks
    BLOCK = 1024
    # grid = (n + BLOCK - 1) // BLOCK
    grid = lambda meta: ( (n + BLOCK - 1) // BLOCK, )  # Meta computes the grid size

    # Warm-up (ensure Triton compiles the kernel before timing)
    vec_add_kernel[grid](x, y, z, n, BLOCK)

    # Measure performance with repeats
    start = time.time()
    for _ in range(repeat):
        vec_add_kernel[grid](x, y, z, n, BLOCK)
    torch.cuda.synchronize()  # Ensure all GPU computations are complete
    end = time.time()

    avg_time = (end - start) / repeat
    print(f"Vector Add (Triton) - Vector size: {n}, Repeat: {repeat}, Avg Time: {avg_time:.6f} seconds.")

def main():
    # Parse command-line arguments for vector size and repeat count
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000  # Default: 10M
    repeat = int(sys.argv[2]) if len(sys.argv) > 2 else 100    # Default: 100
    triton_vector_add(n, repeat)

if __name__ == "__main__":
    main()


import triton
import triton.language as tl
import torch
import time

# Triton Kernel for square operation
@triton.jit
def square_kernel(X, Y, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)  # Get program ID
    block_start = pid * BLOCK_SIZE  # Starting point for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Offsets for this thread
    mask = offsets < N  # Ensure we don't go out of bounds
    x = tl.load(X + offsets, mask=mask, other=0.0)  # Load inputs
    x = x * x  # Square the values
    tl.store(Y + offsets, x, mask=mask)  # Store the results

# Triton Kernel for sin and cos operation
@triton.jit
def trig_kernel(x_ptr, out_ptr, trig_type: tl.constexpr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)  # Get program ID
    block_start = pid * BLOCK_SIZE  # Starting point for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Offsets for this thread
    mask = offsets < n_elements  # Ensure we don't go out of bounds
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)  # Load inputs
    if trig_type == 0:  # sin
        out = tl.math.sin(x)
    elif trig_type == 1:  # cos
        out = tl.math.cos(x)
    tl.store(out_ptr + offsets, out, mask=mask)  # Store output

# Triton Kernel for elementwise multiplication
@triton.jit
def elementwise_mul_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)  # Get program ID
    block_start = pid * BLOCK_SIZE  # Starting point for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Offsets for this thread
    mask = offsets < n_elements  # Ensure we don't go out of bounds
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)  # Load inputs
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)  # Load inputs
    out = a * b  # Multiply
    tl.store(out_ptr + offsets, out, mask=mask)  # Store output

# Triton Kernel for addition
@triton.jit
def add_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)  # Get program ID
    block_start = pid * BLOCK_SIZE  # Starting point for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Offsets for this thread
    mask = offsets < n_elements  # Ensure we don't go out of bounds
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)  # Load inputs
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)  # Load inputs
    out = a + b  # Add
    tl.store(out_ptr + offsets, out, mask=mask)  # Store output

# Main compute function
def compute_with_triton(x1, x2, n_elements):
    BLOCK_SIZE = 1024  # Set block size

    # Allocate output tensors
    x1_square = torch.empty_like(x1, device='cuda')
    x2_square = torch.empty_like(x2, device='cuda')

    x1_sin = torch.empty_like(x1, device='cuda')
    x2_cos = torch.empty_like(x2, device='cuda')

    z_part = torch.empty_like(x1, device='cuda')
    w_part = torch.empty_like(x2, device='cuda')
    result = torch.empty_like(x1, device='cuda')

    # Step 1 & 2: x1^2 and x2^2
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    square_kernel[grid](x1, x1_square, n_elements, BLOCK_SIZE)
    square_kernel[grid](x2, x2_square, n_elements, BLOCK_SIZE)

    # Step 3: z = x1^2 * sin(x1)
    trig_kernel[grid](x1, x1_sin, 0, n_elements, BLOCK_SIZE)  # sin
    elementwise_mul_kernel[grid](x1_square, x1_sin, z_part, n_elements, BLOCK_SIZE)

    # Step 4: w = x2^2 * cos(x2)
    trig_kernel[grid](x2, x2_cos, 1, n_elements, BLOCK_SIZE)  # cos
    elementwise_mul_kernel[grid](x2_square, x2_cos, w_part, n_elements, BLOCK_SIZE)

    # Step 5: Final z = z + w
    add_kernel[grid](z_part, w_part, result, n_elements, BLOCK_SIZE)

    return result
    

def main():
    n_elements = 1_000_000_000
    repeat = 10
    # Allocate input data
    x1 = torch.rand(n_elements, device="cuda", dtype=torch.float32)
    x2 = torch.rand(n_elements, device="cuda", dtype=torch.float32)

    # Warm-up to compile Triton kernels
    for _ in range(repeat):
        compute_with_triton(x1, x2, n_elements)

    # Measure performance
    time_with_triton = 0
    time_without_triton = 0
    

    for _ in range(repeat):
        start = time.time() 
        z_result = compute_with_triton(x1, x2, n_elements)
        end = time.time()
        time_with_triton += end - start


    print(f"With Triton computation took {(time_with_triton / repeat) * 1000:.6f} ms.")

if __name__ == "__main__":
    main()

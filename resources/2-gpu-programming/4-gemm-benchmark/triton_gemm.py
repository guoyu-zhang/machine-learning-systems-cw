import triton
import triton.language as tl
import numpy as np
import torch
import time

# Triton kernel for GEMM computation
@triton.jit
def gemm_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bn, stride_bk, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    # Program Ids
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Memory offsets for the tiles this program should handle
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Create pointers for the A and B matrices
    a_tile = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_tile = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_tile, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_tile, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        
        # Move to the next block in K dimension
        a_tile += BLOCK_K * stride_ak
        b_tile += BLOCK_K * stride_bk
    
    # Write back results to C
    c_tile = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_tile, acc, mask=(offs_m[:, None] < M) & (offs_n[None] < N))


# Benchmark GEMM
def benchmark(size=1000, repeat=10):
    # Create input tensors
    a = torch.randn((size, size), device='cuda', dtype=torch.float32)
    b = torch.randn((size, size), device='cuda', dtype=torch.float32)
    c = torch.empty_like(a)
    
    # Parameters for Triton kernel
    BLOCK = 64  # Block size
    M, N, K = size, size, size
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    
    triton_time = 0
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.time()
        gemm_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(1), b.stride(0),
            c.stride(0), c.stride(1),
            BLOCK_M=BLOCK, BLOCK_N=BLOCK, BLOCK_K=BLOCK
        )
        torch.cuda.synchronize()
        triton_time += time.time() - start
        
    triton_time /= repeat

    # Naive GEMM using PyTorch
    naive_time = 0
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.time()
        torch.mm(a, b, out=c)
        torch.cuda.synchronize()
        naive_time += time.time() - start
    
    naive_time /= repeat

    print(f"Matrix size: {size}x{size}")
    print(f"CuBlas GEMM: {naive_time*1000:.2f} ms")
    print(f"Triton GEMM: {triton_time*1000:.2f} ms")


if __name__ == "__main__":
    benchmark(10240, 10)

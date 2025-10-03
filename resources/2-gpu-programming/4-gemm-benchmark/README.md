# Example 4: GEMM Benchmark

## Task: Matrix Multiplication Optimization

This example implements and benchmarks General Matrix Multiplication (GEMM) using Triton, comparing it with PyTorch's native implementation.

### Implementation Details

The code implements two versions of matrix multiplication:
1. Triton-optimized GEMM kernel
   - Uses block-based computation
   - Implements efficient memory tiling
   - Utilizes Triton's high-performance primitives

2. PyTorch's native `torch.mm` implementation

Key optimizations in the Triton kernel:
- Block-level data loading
- Efficient use of shared memory
- Automatic parallelization across GPU threads
- Configurable block sizes for performance tuning

### Running the Code

```python
python ./triton_gemm.py
```

The benchmark will:
1. Create large matrices (10240 x 10240 by default)
2. Run both implementations multiple times
3. Display:
   - Execution time for both versions
   - Performance comparison (speedup factor)
   - Matrix size and timing details

### Result

Matrix size: 10240x10240

CuBlas GEMM (call from cupy): 63.25 ms
CuBlas GEMM (call from torch): 68.88 ms
Our Cupy Kernel GEMM: 74.83 ms
Our Triton GEMM: 99.03 ms
Final version (cupy kernel + streaming optimization): 55.76 ms
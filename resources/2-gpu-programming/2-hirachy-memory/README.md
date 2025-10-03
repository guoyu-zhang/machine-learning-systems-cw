# Example 2: Hierarchy Memory

## Task: Shared Memory Optimization

This example demonstrates the use of shared memory in CUDA kernels to optimize matrix operations, specifically for computing the Hadamard product:
- Input: Vector A[N], Matrix B[M][N]
- Output: Matrix C[M][N]
- Operation: C[m][n] = A[n] * B[m][n]

### Implementation Details

The code implements three versions of the Hadamard product:
1. Using CuPy's built-in multiply function
2. Custom CUDA kernel without shared memory
3. Custom CUDA kernel with shared memory optimization

The shared memory optimization:
- Loads vector A into fast shared memory buffer
- Reduces global memory access latency
- Uses thread synchronization to ensure data consistency
- Processes data in bulks for better memory access patterns

### Running the Code

```python
python ./cupy_hirachyMem.py
```

# Example 3: Triton Stream Operations

## Task: Vector Addition with Triton

```python
python ./triton_vector_add.py
```

## Task: Stream Operations with Triton

This example demonstrates how to implement complex stream operations using Triton kernels. The implementation follows these steps:
- Step 1: x = x1 * x1 (square operation)
- Step 2: y = x2 * x2 (square operation)
- Step 3: z = x * sin(x1)
- Step 4: w = y * cos(x2)
- Step 5: z = z + w

### Implementation Details

The code implements several Triton kernels:
- `square_kernel`: Computes element-wise square
- `trig_kernel`: Computes sin/cos operations
- `elementwise_mul_kernel`: Performs element-wise multiplication
- `add_kernel`: Performs element-wise addition

Each kernel utilizes Triton's block-based execution model with:
- Configurable block sizes
- Automatic bounds checking
- Efficient memory access patterns

### Running the Code

```python
python ./triton_stream.py
```

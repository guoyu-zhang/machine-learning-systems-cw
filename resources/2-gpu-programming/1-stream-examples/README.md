# Example 1: Stream

## Task 1: Stream Concurrency

Implementation of the stream example
- Step 1: x = x1 * x1
- Step 2: y = x2 * x2
- Step 3: z = x * sin(x1)
- Step 4: w = y * cos(x2)
- Step 5: z = z + w

```python
# CuPy version
python ./cupy_stream.py

# PyTorch version
python ./pytorch_stream.py
```

## Task 2: Stream Pipeline

Implementation of the stream pipeline example:
- copy array x from host to device
- copy array y from host to device
- compute z = x * y
- compute w = y + sin(x)
- z = z * w
- copy z from device to host

To achieve the highest performance, we can use 3 streams to pipeline the computation.

Stream 1:
- Stage 1: copy array x from host to device
- Stage 2: copy array y from host to device

Stream 2:
- Stage 3: compute z = x * y
- Stage 4: compute w = y + sin(x)

Stream 3:
- Stage 5: copy z from device to host

```python
python ./cupy_stream_memory.py
```
# Example 0: Vector Add

This example is a simple vector addition program that demonstrates the basic usage of CUDA. It adds two vectors element-wise and stores the result in a third vector.

Pesudo code:
```text
for i from 0 to n-1
    c[i] = a[i] + b[i]
```

## Cuda version

Implementing a CUDA kernel to add two vectors is straightforward.

```cuda
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}
```

To call this kernel, we need to launch it with the `<<<>>>` syntax.

For example:
```cuda
vector_add<<<gridSize, blockSize>>>(a, b, c, n);
```

This will launch `gridSize` blocks of `blockSize` threads to compute the vector addition.

But this is not the end of the story. We need to allocate memory on the device and copy the data from the host to the device.

The wrong version code is in `cuda_vector_add_err.cu`, and the right version code is in `cuda_vector_add.cu`.

Compile the code:
```bash
make
```

Run the code:
```bash
# wrong version
./cuda_vector_add_err
# right version
./cuda_vector_add
```

## CuPy version

CuPy is a library that provides a Python interface to CUDA. It is a drop-in replacement for NumPy, but with GPU acceleration.

```bash
python ./cupy_vector_add.py
```

## PyTorch version

PyTorch is a library that provides a Python interface to CUDA. It is a drop-in replacement for NumPy, but with GPU acceleration.

```bash
python ./pytorch_vector_add.py
```

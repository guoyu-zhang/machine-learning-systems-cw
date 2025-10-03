# GPU Programming

## Slides

## Prerequisites

- Python 3.10+
- CUDA 11/12
- PyTorch 2.1
- Triton
- Cupy

### Installation

On most systems, you can install the dependencies with pip.
```bash
pip install torch cupy triton
```

On some systems, you may need to install the dependencies manually.
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
For more details on different system, use the following links:
- [PyTorch](https://pytorch.org/get-started/locally/)
- [Triton](https://triton-lang.org/main/getting-started/installation.html)
- [Cupy](https://docs.cupy.dev/en/stable/install.html)

## Examples

- [Example 0: Vector add](./0-vector-examples)
- [Example 1: Streams](./1-stream-examples)
- [Example 2: Memory hierarchy](./2-hirachy-memory)
- [Example 3: Triton](./3-triton)
- [Example 4: GEMM benchmark](./4-gemm-benchmark)

## References and further reading

### Environment
- [Miniconda and virtual environment](https://medium.com/@aminasaeed223/a-comprehensive-tutorial-on-miniconda-creating-virtual-environments-and-setting-up-with-vs-code-f98d22fac8e2)
- [How to install cupy](https://docs.cupy.dev/en/stable/install.html)
- [How to install triton](https://triton-lang.org/main/getting-started/installation.html)
- [How to install Pytorch](https://pytorch.org/get-started/locally/)

### AI Accelerators

- [Brief AI Accelerator History - Chivier's Blog](https://blog.chivier.site/2025-01-16/2025/Brief-AI-Accelerator-History/)
- [GPU Glossary](https://modal.com/gpu-glossary)
- [NVIDIA Ampere Architecture In-Depth | NVIDIA Technical Blog](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)

### Cuda Tutorials

- [NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples)
- [An Even Easier Introduction to CUDA | NVIDIA Technical Blog](https://developer.nvidia.com/blog/even-easier-introduction-cuda/)
- [CUDA Programming Guide](https://www.cs.utexas.edu/~rossbach/cs380p/papers/cuda-programming.pdf)
- [HMUNACHI/cuda-repo: From zero to hero CUDA for accelerating maths and machine learning on GPU.](https://github.com/HMUNACHI/cuda-repo)

### Triton

- [Triton Language](https://triton-lang.org/)
- [Triton GitHub](https://github.com/openai/triton)
- [Triton Documentation](https://triton-lang.org/main/getting-started/introduction.html)

### Cupy

- [Cupy GitHub](https://github.com/cupy/cupy)
- [Cupy Documentation](https://docs.cupy.dev/en/stable/)


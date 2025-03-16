# Optimizing Matmul using CUDA from Scratch
A step-by-step optimization of matrix multiplication using CUDA to achieve cuBLAS-level performance.


## Environment
### System
- 1 x NVIDIA Tesla V100 32GB (Peak FP32 GFLOPS: `15700`)

### Software
- CUDA Version: `12.4`

## Performance
The matrix size is determined by setting the dimensions M, N, and K to `4096`.

Kernel                               | GFLOPS      | Perf. against cuBLAS (%)
------------------------------------ | ----------- | -------------------------
1: Naive                             | `245.7`     | 1.7
2: GMEM Coalescing                   | `2311.7`    | 16.3
3: SMEM Caching                      | `4263.2`    | 30.0
4: Block Tiling 1D                   | `4950.2`    | 40.6
5: Block Tiling 1D (GMEM coalescing) | `8541.0`    | 60.6
6: Block Tiling 2D                   | `12053.7`   | 84.9
7: Block Tiling 2D (Vectorized)      | `13583.9`   | 95.6
8: Warp Tiling                       |             | 
0: cuBLAS                            | `14205.7`   | 100.0

## Usage
### Build
```bash
$ make
```
### Run
```bash
$ run.sh
```


## References
- https://siboehm.com/articles/22/CUDA-MMM
- https://developer.nvidia.com/blog/cutlass-linear-algebra-cuda
- https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
- https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html
- https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
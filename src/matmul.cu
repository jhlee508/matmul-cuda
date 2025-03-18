#include <cstdio>
#include <cublas_v2.h>

#include "matmul.cuh"
#include "kernels.cuh"

#include "common.h"


static float *A_gpu, *B_gpu, *C_gpu;
static cublasHandle_t handle;

void matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  /* Naive */
  // dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
  // dim3 blockDim(32, 32);
  // naive_kernel<<<gridDim, blockDim>>>(A_gpu, B_gpu, C_gpu, M, N, K);

  /* Global Memory Coalescing */
  // dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
  // dim3 blockDim(32, 32);
  // gmem_coalescing_kernel<<<gridDim, blockDim>>>(A_gpu, B_gpu, C_gpu, M, N, K);

  /* Shared Memory Caching */
  // dim3 gridDim(CEIL_DIV(N, SMEM_BS), CEIL_DIV(M, SMEM_BS));
  // dim3 blockDim(SMEM_BS, SMEM_BS);
  // smem_caching_kernel<<<gridDim, blockDim>>>(A_gpu, B_gpu, C_gpu, M, N, K);

  /* Blocktiling 1D */
  // dim3 gridDim(CEIL_DIV(N, BLOCKTILING_1D_BN), CEIL_DIV(M, BLOCKTILING_1D_BM));
  // dim3 blockDim(BLOCKTILING_1D_BN, BLOCKTILING_1D_BM / BLOCKTILING_1D_TM);
  // blocktiling_1d_kernel<<<gridDim, blockDim>>>(A_gpu, B_gpu, C_gpu, M, N, K);

  /* Blocktiling 1D (GMEM Coalescing) */
  // dim3 gridDim(CEIL_DIV(N, BLOCKTILING_1D_BN), CEIL_DIV(M, BLOCKTILING_1D_BM));
  // dim3 blockDim(BLOCKTILING_1D_BN, BLOCKTILING_1D_BM / BLOCKTILING_1D_TM);
  // blocktiling_1d_kernel_v2<<<gridDim, blockDim>>>(A_gpu, B_gpu, C_gpu, M, N, K);

  /* Blocktiling 2D */
  // dim3 gridDim(CEIL_DIV(N, BLOCKTILING_2D_BN), CEIL_DIV(M, BLOCKTILING_2D_BM));
  // dim3 blockDim(BLOCKTILING_2D_BN / BLOCKTILING_2D_TN, BLOCKTILING_2D_BM / BLOCKTILING_2D_TM);
  // blocktiling_2d_kernel<<<gridDim, blockDim>>>(A_gpu, B_gpu, C_gpu, M, N, K);

  /* Blocktiling 2D (Vectorized) */
  dim3 gridDim(CEIL_DIV(N, BLOCKTILING_2D_BN), CEIL_DIV(M, BLOCKTILING_2D_BM));
  dim3 blockDim(BLOCKTILING_2D_BN / BLOCKTILING_2D_TN, BLOCKTILING_2D_BM / BLOCKTILING_2D_TM);
  blocktiling_2d_vec_kernel<<<gridDim, blockDim>>>(A_gpu, B_gpu, C_gpu, M, N, K);

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_cublas(float *_A, float *_B, float *_C, int M, int N, int K) {
  const float one = 1, zero = 0;
  CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one,
                           B_gpu, N, A_gpu, K, &zero, C_gpu, N));
  // CHECK_CUBLAS(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one,
  //                           B_gpu, CUDA_R_32F, N, A_gpu, CUDA_R_32F, K, &zero, C_gpu, CUDA_R_32F, N,
  //                           CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_initialize(float *_A, float *_B, int M, int N, int K) { 
  CHECK_CUDA(cudaMalloc((void **) &A_gpu, sizeof(float) * M * K));
  CHECK_CUDA(cudaMalloc((void **) &B_gpu, sizeof(float) * K * N));
  CHECK_CUDA(cudaMalloc((void **) &C_gpu, sizeof(float) * M * N));

  CHECK_CUDA(
    cudaMemcpy(A_gpu, _A, sizeof(float) * M * K, cudaMemcpyHostToDevice));
  CHECK_CUDA(
    cudaMemcpy(B_gpu, _B, sizeof(float) * K * N, cudaMemcpyHostToDevice));
}

void cublas_initialize() {
  CHECK_CUBLAS(cublasCreate(&handle));
  // CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH)); /* To enable TC */
}

void matmul_finalize(float *_C, int M, int N, int K) {
  CHECK_CUDA(
    cudaMemcpy(_C, C_gpu, sizeof(float) * M * N, cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(A_gpu));
  CHECK_CUDA(cudaFree(B_gpu));
  CHECK_CUDA(cudaFree(C_gpu));
}

void cublas_finalize() {
  CHECK_CUBLAS(cublasDestroy(handle));
}

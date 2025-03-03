#include <cstdio>
#include <cublas_v2.h>

#include "matmul.h"


static float *A_gpu, *B_gpu, *C_gpu;
static cublasHandle_t handle;

__global__ void naive_kernel(float *A, float *B, float *C, int M, int N, int K) {
  const int Row = blockIdx.x * blockDim.x + threadIdx.x;
  const int Col = blockIdx.y * blockDim.y + threadIdx.y;

  float tmp = 0.f;

  for (int k = 0; k < K; k++) {
    tmp += A[Row * K + k] * B[k * N + Col];
  }
  C[Row * N + Col] = tmp;
}

__global__ void gmem_coalescing_kernel(float *A, float *B, float *C, int M, int N, int K) {
  const int Col = blockIdx.x * blockDim.x + threadIdx.x;
  const int Row = blockIdx.y * blockDim.y + threadIdx.y;

  float tmp = 0.f;

  for (int k = 0; k < K; k++) {
    tmp += A[Row * K + k] * B[k * N + Col];
  }
  C[Row * N + Col] = tmp;
}

__global__ void smem_caching_kernel(float *A, float *B, float *C, int M, int N, int K) {
  const int gCol = blockIdx.x * blockDim.x + threadIdx.x;
  const int gRow = blockIdx.y * blockDim.y + threadIdx.y;
  const int lCol = threadIdx.x;
  const int lRow = threadIdx.y;

  float tmp = 0.f;

  // 0. Allocate SMEM
  __shared__ float A_SMEM[SMEM_TILE_SIZE][SMEM_TILE_SIZE];
  __shared__ float B_SMEM[SMEM_TILE_SIZE][SMEM_TILE_SIZE];

  // 1. Proceed matmul over tiles
  for (int t = 0; t < K; t+=SMEM_TILE_SIZE) {
    // 1-1. Load tiles to SMEM
    A_SMEM[lRow][lCol] = A[gRow * K + (lCol + t)];
    B_SMEM[lRow][lCol] = B[(lRow + t) * N + gCol];
    __syncthreads();

    // 1-2. Compute matmul for tiles
    for (int i = 0; i < SMEM_TILE_SIZE; i++) {
      tmp += A_SMEM[lRow][i] * B_SMEM[i][lCol];
    }
    __syncthreads();
  }

  // 2. Store the results
  C[gRow * N + gCol] = tmp;
}

void matmul(float *_A, float *_B, float *_C, int M, int N, int K) {
  dim3 blockDim(32, 32);
  dim3 gridDim(CEIL_DIV(N, 32), CEIL_DIV(M, 32));
  
  void (*matmul_kernel)(float *, float *, float *, int, int, int);
  /* Types of matmul_kernel(s):
    1. naive_kernel
    2. gmem_coalescing_kernel
    3. smem_caching_kernel
  */
  matmul_kernel = smem_caching_kernel; // Choose the kernel here!
  matmul_kernel<<<gridDim, blockDim>>>(A_gpu, B_gpu, C_gpu, M, N, K);

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void matmul_cublas(float *_A, float *_B, float *_C, int M, int N, int K) {
  const float one = 1, zero = 0;
  CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one,
                           B_gpu, N, A_gpu, K, &zero, C_gpu, N));

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

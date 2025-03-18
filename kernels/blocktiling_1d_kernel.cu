#include "common.h"

__global__ void blocktiling_1d_kernel(float *A, float *B, float *C, int M, int N, int K) {
  const int gCol = blockIdx.x * blockDim.x + threadIdx.x;
  const int gRow = (blockIdx.y * blockDim.y + threadIdx.y) * BLOCKTILING_1D_TM;
  const int lCol = threadIdx.x; // 0..63
  const int lRow = threadIdx.y; // 0..7

  float workPerThread[BLOCKTILING_1D_TM] = {0.f};

  /* 0. Allocate SMEM */
  __shared__ float A_SMEM[BLOCKTILING_1D_BM][BLOCKTILING_1D_BK];
  __shared__ float B_SMEM[BLOCKTILING_1D_BK][BLOCKTILING_1D_BN];

  /* 1. Proceed matmul over blocks */
  for (int bk = 0; bk < K; bk += BLOCKTILING_1D_BK) {

    /* 1-1. Load blocks to SMEM */
    // 1) A_SMEM [64][8]
    int rowA = lCol;   // => A_SMEM Row = [0..63]
    int colA = lRow;   // => A_SMEM Col = [0..7]
    int globalArow = (blockIdx.y * BLOCKTILING_1D_BM) + rowA;
    int globalAcol = bk + colA;  
    A_SMEM[rowA][colA] = A[globalArow * K + globalAcol];

    // 2) B_SMEM [8][64]
    int rowB = lRow;   // => B_SMEM Row = [0..7]
    int colB = lCol;   // => B_SMEM Col = [0..63]
    B_SMEM[rowB][colB] = B[(bk + rowB) * N + gCol];
    __syncthreads();

    /* 1-2. Compute matmul for blocks */
    for (int tk = 0; tk < BLOCKTILING_1D_BK; tk++) {
      float regB = B_SMEM[tk][lCol];
      for (int w = 0; w < BLOCKTILING_1D_TM; w++) {
        workPerThread[w] += A_SMEM[lRow * BLOCKTILING_1D_TM + w][tk] * regB;
      }
    }
    __syncthreads();
  }

  /* 2. Store the results */
  for (int w = 0; w < BLOCKTILING_1D_TM; w++) {
    if (gRow + w < M && gCol < N) {
      C[(gRow + w) * N + gCol] = workPerThread[w];
    }
  }
}
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
  const int gCol = blockIdx.x * blockDim.x + threadIdx.x; // global index of col in B 
  const int gRow = blockIdx.y * blockDim.y + threadIdx.y; // global index of row in A
  const int lCol = threadIdx.x; // local index of col in B (0..31) 
  const int lRow = threadIdx.y; // local index of row in A (0..31)

  float tmp = 0.f;

  /* 0. Allocate SMEM */
  __shared__ float A_SMEM[SMEM_BS][SMEM_BS];
  __shared__ float B_SMEM[SMEM_BS][SMEM_BS];

  /* 1. Proceed matmul over blocks */
  for (int bk = 0; bk < K; bk += SMEM_BS) {

    /* 1-1. Load blocks to SMEM */
    A_SMEM[lRow][lCol] = A[gRow * K + (bk + lCol)]; 
    B_SMEM[lRow][lCol] = B[(bk + lRow) * N + gCol];
    __syncthreads();

    /* 1-2. Compute matmul for blocks */
    for (int i = 0; i < SMEM_BS; i++) {
      tmp += A_SMEM[lRow][i] * B_SMEM[i][lCol];
    }
    __syncthreads();
  }

  /* 2. Store the results */
  C[gRow * N + gCol] = tmp;
}

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

__global__ void blocktiling_1d_kernel_v2(float *A, float *B, float *C, int M, int N, int K) {
  const int gCol = blockIdx.x * blockDim.x + threadIdx.x;
  const int gRow = (blockIdx.y * blockDim.y + threadIdx.y) * BLOCKTILING_1D_TM;
  const int lCol = threadIdx.x;
  const int lRow = threadIdx.y;

  float workPerThread[BLOCKTILING_1D_TM] = {0.f};

  /* 0. Allocate SMEM */
  __shared__ float A_SMEM[BLOCKTILING_1D_BM][BLOCKTILING_1D_BK];
  __shared__ float B_SMEM[BLOCKTILING_1D_BK][BLOCKTILING_1D_BN];

  // 1. Proceed matmul over blocks
  for (int bk = 0; bk < K; bk += BLOCKTILING_1D_BK) {

    /* 1-1. Load blocks to SMEM */
    // 1) Thread flattening (for GMEM coalescing!)
    int localIdx = threadIdx.y * blockDim.x + threadIdx.x; // 0..511

    // 2) A_SMEM [64][8]
    int rowA = localIdx / BLOCKTILING_1D_BK;  // (0..63)
    int colA = localIdx % BLOCKTILING_1D_BK;  // (0..7)
    int globalArow = (blockIdx.y * BLOCKTILING_1D_BM) + rowA;
    int globalAcol = bk + colA;
    A_SMEM[rowA][colA] = A[globalArow * K + globalAcol];

    // 3) B_SMEM [8][64]
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

__global__ void blocktiling_2d_kernel(float *A, float *B, float *C, int M, int N, int K) {
  const int gCol = (blockIdx.x * blockDim.x + threadIdx.x) * BLOCKTILING_2D_TN;
  const int gRow = (blockIdx.y * blockDim.y + threadIdx.y) * BLOCKTILING_2D_TM;
  const int lCol = threadIdx.x; // 0..15
  const int lRow = threadIdx.y; // 0..15

  float workPerThread[BLOCKTILING_2D_TM * BLOCKTILING_2D_TN] = {0.f}; // 8*8 = 64

  float regA[BLOCKTILING_2D_TM] = {0.f}; // 8
  float regB[BLOCKTILING_2D_TN] = {0.f}; // 8

  /* 0. Allocate SMEM */
  __shared__ float A_SMEM[BLOCKTILING_2D_BM][BLOCKTILING_2D_BK]; // [128][16]
  __shared__ float B_SMEM[BLOCKTILING_2D_BK][BLOCKTILING_2D_BN]; // [16][128]

  int threadInSMEM = BLOCKTILING_2D_BM * BLOCKTILING_2D_BK; // 2048
  int threadInBlock = blockDim.x * blockDim.y; // 256
  int numLoadPerThread = threadInSMEM / threadInBlock; // 8

  /* 1. Proceed matmul over blocks */
  #pragma unroll
  for (int bk = 0; bk < K; bk += BLOCKTILING_2D_BK) {

    /* 1-1. Load blocks to SMEM */
    // 1) Thread flattening
    int tid = threadIdx.y * blockDim.x + threadIdx.x; // 0..255
    
    #pragma unroll
    for (int n = 0; n < numLoadPerThread; n++) {
      int localIdx = tid + n * threadInBlock; // 0, 256, 512, 768

      // 2) A_SMEM [128 * 16] = 2048
      int rowA = localIdx / BLOCKTILING_2D_BK;  // (0..127)
      int colA = localIdx % BLOCKTILING_2D_BK;  // (0..15)
      int globalArow = (blockIdx.y * BLOCKTILING_2D_BM) + rowA;
      int globalAcol = bk + colA;
      A_SMEM[rowA][colA] = A[globalArow * K + globalAcol];
      
      // 3) B_SMEM [16 * 128] = 2048
      int rowB = localIdx / BLOCKTILING_2D_BN;  // (0..15)
      int colB = localIdx % BLOCKTILING_2D_BN;  // (0..127)
      int globalBrow = bk + rowB;
      int globalBcol = (blockIdx.x * BLOCKTILING_2D_BN) + colB; 
      B_SMEM[rowB][colB] = B[globalBrow * N + globalBcol];
    }
    __syncthreads();

    /* 1-2. Compute matmul for blocks */
    #pragma unroll
    for (int tk = 0; tk < BLOCKTILING_2D_BK; tk++) {

      // 1) Load A_SMEM to regA
      #pragma unroll
      for (int i = 0; i < BLOCKTILING_2D_TM; i++) {
        regA[i] = A_SMEM[lRow * BLOCKTILING_2D_TM + i][tk];
      }
      
      // 2) Load B_SMEM to regB
      #pragma unroll
      for (int i = 0; i < BLOCKTILING_2D_TN; i++) {
        regB[i] = B_SMEM[tk][lCol * BLOCKTILING_2D_TN + i];
      }
      
      // 3) Compute matmul
      #pragma unroll
      for (int i = 0; i < BLOCKTILING_2D_TM; i++) {
        float regAs = regA[i];
        #pragma unroll
        for (int j = 0; j < BLOCKTILING_2D_TN; j++) {
          workPerThread[i * BLOCKTILING_2D_TN + j] += regAs * regB[j];
        }
      }
    }
    __syncthreads();
  }

  /* 2. Store the results */
  #pragma unroll
  for (int i = 0; i < BLOCKTILING_2D_TM; i++) {
    #pragma unroll
    for (int j = 0; j < BLOCKTILING_2D_TN; j++) {
      C[(gRow + i) * N + gCol + j] = workPerThread[i * BLOCKTILING_2D_TN + j];
    }
  }
}

__global__ void blocktiling_2d_vec_kernel(float *A, float *B, float *C, int M, int N, int K) {
  const int gCol = (blockIdx.x * blockDim.x + threadIdx.x) * BLOCKTILING_2D_TN;
  const int gRow = (blockIdx.y * blockDim.y + threadIdx.y) * BLOCKTILING_2D_TM;
  const int lCol = threadIdx.x; // 0..15
  const int lRow = threadIdx.y; // 0..15

  float workPerThread[BLOCKTILING_2D_TM * BLOCKTILING_2D_TN] = {0.f}; // 8*8 = 64

  float regA[BLOCKTILING_2D_TM] = {0.f}; // 8
  float regB[BLOCKTILING_2D_TN] = {0.f}; // 8

  /* 0. Allocate SMEM */
  __shared__ float A_SMEM[BLOCKTILING_2D_BM][BLOCKTILING_2D_BK]; // [128][16]
  __shared__ float B_SMEM[BLOCKTILING_2D_BK][BLOCKTILING_2D_BN]; // [16][128]

  int threadInSMEM = BLOCKTILING_2D_BM * BLOCKTILING_2D_BK; // 2048
  int threadInBlock = blockDim.x * blockDim.y; // 256
  int numLoadPerThread = threadInSMEM / threadInBlock; // 8
  int numVecLoadPerThread = numLoadPerThread / 4; // 8 -> 2 (vectorized)

  /* 1. Proceed matmul over blocks */
  #pragma unroll
  for (int bk = 0; bk < K; bk += BLOCKTILING_2D_BK) {

    /* 1-1. Load blocks to SMEM */
    // 1) Thread flattening
    int tid = threadIdx.y * blockDim.x + threadIdx.x; // 0..255
    
    #pragma unroll
    for (int n = 0; n < numVecLoadPerThread; n++) {
      int localIdx = tid + n * threadInBlock; // 0, 256, 512, 768

      // 2) A_SMEM [128 * 16] = 2048
      int rowA = localIdx / (BLOCKTILING_2D_BK / 4);  // (0..3127) -> (0..31) 
      int colA = localIdx % (BLOCKTILING_2D_BK / 4);  // (0..15) -> (0..3)
      int globalArow = (blockIdx.y * BLOCKTILING_2D_BM) + rowA;
      int globalAcol = bk + colA * 4; // (0..15) -> (0..63)
      float4 tmpA = reinterpret_cast<float4*>(&A[globalArow * K + globalAcol])[0];
      reinterpret_cast<float4*>(&A_SMEM[rowA][colA * 4])[0] = tmpA;
      
      // 3) B_SMEM [16 * 128] = 2048
      int rowB = localIdx / (BLOCKTILING_2D_BN / 4);  // (0..15) -> (0..3)
      int colB = localIdx % (BLOCKTILING_2D_BN / 4);  // (0..127) -> (0..31)
      int globalBrow = bk + rowB; 
      int globalBcol = (blockIdx.x * BLOCKTILING_2D_BN) + colB * 4; // (0..31) -> (0..127)
      float4 tmpB = reinterpret_cast<float4*>(&B[globalBrow * N + globalBcol])[0];
      reinterpret_cast<float4*>(&B_SMEM[rowB][colB * 4])[0] = tmpB;
    }
    __syncthreads();

    /* 1-2. Compute matmul for blocks */
    #pragma unroll
    for (int tk = 0; tk < BLOCKTILING_2D_BK; tk++) {

      // 1) Load A_SMEM to regA
      #pragma unroll
      for (int i = 0; i < BLOCKTILING_2D_TM; i++) {
        regA[i] = A_SMEM[lRow * BLOCKTILING_2D_TM + i][tk];
      }
      
      // 2) Load B_SMEM to regB
      #pragma unroll
      for (int i = 0; i < BLOCKTILING_2D_TN; i++) {
        regB[i] = B_SMEM[tk][lCol * BLOCKTILING_2D_TN + i];
      }
      
      // 3) Compute matmul
      #pragma unroll
      for (int i = 0; i < BLOCKTILING_2D_TM; i++) {
        float regAs = regA[i];
        #pragma unroll
        for (int j = 0; j < BLOCKTILING_2D_TN; j++) {
          workPerThread[i * BLOCKTILING_2D_TN + j] += regAs * regB[j];
        }
      }
    }
    __syncthreads();
  }

  /* 2. Store the results */
  #pragma unroll
  for (int i = 0; i < BLOCKTILING_2D_TM; i++) {
    #pragma unroll
    for (int j = 0; j < BLOCKTILING_2D_TN; j+=4) {

      // Vectorized GMEM load & store of C
      float4 tmp = reinterpret_cast<float4*>(&C[(gRow + i) * N + gCol + j])[0];
      tmp.x = workPerThread[i * BLOCKTILING_2D_TN + j];
      tmp.y = workPerThread[i * BLOCKTILING_2D_TN + j + 1];
      tmp.z = workPerThread[i * BLOCKTILING_2D_TN + j + 2];
      tmp.w = workPerThread[i * BLOCKTILING_2D_TN + j + 3];
      reinterpret_cast<float4*>(&C[(gRow + i) * N + gCol + j])[0] = tmp;
    }
  }
}

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

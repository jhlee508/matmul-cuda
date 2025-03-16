#pragma once

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

#define CHECK_CUBLAS(call)                                                   \
  do {                                                                       \
    cublasStatus_t status_ = call;                                           \
    if (status_ != CUBLAS_STATUS_SUCCESS) {                                  \
      fprintf(stderr, "CUBLAS error (%s:%d): %s, %s\n", __FILE__, __LINE__,  \
              cublasGetStatusName(status_), cublasGetStatusString(status_)); \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)


#define CEIL_DIV(X, Y) (((X) + (Y) - 1) / (Y))

/* Shared Memory Caching */
#define SMEM_BS 32

/* Blocktiling 1D */
#define BLOCKTILING_1D_BM 64
#define BLOCKTILING_1D_BN 64
#define BLOCKTILING_1D_BK 8
#define BLOCKTILING_1D_TM 8 

/* Blocktiling 2D */
#define BLOCKTILING_2D_BM 128
#define BLOCKTILING_2D_BN 128
#define BLOCKTILING_2D_BK 16
#define BLOCKTILING_2D_TM 8 
#define BLOCKTILING_2D_TN 8

/* Blocktiling 2D (Vectorized) */
#define BLOCKTILING_2D_V_BM 128
#define BLOCKTILING_2D_V_BN 128
#define BLOCKTILING_2D_V_BK 16
#define BLOCKTILING_2D_V_TM 8 
#define BLOCKTILING_2D_V_TN 8


void matmul(float *_A, float *_B, float *_C, int M, int N, int K);

void matmul_cublas(float *_A, float *_B, float *_C, int M, int N, int K);

void matmul_initialize(float *_A, float *_B, int M, int N, int K);

void cublas_initialize(); 

void matmul_finalize(float *_C, int M, int N, int K);

void cublas_finalize();

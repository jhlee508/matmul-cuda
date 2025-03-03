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

#define SMEM_TILE_SIZE 32


void matmul(float *_A, float *_B, float *_C, int M, int N, int K);

void matmul_cublas(float *_A, float *_B, float *_C, int M, int N, int K);

void matmul_initialize(float *_A, float *_B, int M, int N, int K);

void cublas_initialize(); 

void matmul_finalize(float *_C, int M, int N, int K);

void cublas_finalize();

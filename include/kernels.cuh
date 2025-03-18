#pragma once

__global__ void naive_kernel(float *A, float *B, float *C, int M, int N, int K);

__global__ void gmem_coalescing_kernel(float *A, float *B, float *C, int M, int N, int K);

__global__ void smem_caching_kernel(float *A, float *B, float *C, int M, int N, int K);

__global__ void blocktiling_1d_kernel(float *A, float *B, float *C, int M, int N, int K);

__global__ void blocktiling_1d_kernel_v2(float *A, float *B, float *C, int M, int N, int K);

__global__ void blocktiling_2d_kernel(float *A, float *B, float *C, int M, int N, int K);

__global__ void blocktiling_2d_vec_kernel(float *A, float *B, float *C, int M, int N, int K);

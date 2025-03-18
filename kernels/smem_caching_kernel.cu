#include "common.h"

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
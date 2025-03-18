#include "common.h"

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
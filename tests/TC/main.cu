#include <cuda_fp16.h>
#include <mma.h>
#include <sys/time.h>
#include <stdio.h>

using namespace nvcuda;
using namespace std;

#define numThreadsPerBlock 256
#define numThreadsPerWarp 32
#define numThreadBlock 10000
#define numWMMAOps 100
#define M 16
#define N 16
#define K 16

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

__global__ void wmma_test_kernel() {
	// Warps within a block read from 256 byte aligned strided addresses 
	// to avoid bank conflicts (makes no difference).
	__shared__ __half SMEM[1024 * 8];
	__half *A = SMEM + threadIdx.y * 1024 + threadIdx.y * 16;
	__half *B = SMEM + threadIdx.y * 1024 + threadIdx.y * 16 + 256;
	__half *C = SMEM + threadIdx.y * 1024 + threadIdx.y * 16 + 512;

	/* Declare the fragments */
	// Matrix A is read once, and accumulator is filled once.
	wmma::fragment<wmma::matrix_a, M, N, K, __half, wmma::row_major> a_frag;
	wmma::fragment<wmma::matrix_b, M, N, K, __half, wmma::col_major> b_frag;
	wmma::fragment<wmma::accumulator, M, N, K, __half> acc_frag;

	/* Initialize the output to zero */
	wmma::fill_fragment(acc_frag, 0.0f);

	/* Load the inputs into the fragment */
	wmma::load_matrix_sync(a_frag, A, 16);
	wmma::load_matrix_sync(b_frag, B, 16);

#pragma unroll
	for (int i = 0; i < numWMMAOps; i++) {
		/* Perform the matrix multiplication */
		wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
	}

	/* Store the result from fragment to GMEM */
	wmma::store_matrix_sync(C, acc_frag, 16, wmma::mem_col_major);
}

void TestWMMA() {
	dim3 blockDim(numThreadsPerBlock);
	dim3 gridDim(numThreadBlock);
	wmma_test_kernel<<<gridDim, blockDim>>>();

	cudaDeviceSynchronize();
}

int main(){
	/* For warmup */
	TestWMMA();

	/* Actual test */
	double start = get_time();
	TestWMMA();
	double end = get_time();

	double elapsed_time = end - start;
	printf("> Elapsed time: %f sec\n", elapsed_time);
	printf("> TFLOPS: %.1f\n", (double)(M * N * K * 2) * 
		(numWMMAOps) * (numThreadsPerBlock / numThreadsPerWarp) * 
		(numThreadBlock) / elapsed_time / 1e12);

	return 0;
}
__global__ void gmem_coalescing_kernel(float *A, float *B, float *C, int M, int N, int K) {
	const int Col = blockIdx.x * blockDim.x + threadIdx.x;
	const int Row = blockIdx.y * blockDim.y + threadIdx.y;

	float tmp = 0.f;

	for (int k = 0; k < K; k++) {
		tmp += A[Row * K + k] * B[k * N + Col];
	}
	C[Row * N + Col] = tmp;
}
  
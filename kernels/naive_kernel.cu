__global__ void naive_kernel(float *A, float *B, float *C, int M, int N, int K) {
	const int Row = blockIdx.x * blockDim.x + threadIdx.x;
	const int Col = blockIdx.y * blockDim.y + threadIdx.y;

	float tmp = 0.f;

	for (int k = 0; k < K; k++) {
		tmp += A[Row * K + k] * B[k * N + Col];
	}
	C[Row * N + Col] = tmp;
}
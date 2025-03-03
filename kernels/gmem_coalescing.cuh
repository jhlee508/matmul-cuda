__global__ void gmem_coalescing_kernel(float *A, float *B, float *C, int M, int N, int K) {
  const int Col = blockIdx.x * blockDim.x + threadIdx.x;
  const int Row = blockIdx.y * blockDim.y + threadIdx.y;

  float tmp = 0.f;

  for (int k = 0; k < K; k++) {
    tmp += A[Row * K + k] * B[k * N + Col];
  }
  C[Row * N + Col] = tmp;
}

void matmul_gmem_coalescing() {
  
  dim3 blockDim(32, 32);
  dim3 gridDim(CEIL_DIV(N / 32), CEIL_DIV(M / 32));
  gmem_coalescing_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

  CHECK_CUDA(cudaDeviceSynchronize());
}
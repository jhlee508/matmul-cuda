#define SMEM_TILE_SIZE 32

__global__ void smem_caching_kernel(float *A, float *B, float *C, int M, int N, int K) {
  const int gCol = blockIdx.x * blockDim.x + threadIdx.x;
  const int gRow = blockIdx.y * blockDim.y + threadIdx.y;
  const int lCol = threadIdx.x;
  const int lRow = threadIdx.y;

  float tmp = 0.f;

  // 0. Allocate SMEM
  __shared__ float A_SMEM[SMEM_TILE_SIZE][SMEM_TILE_SIZE];
  __shared__ float B_SMEM[SMEM_TILE_SIZE][SMEM_TILE_SIZE];

  // 1-1. Load tiles to SMEM
  for (int bk = 0; bk < K; bk+=SMEM_TILE_SIZE) {
    A_SMEM[lRow][lCol] = A[gRow * K + (lCol + bk)];
    B_SMEM[lRow][lCol] = B[(lRow + bk) * N + gCol];

    __syncthreads();

    // 1-2. Tiled matmul
    for (int tk = 0; tk < SMEM_TILE_SIZE; tk++) {
      tmp += A_SMEM[lRow][tk] * B_SMEM[tk][lCol];
    }

    __syncthreads();
  }

  // 2. Accumulate tiling results
  C[gRow * N + gCol] = tmp;
}

void matmul_smem_caching() {
  
  // The size of the 'SMEM tile' is the same as the size of the 'thread block'
  dim3 blockDim(SMEM_TILE_SIZE, SMEM_TILE_SIZE);
  dim3 gridDim(CEIL_DIV(N / SMEM_TILE_SIZE), CEIL_DIV(M / SMEM_TILE_SIZE));
  smem_caching_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);

  CHECK_CUDA(cudaDeviceSynchronize());
}
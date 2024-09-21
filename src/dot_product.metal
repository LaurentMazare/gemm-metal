#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

#define BLOCKSIZE 32

[[kernel]]
void sgemm_naive(
  device const float *A,
  device const float *B,
  device float *C,
  constant uint32_t &M,
  constant uint32_t &N,
  constant uint32_t &K,
  constant float &alpha,
  constant float &beta,
  uint3 tgpig[[threadgroup_position_in_grid]],
  uint3 tpitg[[thread_position_in_threadgroup]],
  uint3   ntg[[threads_per_threadgroup]]
) {
  // compute position in C that this thread is responsible for
  const uint x = ntg.x * tgpig.x + tpitg.x;
  const uint y = ntg.y * tgpig.y + tpitg.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < M && y < N) {
    float tmp = 0.0;
    for (uint i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

[[kernel]]
void sgemm_coalescing(
  device const float *A,
  device const float *B,
  device float *C,
  constant uint32_t &M,
  constant uint32_t &N,
  constant uint32_t &K,
  constant float &alpha,
  constant float &beta,
  uint3 tgpig[[threadgroup_position_in_grid]],
  uint3 tpitg[[thread_position_in_threadgroup]],
  uint3   ntg[[threads_per_threadgroup]]
) {
  // compute position in C that this thread is responsible for
  const uint x = tgpig.x * BLOCKSIZE + tpitg.x / BLOCKSIZE;
  const uint y = tgpig.y * BLOCKSIZE + tpitg.x % BLOCKSIZE;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < M && y < N) {
    float tmp = 0.0;
    for (uint i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    // C = α*(A@B)+β*C
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
  }
}

[[kernel]]
void sgemm_shared_mem_block(
  device const float *A,
  device const float *B,
  device float *C,
  constant uint32_t &M,
  constant uint32_t &N,
  constant uint32_t &K,
  constant float &alpha,
  constant float &beta,
  threadgroup float* As [[threadgroup(0)]],
  threadgroup float* Bs [[threadgroup(1)]],
  uint3 tgpig[[threadgroup_position_in_grid]],
  uint3 tpitg[[thread_position_in_threadgroup]],
  uint3   ntg[[threads_per_threadgroup]]
) {
  // the output block that we want to compute in this threadblock
  const uint cRow = tgpig.x;
  const uint cCol = tgpig.y;

  // allocate buffer for current block in fast shared mem
  // shared mem is shared between all threads in a block
  // __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  // __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  // the inner row & col that we're accessing in this thread
  const uint threadCol = tpitg.x % BLOCKSIZE;
  const uint threadRow = tpitg.x / BLOCKSIZE;
  if (threadCol >= N || threadRow >= M) {
    return;
  }

  // advance pointers to the starting positions
  A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
  B += cCol * BLOCKSIZE;                        // row=0, col=cCol
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

  float tmp = 0.0;
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // Have each thread load one of the elements in A & B
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    // TODO(laurent): this copies potentially out of bound data.
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    // block threads in this block until cache is fully populated
    threadgroup_barrier(mem_flags::mem_threadgroup);
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    // execute the dotproduct on the currently cached block
    // TODO(laurent): this copies potentially out of bound data.
    for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCKSIZE + dotIdx] *
             Bs[dotIdx * BLOCKSIZE + threadCol];
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  C[threadRow * N + threadCol] =
      alpha * tmp + beta * C[threadRow * N + threadCol];
}

template <const int BM, const int BN, const int BK, const int TM>
kernel void sgemm_1d_block_tiling(
  device const float *A,
  device const float *B,
  device float *C,
  constant uint32_t &M,
  constant uint32_t &N,
  constant uint32_t &K,
  constant float &alpha,
  constant float &beta,
  threadgroup float* As [[threadgroup(0)]],
  threadgroup float* Bs [[threadgroup(1)]],
  uint3 tgpig[[threadgroup_position_in_grid]],
  uint3 tpitg[[thread_position_in_threadgroup]],
  uint3   ntg[[threads_per_threadgroup]]
) {
  // If we flip x and y here we get ~30% less performance for large matrices.
  // The current, 30% faster configuration ensures that blocks with sequential
  // blockIDs access columns of B sequentially, while sharing the same row of A.
  // The slower configuration would share columns of A, but access into B would
  // be non-sequential. So the faster configuration has better spatial locality
  // and hence a greater L2 hit rate.
  const uint cRow = tgpig.y;
  const uint cCol = tgpig.x;

  // each warp will calculate 32*TM elements, with 32 being the columnar dim.
  const int threadCol = tpitg.x % BN;
  const int threadRow = tpitg.x / BN;

  // allocate space for the current blocktile in SMEM
  // __shared__ float As[BM * BK];
  // __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // todo: adjust this to each thread to load multiple entries and
  // better exploit the cache sizes
  assert(BM * BK == blockDim.x);
  assert(BN * BK == blockDim.x);
  const uint innerColA = tpitg.x % BK; // warp-level GMEM coalescing
  const uint innerRowA = tpitg.x / BK;
  const uint innerColB = tpitg.x % BN; // warp-level GMEM coalescing
  const uint innerRowB = tpitg.x / BN;

  // allocate thread-local cache for results in registerfile
  float threadResults[TM] = {0.0};

  // outer loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // advance blocktile
    A += BK;
    B += BK * N;

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // we make the dotproduct loop the outside loop, which facilitates
      // reuse of the Bs entry, which we can cache in a tmp var.
      float tmpB = Bs[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // write out the results
  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] =
        alpha * threadResults[resIdx] +
        beta * C[(threadRow * TM + resIdx) * N + threadCol];
  }
}

typedef void (sgemm_shared_ab)(
  device const float *A,
  device const float *B,
  device float *C,
  constant uint32_t &M,
  constant uint32_t &N,
  constant uint32_t &K,
  constant float &alpha,
  constant float &beta,
  threadgroup float* As [[threadgroup(0)]],
  threadgroup float* Bs [[threadgroup(1)]],
  uint3, uint3, uint3);

template [[host_name("sgemm_1d_bt_64_64_8_8")]] kernel sgemm_shared_ab sgemm_1d_block_tiling<64, 64, 8, 8>;

template <const int BM, const int BN, const int BK, const int TM, const int TN>
kernel void sgemm_2d_block_tiling(
  device const float *A,
  device const float *B,
  device float *C,
  constant uint32_t &M,
  constant uint32_t &N,
  constant uint32_t &K,
  constant float &alpha,
  constant float &beta,
  threadgroup float* As [[threadgroup(0)]],
  threadgroup float* Bs [[threadgroup(1)]],
  uint3 tgpig[[threadgroup_position_in_grid]],
  uint3 tpitg[[thread_position_in_threadgroup]],
  uint3   ntg[[threads_per_threadgroup]]
) {
  const uint cRow = tgpig.y;
  const uint cCol = tgpig.x;

  const uint totalResultsBlocktile = BM * BN;
  // A thread is responsible for calculating TM*TN elements in the blocktile
  const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

  // ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
  assert(numThreadsBlocktile == blockDim.x);

  // BN/TN are the number of threads to span a column
  const int threadCol = tpitg.x % (BN / TN);
  const int threadRow = tpitg.x / (BN / TN);

  // allocate space for the current blocktile in smem
  // __shared__ float As[BM * BK];
  // __shared__ float Bs[BK * BN];

  // Move blocktile to beginning of A's row and B's column
  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  // calculating the indices that this thread will load into SMEM
  const uint innerRowA = tpitg.x / BK;
  const uint innerColA = tpitg.x % BK;
  // calculates the number of rows of As that are being loaded in a single step
  // by a single block
  const uint strideA = numThreadsBlocktile / BK;
  const uint innerRowB = tpitg.x / BN;
  const uint innerColB = tpitg.x % BN;
  // for both As and Bs we want each load to span the full column-width, for
  // better GMEM coalescing (as opposed to spanning full row-width and iterating
  // across columns)
  const uint strideB = numThreadsBlocktile / BN;

  // allocate thread-local cache for results in registerfile
  // TODO(laurent): this doesn't seem to fit in registers with TM=8, TN=8 for a M3 Max,
  // so we might have to reduce the sizes.
  float threadResults[TM * TN] = {0.0};
  // register caches for As and Bs
  float regM[TM] = {0.0};
  float regN[TN] = {0.0};

  // outer-most loop over block tiles
  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      As[(innerRowA + loadOffset) * BK + innerColA] =
          A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      Bs[(innerRowB + loadOffset) * BN + innerColB] =
          B[(innerRowB + loadOffset) * N + innerColB];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
      }
      for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
      }
      for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] +=
              regM[resIdxM] * regN[resIdxN];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // write out the results
  for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
      C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
          alpha * threadResults[resIdxM * TN + resIdxN] +
          beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
    }
  }
}

template [[host_name("sgemm_2d_bt_64_64_8_8_8")]] kernel sgemm_shared_ab sgemm_2d_block_tiling<64, 64, 8, 8, 8>;
template [[host_name("sgemm_2d_bt_64_64_8_4_4")]] kernel sgemm_shared_ab sgemm_2d_block_tiling<64, 64, 8, 4, 4>;
template [[host_name("sgemm_2d_bt_64_64_8_8_4")]] kernel sgemm_shared_ab sgemm_2d_block_tiling<64, 64, 8, 8, 4>;
template [[host_name("sgemm_2d_bt_64_64_8_4_8")]] kernel sgemm_shared_ab sgemm_2d_block_tiling<64, 64, 8, 4, 8>;

[[kernel]]
void sgemm_naive_simd(
  device const float *data1,
  device const float *data2,
  device float *a,
  constant uint32_t &M,
  constant uint32_t &N,
  constant uint32_t &K,
  constant float &alpha, // TODO: unused, assumed to be 1
  constant float &beta, // TODO: unused, assumed to be 0
  uint3 gid[[threadgroup_position_in_grid]],
  uint3 lid[[thread_position_in_threadgroup]],
  uint3 ntg[[threads_per_threadgroup]]
) {
  a += gid.x * 8 * N + (gid.y * ntg.y + lid.y) * 8;
  data1 += gid.x * 8 * K;
  data2 += (gid.y * ntg.y + lid.y) * 8;
  simdgroup_float8x8 acc = simdgroup_float8x8(0);
  simdgroup_float8x8 A;
  simdgroup_float8x8 B;
  for (uint k = 0; k < K; k+=8) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_load(A, data1 + k, K, ulong2(0, 0));
    simdgroup_load(B, data2 + k*N, N, ulong2(0, 0));
    simdgroup_multiply_accumulate(acc, A, B, acc);
  }
  simdgroup_store(acc, a, N, ulong2(0, 0));
}

typedef void (sgemm_simple_sig)(
  device const float *A,
  device const float *B,
  device float *C,
  constant uint32_t &M,
  constant uint32_t &N,
  constant uint32_t &K,
  constant float &alpha,
  constant float &beta,
  uint3, uint3, uint3);


// https://github.com/tinygrad/tinygrad/blob/750696a0269d87f09f7d95da71b71f9ea7dc3a7e/extra/gemm/metal_matmul.py#L33
template <const short NBLK>
kernel void sgemm_tiled_simd(
  device const float *data1,
  device const float *data2,
  device float *a,
  constant uint32_t &M,
  constant uint32_t &N,
  constant uint32_t &K,
  constant float &alpha, // TODO: unused, assumed to be 1
  constant float &beta, // TODO: unused, assumed to be 0
  uint3 gid[[threadgroup_position_in_grid]],
  uint3 lid[[thread_position_in_threadgroup]],
  uint3 ntg[[threads_per_threadgroup]]
) {
  short block_size = NBLK * 8;
  a += gid.x * block_size * N + (gid.y * ntg.y + lid.y) * block_size;
  data1 += gid.x * block_size * K;
  data2 += (gid.y * ntg.y + lid.y) * block_size;

  simdgroup_float8x8 acc[NBLK][NBLK];
  #pragma unroll(NBLK)
  for (uint i = 0; i < NBLK; i++) {
    #pragma unroll(NBLK)
    for (uint j = 0; j < NBLK; j++) {
      acc[i][j] = simdgroup_float8x8(0);
    }
  }
  simdgroup_float8x8 A[NBLK];
  simdgroup_float8x8 B[NBLK];
  for (uint k = 0; k < K; k += 8) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    #pragma unroll(NBLK)
    for (uint i = 0; i < NBLK; ++i) {
      simdgroup_load(A[i], data1+k+i*8*K, K, ulong2(0, 0));
      simdgroup_load(B[i], data2+8*i+k*N, N, ulong2(0, 0));
    }

    #pragma unroll(NBLK)
    for (uint i = 0; i < NBLK; ++i) {
      #pragma unroll(NBLK)
      for (uint j = 0; j < NBLK; ++j) {
        simdgroup_multiply_accumulate(acc[i][j], A[j], B[i], acc[i][j]);
      }
    }
  }
  #pragma unroll(NBLK)
  for (uint i = 0; i < NBLK; ++i) {
    #pragma unroll(NBLK)
    for (uint j = 0; j < NBLK; ++j) {
      simdgroup_store(acc[i][j], a+8*i+8*j*N, N, ulong2(0, 0));
    }
  }
}

template [[host_name("sgemm_tiled_simd1")]] kernel sgemm_simple_sig sgemm_tiled_simd<1>;
template [[host_name("sgemm_tiled_simd2")]] kernel sgemm_simple_sig sgemm_tiled_simd<2>;
template [[host_name("sgemm_tiled_simd4")]] kernel sgemm_simple_sig sgemm_tiled_simd<4>;
template [[host_name("sgemm_tiled_simd8")]] kernel sgemm_simple_sig sgemm_tiled_simd<8>;

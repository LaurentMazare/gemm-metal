#include <metal_stdlib>
using namespace metal;

#define BLOCKSIZE 32

[[kernel]]
void dot_product(
  device const float *a,
  device const float *b,
  device float *c,
  uint3 tpig[[thread_position_in_grid]])
{
  int index = tpig.x;
  c[index] = a[index] * b[index];
}

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
    for (int i = 0; i < K; ++i) {
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
    for (int i = 0; i < K; ++i) {
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

  // advance pointers to the starting positions
  A += cRow * BLOCKSIZE * K;                    // row=cRow, col=0
  B += cCol * BLOCKSIZE;                        // row=0, col=cCol
  C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE; // row=cRow, col=cCol

  float tmp = 0.0;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
    // Have each thread load one of the elements in A & B
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

    // block threads in this block until cache is fully populated
    threadgroup_barrier(mem_flags::mem_threadgroup);
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    // execute the dotproduct on the currently cached block
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

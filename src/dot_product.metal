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
  const uint x = tpitg.x; // ntg.x * tgpig.x + tpitg.x;
  const uint y = tpitg.y; // ntg.y * tgpig.y + tpitg.y;

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
  const uint y = tgpig.y + tpitg.y % BLOCKSIZE;

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

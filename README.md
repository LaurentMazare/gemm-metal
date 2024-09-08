# gemm-metal

This repo contains some metal implementation for the kernels and techniques
described in the amazing blog post [How to Optimize a CUDA Matmul Kernel for
cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM).
The original cuda implementation can be found in
[siboehm/SGEMM_CUDA](https://github.com/siboehm/SGEMM_CUDA).

This code has been written so has to get more familiar with the metal api so
the kernels are certainly naive / a work in progress.

## Benchmarks

All the benchmarks below are for rev `247ddaa`. Numbers are in GFLOPS.

*MacBook Air 16GB 2024 (10 GPU cores)*

| Kernel | 512 | 1024 | 2048 | 4096 |
| ------ | --- | ---- | ---- | ---- |
| Naive      | 72 | 104 | 106 | 112 |
| Coalescing | 203 | 247 | 213 | 209 |
| SharedMem  | 326 | 453 | 477 | 474 |
| Tiling1D   | 405 | 664 | 729 | 736 |
| Tiling2D   | 601 | 1090 | 1217 | 1220 |


*MacBook Pro 14" 16GB 2023 (16 GPU cores)*

| Kernel | 512 | 1024 | 2048 | 4096 | 6144 | 8192 |
| ------ | --- | ---- | ---- | ---- | ---- | ---- |
| Naive  | 39 | 35 | 34 | 51 | 57 | 65 |
| Coalescing | 256 | 351 | 348 | 289 | 287 | 279 |
| SharedMem | 378 | 492 | 475 | 479 | 418 | 434 |
| Tiling1D | 583 | 925 | 979 | 1015 | 1009 | 1016 |
| Tiling2D | 778 | 1319 | 1487 | 1619 | 1646 | 1658 |

*MacBook Pro 14" 36GB 2023 (30 GPU cores)*

| Kernel | 512 | 1024 | 2048 | 4096 | 6144 | 8192 |
| ------ | --- | ---- | ---- | ---- | ---- | ---- |
| Naive  | 162 | 385 | 345 | 340 | 286 | 366 |
| Coalescing | 456 | 772 | 701 | 516 | 517 | 511 |
| SharedMem | 660 | 1276 | 1467 | 1443 | 1489 | 1484 |
| Tiling1D | 722 | 1591 | 2131 | 2157 | 2284 | 2298 |
| Tiling2D | 885 | 2530 | 3510 | 3603 | 3806 | 3894 |


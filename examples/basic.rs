fn main() -> anyhow::Result<()> {
    objc::rc::autoreleasepool(|| {
        for sz in [2, 4, 8, 16, 32, 33, 35, 63, 64, 65, 512, 1024, 2048, 4096, 4096 + 2048, 8192] {
            if sz < 100 {
                gemm_metal::gemm_naive_check(sz, sz, sz, 20)?;
                gemm_metal::gemm_coalescing_check(sz, sz, sz, 20)?;
            } else {
                gemm_metal::gemm_naive(sz, sz, sz, 20)?;
                gemm_metal::gemm_coalescing(sz, sz, sz, 20)?;
            }
        }
        Ok(())
    })
}

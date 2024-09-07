fn main() -> anyhow::Result<()> {
    objc::rc::autoreleasepool(|| {
        for sz in [2, 512, 1024, 2048, 4096, 4096 + 2048, 8192] {
            gemm_metal::gemm_naive(sz, sz, sz, 20)?;
            gemm_metal::gemm_coalescing(sz, sz, sz, 20)?;
        }
        Ok(())
    })
}

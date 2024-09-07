fn main() -> anyhow::Result<()> {
    objc::rc::autoreleasepool(|| {
        gemm_metal::dot_product()?;
        for sz in [512, 1024, 2048, 4096, 4096 + 2048, 8192] {
            gemm_metal::gemm_naive(sz, sz, sz, 100)?;
            gemm_metal::gemm_coalescing(sz, sz, sz, 100)?;
        }
        Ok(())
    })
}

fn main() -> anyhow::Result<()> {
    objc::rc::autoreleasepool(|| {
        gemm_metal::dot_product()?;
        gemm_metal::gemm_naive(1024, 1024)?;
        Ok(())
    })
}

fn main() -> anyhow::Result<()> {
    objc::rc::autoreleasepool(|| {
        gemm_metal::dot_product()?;
        Ok(())
    })
}

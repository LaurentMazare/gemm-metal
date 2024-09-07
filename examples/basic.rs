use metal::Device;

fn main() -> anyhow::Result<()> {
    objc::rc::autoreleasepool(|| {
        let device = Device::system_default().expect("No device found");
        println!("LocationNumber:             {}", device.location_number());
        println!("IsLowPower:                 {}", device.is_low_power());
        println!("MaxThreadgroupMemoryLength: {}", device.max_threadgroup_memory_length());
        println!("MaxThreadsPerThreadgroup:   {:?}", device.max_threads_per_threadgroup());
        println!("MaxTransferRate:            {}", device.max_transfer_rate());
        println!("MaxBufferLength:            {}", device.max_buffer_length());

        for sz in [2, 4, 8, 16, 32, 33, 35, 63, 64, 65, 512, 1024, 2048, 4096, 4096 + 2048, 8192] {
            let cnt = if sz < 100 {
                100
            } else if sz < 2048 {
                20
            } else if sz < 4096 {
                10
            } else {
                5
            };
            if sz < 100 {
                gemm_metal::gemm_naive_check(sz, sz, sz, cnt)?;
                gemm_metal::gemm_coalescing_check(sz, sz, sz, cnt)?;
            } else {
                gemm_metal::gemm_naive(sz, sz, sz, cnt)?;
                gemm_metal::gemm_coalescing(sz, sz, sz, cnt)?;
            }
        }
        Ok(())
    })
}

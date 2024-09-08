use gemm_metal::{GemmKernel, Matrix};
use metal::Device;

fn run_bench<K: gemm_metal::GemmKernel>(
    n: usize,
    repeats: usize,
    check: bool,
) -> anyhow::Result<()> {
    let device = match metal::Device::system_default() {
        Some(device) => device,
        None => anyhow::bail!("no default device found"),
    };

    let pl = gemm_metal::pipeline::<K>()?;
    let (m, k) = (n, n);

    let a: Matrix<f32> = Matrix::randn(&device, m, k);
    let b: Matrix<f32> = Matrix::randn(&device, k, n);
    let c: Matrix<f32> = Matrix::zeros(&device, m, n);

    let cq = device.new_command_queue();
    let start_time = std::time::Instant::now();
    for _ in 0..repeats {
        gemm_metal::mm_sync::<K>(&a, &b, &c, &pl, &cq)?;
    }
    let dt = start_time.elapsed().as_secs_f64() / repeats as f64;
    let gflops = (n * m * k) as f64 * 2. / (1e9 * dt);
    println!("{:24} {m:5} {gflops:.2}", K::NAME);

    if check {
        gemm_metal::mm_check(&a, &b, &c, m, n, k)
    }
    Ok(())
}

fn main() -> anyhow::Result<()> {
    objc::rc::autoreleasepool(|| {
        let device = Device::system_default().expect("No device found");
        println!("LocationNumber:             {}", device.location_number());
        println!("IsLowPower:                 {}", device.is_low_power());
        println!("MaxThreadgroupMemoryLength: {}", device.max_threadgroup_memory_length());
        println!("MaxThreadsPerThreadgroup:   {:?}", device.max_threads_per_threadgroup());
        println!("MaxTransferRate:            {}", device.max_transfer_rate());
        println!("MaxBufferLength:            {}", device.max_buffer_length());

        for sz in [32, 64, 128, 256, 512, 1024, 2048, 4096, 4096 + 2048, 8192] {
            let cnt = if sz < 100 {
                100
            } else if sz < 2048 {
                20
            } else if sz < 4096 {
                10
            } else {
                5
            };
            if sz <= 256 {
                // gemm_metal::gemm_naive_check(sz, sz, sz, cnt)?;
                // gemm_metal::gemm_coalescing_check(sz, sz, sz, cnt)?;
                // gemm_metal::gemm_shared_mem_block_check(sz, sz, sz, cnt)?;
                run_bench::<gemm_metal::Tiling1D>(sz, cnt, true)?;
                run_bench::<gemm_metal::Tiling2D>(sz, cnt, true)?;
            } else {
                run_bench::<gemm_metal::Naive>(sz, cnt, false)?;
                run_bench::<gemm_metal::Coalescing>(sz, cnt, false)?;
                run_bench::<gemm_metal::SharedMem>(sz, cnt, false)?;
                run_bench::<gemm_metal::Tiling1D>(sz, cnt, false)?;
                run_bench::<gemm_metal::Tiling2D>(sz, cnt, false)?;
            }
        }
        Ok(())
    })
}

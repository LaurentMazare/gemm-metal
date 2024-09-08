use gemm_metal::{GemmKernel, Matrix};
use metal::Device;

fn check<K: GemmKernel>(n: usize) -> anyhow::Result<f32> {
    let device = match metal::Device::system_default() {
        Some(device) => device,
        None => anyhow::bail!("no default device found"),
    };

    let (m, k) = (n, n);

    let a: Matrix<f32> = Matrix::randn(&device, m, k);
    let b: Matrix<f32> = Matrix::randn(&device, k, n);
    let c: Matrix<f32> = Matrix::zeros(&device, m, n);

    let pl = gemm_metal::pipeline::<K>()?;
    let cq = device.new_command_queue();
    gemm_metal::mm_sync::<K>(&a, &b, &c, &pl, &cq)?;
    let c_vec = c.to_vec();
    let pl = gemm_metal::pipeline::<gemm_metal::Naive>()?;
    let cq = device.new_command_queue();
    gemm_metal::mm_sync::<gemm_metal::Naive>(&a, &b, &c, &pl, &cq)?;
    let c_ref = c.to_vec();
    let max_diff = c_vec
        .iter()
        .zip(c_ref.iter())
        .map(|(v1, v2)| (v1 - v2).abs())
        .max_by(f32::total_cmp)
        .unwrap();
    Ok(max_diff)
}

fn run_bench<K: GemmKernel>(n: usize, repeats: usize) -> anyhow::Result<f64> {
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
    Ok(gflops)
}

const SIZES_TO_CHECK: &[usize] = &[8, 32, 64, 128, 256];
const SIZES_TO_BENCH: &[usize] = &[64, 128, 256, 512, 1024, 2048, 4096, 4096 + 2048, 8192];

fn run_benchs<K: GemmKernel>() -> anyhow::Result<()> {
    use std::io::Write;

    print!("{:26} ", K::NAME);
    for &sz in SIZES_TO_BENCH {
        let repeats = if sz < 100 {
            100
        } else if sz < 2048 {
            20
        } else if sz < 4096 {
            10
        } else {
            5
        };
        let gflops = run_bench::<K>(sz, repeats)?;
        print!("{gflops:6.0} ");
        std::io::stdout().flush()?;
    }
    println!();
    Ok(())
}

fn run_checks<K: GemmKernel>() -> anyhow::Result<()> {
    for &sz in SIZES_TO_CHECK {
        let diff = check::<K>(sz)?;
        if diff.is_nan() || diff > 1e-5 {
            println!("DIFF SPOTTED {} {sz} {diff}", K::NAME);
        }
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

        run_checks::<gemm_metal::TiledSimd>()?;
        run_checks::<gemm_metal::NaiveSimd>()?;
        run_checks::<gemm_metal::Tiling2D>()?;
        run_checks::<gemm_metal::Tiling1D>()?;
        run_checks::<gemm_metal::SharedMem>()?;
        run_checks::<gemm_metal::Coalescing>()?;
        run_checks::<gemm_metal::SharedMem>()?;

        print!("{:26} ", "");
        for &sz in SIZES_TO_BENCH {
            print!("{sz:6} ");
        }
        println!();
        run_benchs::<gemm_metal::TiledSimd>()?;
        run_benchs::<gemm_metal::NaiveSimd>()?;
        run_benchs::<gemm_metal::Tiling2D>()?;
        run_benchs::<gemm_metal::Tiling1D>()?;
        run_benchs::<gemm_metal::SharedMem>()?;
        run_benchs::<gemm_metal::Coalescing>()?;
        run_benchs::<gemm_metal::Naive>()?;

        Ok(())
    })
}

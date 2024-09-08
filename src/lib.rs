// https://siboehm.com/articles/22/CUDA-MMM
use anyhow::{Error as E, Result};

const DOT: &str = include_str!("dot_product.metal");

mod utils;

pub struct Matrix<T> {
    buffer: metal::Buffer,
    _phantom: std::marker::PhantomData<T>,
    rows: usize,
    cols: usize,
}

impl<T> Matrix<T> {
    fn numels(&self) -> usize {
        self.rows * self.cols
    }
}

impl Matrix<f32> {
    fn randn(device: &metal::Device, rows: usize, cols: usize) -> Self {
        use rand_distr::Distribution;

        let options = metal::MTLResourceOptions::StorageModeManaged;
        let bytes_len = (rows * cols * std::mem::size_of::<f32>()) as u64;
        let mut rng = rand::thread_rng();
        let normal = rand_distr::Normal::new(0., 1.).unwrap();
        let data = (0..(rows * cols)).map(|_| normal.sample(&mut rng)).collect::<Vec<f32>>();
        let ptr = data.as_ptr() as *const std::ffi::c_void;
        let buffer = device.new_buffer_with_data(ptr, bytes_len, options);
        Self { buffer, _phantom: std::marker::PhantomData, rows, cols }
    }
}

impl<T: Clone> Matrix<T> {
    fn zeros(device: &metal::Device, rows: usize, cols: usize) -> Self {
        let options = metal::MTLResourceOptions::StorageModeManaged;
        let bytes_len = (rows * cols * std::mem::size_of::<T>()) as u64;
        let buffer = device.new_buffer(bytes_len, options);
        Self { buffer, _phantom: std::marker::PhantomData, rows, cols }
    }

    fn new(device: &metal::Device, data: &[T], rows: usize, cols: usize) -> Result<Self> {
        let options = metal::MTLResourceOptions::StorageModeManaged;
        let ptr = data.as_ptr() as *const std::ffi::c_void;
        let size = std::mem::size_of_val(data) as u64;
        let buffer = device.new_buffer_with_data(ptr, size, options);
        Ok(Self { buffer, _phantom: std::marker::PhantomData, rows, cols })
    }

    fn to_vec(&self) -> Vec<T> {
        let ptr = self.buffer.contents() as *const T;
        assert!(!ptr.is_null());
        let slice = unsafe { std::slice::from_raw_parts(ptr, self.numels()) };
        slice.to_vec()
    }

    fn buffer(&self) -> &metal::Buffer {
        &self.buffer
    }
}

fn mm_check(a: &Matrix<f32>, b: &Matrix<f32>, c: &Matrix<f32>, m: usize, n: usize, k: usize) {
    let a = a.to_vec();
    let b = b.to_vec();
    let c = c.to_vec();
    let mut cc = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            for i_k in 0..k {
                cc[i * n + j] += a[i * k + i_k] * b[i_k * n + j]
            }
        }
    }
    let max_diff = c
        .iter()
        .zip(cc.iter())
        .map(|(v1, v2)| (v1 - v2).abs())
        .max_by(|v1, v2| f32::total_cmp(v1, v2))
        .unwrap();
    println!("N-DIFF {max_diff}");
}

fn launch_gemm(
    kernel_name: &str,
    m: usize,
    n: usize,
    k: usize,
    repeats: usize,
    check: bool,
    grid_size: metal::MTLSize,
    threadgroup_size: metal::MTLSize,
    shared_mems: &[u64],
) -> anyhow::Result<()> {
    use metal::Device;

    let device = Device::system_default().expect("No device found");
    let lib = device.new_library_with_source(DOT, &metal::CompileOptions::new()).map_err(E::msg)?;
    let function = lib.get_function(kernel_name, None).map_err(E::msg)?;
    let pipeline = device.new_compute_pipeline_state_with_function(&function).map_err(E::msg)?;
    let a: Matrix<f32> = Matrix::randn(&device, m, k);
    let b: Matrix<f32> = Matrix::randn(&device, k, n);
    let c: Matrix<f32> = Matrix::zeros(&device, m, n);
    let command_queue = device.new_command_queue();
    let start_time = std::time::Instant::now();
    for _ in 0..repeats {
        let (a, b, c) = (a.buffer(), b.buffer(), c.buffer());
        let cb = command_queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        set_params!(encoder, (a, b, c, m, n, k, 1., 0.));
        encoder.use_resource(a, metal::MTLResourceUsage::Read);
        encoder.use_resource(b, metal::MTLResourceUsage::Read);
        encoder.use_resource(c, metal::MTLResourceUsage::Write);
        for (sm_idx, sm) in shared_mems.iter().enumerate() {
            encoder.set_threadgroup_memory_length(
                sm_idx as u64,
                (*sm as usize * std::mem::size_of::<f32>()) as u64,
            );
        }
        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        // Somehow, using dispatch_threads with non-even group size doesn't work properly here.
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
    let dt = start_time.elapsed().as_secs_f64() / repeats as f64;
    let gflops = (n * m * k) as f64 * 2. / (1e9 * dt);
    println!("{kernel_name:24} {m} {n} {k} {gflops:.2}");

    if check {
        mm_check(&a, &b, &c, m, n, k)
    }
    Ok(())
}

pub fn gemm_naive(m: usize, n: usize, k: usize, repeats: usize) -> anyhow::Result<()> {
    gemm_naive_(m, n, k, repeats, false)
}
pub fn gemm_naive_check(m: usize, n: usize, k: usize, repeats: usize) -> anyhow::Result<()> {
    gemm_naive_(m, n, k, repeats, true)
}

fn gemm_naive_(m: usize, n: usize, k: usize, repeats: usize, check: bool) -> anyhow::Result<()> {
    let grid_size = metal::MTLSize::new(m.div_ceil(32) as u64, n.div_ceil(32) as u64, 1);
    let threadgroup_size = metal::MTLSize::new(32, 32, 1);
    launch_gemm("sgemm_naive", m, n, k, repeats, check, grid_size, threadgroup_size, &[])
}

pub fn gemm_coalescing(m: usize, n: usize, k: usize, repeats: usize) -> anyhow::Result<()> {
    gemm_coalescing_(m, n, k, repeats, false)
}

pub fn gemm_coalescing_check(m: usize, n: usize, k: usize, repeats: usize) -> anyhow::Result<()> {
    gemm_coalescing_(m, n, k, repeats, true)
}

fn gemm_coalescing_(
    m: usize,
    n: usize,
    k: usize,
    repeats: usize,
    check: bool,
) -> anyhow::Result<()> {
    let grid_size = metal::MTLSize::new(m.div_ceil(32) as u64, n.div_ceil(32) as u64, 1);
    let threadgroup_size = metal::MTLSize::new(32 * 32, 1, 1);
    launch_gemm("sgemm_coalescing", m, n, k, repeats, check, grid_size, threadgroup_size, &[])
}

pub fn gemm_shared_mem_block(m: usize, n: usize, k: usize, repeats: usize) -> anyhow::Result<()> {
    gemm_shared_mem_block_(m, n, k, repeats, false)
}

pub fn gemm_shared_mem_block_check(
    m: usize,
    n: usize,
    k: usize,
    repeats: usize,
) -> anyhow::Result<()> {
    gemm_shared_mem_block_(m, n, k, repeats, true)
}

// This is only correct when the block size is divisible by 32.
fn gemm_shared_mem_block_(
    m: usize,
    n: usize,
    k: usize,
    repeats: usize,
    check: bool,
) -> anyhow::Result<()> {
    let grid_size = metal::MTLSize::new(m.div_ceil(32) as u64, n.div_ceil(32) as u64, 1);
    let threadgroup_size = metal::MTLSize::new(32 * 32, 1, 1);
    launch_gemm(
        "sgemm_shared_mem_block",
        m,
        n,
        k,
        repeats,
        check,
        grid_size,
        threadgroup_size,
        &[32 * 32, 32 * 32],
    )
}

pub fn gemm_1d_tiling(m: usize, n: usize, k: usize, repeats: usize) -> anyhow::Result<()> {
    gemm_1d_tiling_(m, n, k, repeats, false)
}

pub fn gemm_1d_tiling_check(m: usize, n: usize, k: usize, repeats: usize) -> anyhow::Result<()> {
    gemm_1d_tiling_(m, n, k, repeats, true)
}

// This is only correct when the block size is divisible by 64.
fn gemm_1d_tiling_(
    m: usize,
    n: usize,
    k: usize,
    repeats: usize,
    check: bool,
) -> anyhow::Result<()> {
    const BM: u64 = 64;
    const BN: u64 = 64;
    const BK: u64 = 8;
    const TM: u64 = 8;
    let grid_size = metal::MTLSize::new((n as u64).div_ceil(BN), (m as u64).div_ceil(BM), 1);
    let threadgroup_size = metal::MTLSize::new((BM * BN) / TM, 1, 1);
    launch_gemm(
        "sgemm_1d_bt_64_64_8_8",
        m,
        n,
        k,
        repeats,
        check,
        grid_size,
        threadgroup_size,
        &[BM * BK, BK * BN],
    )
}

pub fn gemm_2d_tiling(m: usize, n: usize, k: usize, repeats: usize) -> anyhow::Result<()> {
    gemm_2d_tiling_(m, n, k, repeats, false)
}

pub fn gemm_2d_tiling_check(m: usize, n: usize, k: usize, repeats: usize) -> anyhow::Result<()> {
    gemm_2d_tiling_(m, n, k, repeats, true)
}

// This is only correct when the block size is divisible by 64.
fn gemm_2d_tiling_(
    m: usize,
    n: usize,
    k: usize,
    repeats: usize,
    check: bool,
) -> anyhow::Result<()> {
    // Maybe this should use 128 rather than 64?
    // https://github.com/siboehm/SGEMM_CUDA/blob/60cba6f9b20a198116c76f18de8047f44df8c8b8/src/runner.cu#L198
    const BM: u64 = 64;
    const BN: u64 = 64;
    const BK: u64 = 8;
    const TM: u64 = 4;
    const TN: u64 = 4;
    let grid_size = metal::MTLSize::new((n as u64).div_ceil(BN), (m as u64).div_ceil(BM), 1);
    let threadgroup_size = metal::MTLSize::new((BM * BN) / (TN * TM), 1, 1);
    launch_gemm(
        "sgemm_2d_bt_64_64_8_4_4",
        m,
        n,
        k,
        repeats,
        check,
        grid_size,
        threadgroup_size,
        &[BM * BK, BK * BN],
    )
}

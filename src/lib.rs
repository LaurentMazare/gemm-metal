// https://siboehm.com/articles/22/CUDA-MMM
use anyhow::{Error as E, Result};

const DOT: &str = include_str!("dot_product.metal");
pub mod candle;
mod utils;

pub struct Matrix<T> {
    buffer: metal::Buffer,
    _phantom: std::marker::PhantomData<T>,
    rows: usize,
    cols: usize,
}

impl<T> Matrix<T> {
    pub fn numels(&self) -> usize {
        self.rows * self.cols
    }
}

impl Matrix<f32> {
    pub fn randn(device: &metal::Device, rows: usize, cols: usize) -> Self {
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
    pub fn zeros(device: &metal::Device, rows: usize, cols: usize) -> Self {
        let options = metal::MTLResourceOptions::StorageModeManaged;
        let bytes_len = (rows * cols * std::mem::size_of::<T>()) as u64;
        let buffer = device.new_buffer(bytes_len, options);
        Self { buffer, _phantom: std::marker::PhantomData, rows, cols }
    }

    pub fn new(device: &metal::Device, data: &[T], rows: usize, cols: usize) -> Result<Self> {
        let options = metal::MTLResourceOptions::StorageModeManaged;
        let ptr = data.as_ptr() as *const std::ffi::c_void;
        let size = std::mem::size_of_val(data) as u64;
        let buffer = device.new_buffer_with_data(ptr, size, options);
        Ok(Self { buffer, _phantom: std::marker::PhantomData, rows, cols })
    }

    pub fn to_vec(&self) -> Vec<T> {
        let ptr = self.buffer.contents() as *const T;
        assert!(!ptr.is_null());
        let slice = unsafe { std::slice::from_raw_parts(ptr, self.numels()) };
        slice.to_vec()
    }

    pub fn buffer(&self) -> &metal::Buffer {
        &self.buffer
    }
}

pub fn mm_check(a: &Matrix<f32>, b: &Matrix<f32>, c: &Matrix<f32>, m: usize, n: usize, k: usize) {
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
    let max_diff =
        c.iter().zip(cc.iter()).map(|(v1, v2)| (v1 - v2).abs()).max_by(f32::total_cmp).unwrap();
    println!("N-DIFF {max_diff}");
}

pub trait GemmKernel {
    const NAME: &'static str;
    const SHARED_MEM: &'static [u64];

    fn grid_size(m: usize, n: usize) -> metal::MTLSize;
    fn threadgroup_size(m: usize, n: usize) -> metal::MTLSize;
}

pub struct Naive;
impl GemmKernel for Naive {
    const NAME: &'static str = "sgemm_naive";
    const SHARED_MEM: &'static [u64] = &[];
    fn grid_size(m: usize, n: usize) -> metal::MTLSize {
        metal::MTLSize::new(m.div_ceil(32) as u64, n.div_ceil(32) as u64, 1)
    }
    fn threadgroup_size(_: usize, _: usize) -> metal::MTLSize {
        metal::MTLSize::new(32, 32, 1)
    }
}

pub struct Coalescing;
impl GemmKernel for Coalescing {
    const NAME: &'static str = "sgemm_coalescing";
    const SHARED_MEM: &'static [u64] = &[];
    fn grid_size(m: usize, n: usize) -> metal::MTLSize {
        metal::MTLSize::new(m.div_ceil(32) as u64, n.div_ceil(32) as u64, 1)
    }
    fn threadgroup_size(_: usize, _: usize) -> metal::MTLSize {
        metal::MTLSize::new(32 * 32, 1, 1)
    }
}

pub struct SharedMem;
impl GemmKernel for SharedMem {
    const NAME: &'static str = "sgemm_shared_mem_block";
    const SHARED_MEM: &'static [u64] = &[32 * 32, 32 * 32];
    fn grid_size(m: usize, n: usize) -> metal::MTLSize {
        metal::MTLSize::new(m.div_ceil(32) as u64, n.div_ceil(32) as u64, 1)
    }
    fn threadgroup_size(_: usize, _: usize) -> metal::MTLSize {
        metal::MTLSize::new(32 * 32, 1, 1)
    }
}

// This is only correct when the block size is divisible by 64.
pub struct Tiling1D;
impl Tiling1D {
    const BM: u64 = 64;
    const BN: u64 = 64;
    const BK: u64 = 8;
    const TM: u64 = 8;
}

impl GemmKernel for Tiling1D {
    const NAME: &'static str = "sgemm_1d_bt_64_64_8_8";
    const SHARED_MEM: &'static [u64] = &[Self::BM * Self::BK, Self::BK * Self::BN];
    fn grid_size(m: usize, n: usize) -> metal::MTLSize {
        metal::MTLSize::new((n as u64).div_ceil(Self::BN), (m as u64).div_ceil(Self::BM), 1)
    }
    fn threadgroup_size(_: usize, _: usize) -> metal::MTLSize {
        metal::MTLSize::new((Self::BM * Self::BN) / Self::TM, 1, 1)
    }
}

// This is only correct when the block size is divisible by 64.
pub struct Tiling2D;
impl Tiling2D {
    // Maybe this should use 128 rather than 64?
    // https://github.com/siboehm/SGEMM_CUDA/blob/60cba6f9b20a198116c76f18de8047f44df8c8b8/src/runner.cu#L198
    const BM: u64 = 64;
    const BN: u64 = 64;
    const BK: u64 = 8;
    const TM: u64 = 4;
    const TN: u64 = 4;
}

impl GemmKernel for Tiling2D {
    const NAME: &'static str = "sgemm_2d_bt_64_64_8_4_4";

    const SHARED_MEM: &'static [u64] = &[Self::BM * Self::BK, Self::BK * Self::BN];
    fn grid_size(m: usize, n: usize) -> metal::MTLSize {
        metal::MTLSize::new((n as u64).div_ceil(Self::BN), (m as u64).div_ceil(Self::BM), 1)
    }
    fn threadgroup_size(_: usize, _: usize) -> metal::MTLSize {
        metal::MTLSize::new((Self::BM * Self::BN) / (Self::TN * Self::TM), 1, 1)
    }
}

pub struct NaiveSimd;
impl GemmKernel for NaiveSimd {
    const NAME: &'static str = "sgemm_naive_simd";
    const SHARED_MEM: &'static [u64] = &[];
    fn grid_size(m: usize, n: usize) -> metal::MTLSize {
        metal::MTLSize::new((m as u64).div_ceil(8), (n as u64).div_ceil(8), 1)
    }
    fn threadgroup_size(_: usize, _: usize) -> metal::MTLSize {
        metal::MTLSize::new(32, 1, 1)
    }
}

pub struct TiledSimd1;
impl GemmKernel for TiledSimd1 {
    const NAME: &'static str = "sgemm_tiled_simd1";
    const SHARED_MEM: &'static [u64] = &[];
    fn grid_size(m: usize, n: usize) -> metal::MTLSize {
        metal::MTLSize::new((m as u64).div_ceil(8), (n as u64).div_ceil(16), 1)
    }
    fn threadgroup_size(_: usize, _: usize) -> metal::MTLSize {
        metal::MTLSize::new(32, 2, 1)
    }
}

pub struct TiledSimd2;
impl GemmKernel for TiledSimd2 {
    const NAME: &'static str = "sgemm_tiled_simd2";
    const SHARED_MEM: &'static [u64] = &[];
    fn grid_size(m: usize, n: usize) -> metal::MTLSize {
        metal::MTLSize::new((m as u64).div_ceil(16), (n as u64).div_ceil(32), 1)
    }
    fn threadgroup_size(_: usize, _: usize) -> metal::MTLSize {
        metal::MTLSize::new(32, 2, 1)
    }
}

pub struct TiledSimd4;
impl GemmKernel for TiledSimd4 {
    const NAME: &'static str = "sgemm_tiled_simd4";
    const SHARED_MEM: &'static [u64] = &[];
    fn grid_size(m: usize, n: usize) -> metal::MTLSize {
        metal::MTLSize::new((m as u64).div_ceil(32), (n as u64).div_ceil(64), 1)
    }
    fn threadgroup_size(_: usize, _: usize) -> metal::MTLSize {
        metal::MTLSize::new(32, 2, 1)
    }
}

pub struct TiledSimd8;
impl GemmKernel for TiledSimd8 {
    const NAME: &'static str = "sgemm_tiled_simd8";
    const SHARED_MEM: &'static [u64] = &[];
    fn grid_size(m: usize, n: usize) -> metal::MTLSize {
        metal::MTLSize::new((m as u64).div_ceil(64), (n as u64).div_ceil(64), 1)
    }
    fn threadgroup_size(_: usize, _: usize) -> metal::MTLSize {
        metal::MTLSize::new(32, 1, 1)
    }
}

pub fn mm_sync<K: GemmKernel>(
    a: &Matrix<f32>,
    b: &Matrix<f32>,
    c: &Matrix<f32>,
    pl: &metal::ComputePipelineState,
    cq: &metal::CommandQueue,
) -> Result<()> {
    let (m, n) = (c.rows, c.cols);
    let (m_a, k) = (a.rows, a.cols);
    let (k_c, n_c) = (c.rows, c.cols);
    if m_a != m || k != k_c || n != n_c {
        anyhow::bail!("size mismatch in matmul ({m}, {n}) ({m_a}, {k}) ({k},{n_c})")
    }
    let cb = cq.new_command_buffer();
    let (a, b, c) = (a.buffer(), b.buffer(), c.buffer());
    let encoder = cb.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(pl);
    set_params!(encoder, (a, b, c, m, n, k, 1., 0.));
    encoder.use_resource(a, metal::MTLResourceUsage::Read);
    encoder.use_resource(b, metal::MTLResourceUsage::Read);
    encoder.use_resource(c, metal::MTLResourceUsage::Write);
    for (sm_idx, sm) in K::SHARED_MEM.iter().enumerate() {
        encoder.set_threadgroup_memory_length(
            sm_idx as u64,
            (*sm as usize * std::mem::size_of::<f32>()) as u64,
        );
    }
    let grid_size = K::grid_size(m, n);
    let threadgroup_size = K::threadgroup_size(m, n);
    encoder.dispatch_thread_groups(grid_size, threadgroup_size);
    // Somehow, using dispatch_threads with non-even group size doesn't work properly here.
    encoder.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    Ok(())
}

pub fn pipeline<K: GemmKernel>() -> Result<metal::ComputePipelineState> {
    let device = match metal::Device::system_default() {
        Some(device) => device,
        None => anyhow::bail!("no default device found"),
    };
    let lib = device.new_library_with_source(DOT, &metal::CompileOptions::new()).map_err(E::msg)?;
    let function = lib.get_function(K::NAME, None).map_err(E::msg)?;
    let pipeline = device.new_compute_pipeline_state_with_function(&function).map_err(E::msg)?;
    Ok(pipeline)
}

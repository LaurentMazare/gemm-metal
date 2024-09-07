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
    println!("{c:?}");
    println!("{cc:?}");
    let max_diff = c
        .iter()
        .zip(cc.iter())
        .map(|(v1, v2)| (v1 - v2).abs())
        .max_by(|v1, v2| f32::total_cmp(v1, v2))
        .unwrap();
    println!("N-DIFF {max_diff}");
}

pub fn gemm_naive(m: usize, n: usize, k: usize, repeats: usize) -> anyhow::Result<()> {
    gemm_naive_(m, n, k, repeats, false)
}
pub fn gemm_naive_check(m: usize, n: usize, k: usize, repeats: usize) -> anyhow::Result<()> {
    gemm_naive_(m, n, k, repeats, true)
}

fn gemm_naive_(m: usize, n: usize, k: usize, repeats: usize, check: bool) -> anyhow::Result<()> {
    use metal::Device;

    let device = Device::system_default().expect("No device found");
    let lib = device.new_library_with_source(DOT, &metal::CompileOptions::new()).map_err(E::msg)?;
    let function = lib.get_function("sgemm_naive", None).map_err(E::msg)?;
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
        let grid_size = metal::MTLSize::new(m.div_ceil(32) as u64, n.div_ceil(32) as u64, 1);
        let threadgroup_size = metal::MTLSize::new(32, 32, 1);
        // Somehow, using dispatch_threads with non-even group size doesn't work properly here.
        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
    let dt = start_time.elapsed().as_secs_f64() / repeats as f64;
    let gflops = (n * m * k) as f64 * 2. / (1e9 * dt);
    println!("N {m} {n} {k} {gflops:.2}");

    if check {
        mm_check(&a, &b, &c, m, n, k)
    }
    Ok(())
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
    use metal::Device;

    let device = Device::system_default().expect("No device found");
    let lib = device.new_library_with_source(DOT, &metal::CompileOptions::new()).map_err(E::msg)?;
    let function = lib.get_function("sgemm_coalescing", None).map_err(E::msg)?;
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
        let grid_size = metal::MTLSize::new(m.div_ceil(32) as u64, n.div_ceil(32) as u64, 1);
        let threadgroup_size = metal::MTLSize::new(32 * 32, 1, 1);
        // Somehow, using dispatch_threads with non-even group size doesn't work properly here.
        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
    let dt = start_time.elapsed().as_secs_f64() / repeats as f64;
    let gflops = (n * m * k) as f64 * 2. / (1e9 * dt);
    println!("C {m} {n} {k} {gflops:.2}");

    if check {
        mm_check(&a, &b, &c, m, n, k)
    }
    Ok(())
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

fn gemm_shared_mem_block_(
    m: usize,
    n: usize,
    k: usize,
    repeats: usize,
    check: bool,
) -> anyhow::Result<()> {
    use metal::Device;

    let device = Device::system_default().expect("No device found");
    let lib = device.new_library_with_source(DOT, &metal::CompileOptions::new()).map_err(E::msg)?;
    let function = lib.get_function("sgemm_shared_mem_block", None).map_err(E::msg)?;
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
        encoder.set_threadgroup_memory_length(0, 32 * 32 * std::mem::size_of::<f32>() as u64); // As
        encoder.set_threadgroup_memory_length(1, 32 * 32 * std::mem::size_of::<f32>() as u64); // Bs
        let grid_size = metal::MTLSize::new(m.div_ceil(32) as u64, n.div_ceil(32) as u64, 1);
        let threadgroup_size = metal::MTLSize::new(32 * 32, 1, 1);
        // Somehow, using dispatch_threads with non-even group size doesn't work properly here.
        encoder.dispatch_thread_groups(grid_size, threadgroup_size);
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
    let dt = start_time.elapsed().as_secs_f64() / repeats as f64;
    let gflops = (n * m * k) as f64 * 2. / (1e9 * dt);
    println!("S {m} {n} {k} {gflops:.2}");

    if check {
        mm_check(&a, &b, &c, m, n, k)
    }
    Ok(())
}

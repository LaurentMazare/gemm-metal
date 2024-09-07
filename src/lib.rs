use anyhow::Result;

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

pub fn gemm_naive(m: usize, n: usize, k: usize, repeats: usize) -> anyhow::Result<()> {
    use metal::Device;

    let device = Device::system_default().expect("No device found");
    let lib = device.new_library_with_source(DOT, &metal::CompileOptions::new()).unwrap();
    let function = lib.get_function("sgemm_naive", None).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(&function).unwrap();
    let a: Matrix<f32> = Matrix::zeros(&device, m, k);
    let b: Matrix<f32> = Matrix::zeros(&device, k, n);
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
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
    let dt = start_time.elapsed().as_secs_f64() / repeats as f64;
    let gflops = (n * m * k) as f64 * 2. / (1e9 * dt);
    println!("{m} {n} {k} {gflops:.2}");
    Ok(())
}

pub fn gemm_coalescing(m: usize, n: usize, k: usize, repeats: usize) -> anyhow::Result<()> {
    use metal::Device;

    let device = Device::system_default().expect("No device found");
    let lib = device.new_library_with_source(DOT, &metal::CompileOptions::new()).unwrap();
    let function = lib.get_function("sgemm_coalescing", None).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(&function).unwrap();
    let a: Matrix<f32> = Matrix::zeros(&device, m, k);
    let b: Matrix<f32> = Matrix::zeros(&device, k, n);
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
        encoder.dispatch_threads(grid_size, threadgroup_size);
        encoder.end_encoding();
        cb.commit();
        cb.wait_until_completed();
    }
    let dt = start_time.elapsed().as_secs_f64() / repeats as f64;
    let gflops = (n * m * k) as f64 * 2. / (1e9 * dt);
    println!("{m} {n} {k} {gflops:.2}");
    Ok(())
}

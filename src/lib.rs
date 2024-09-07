const DOT: &str = include_str!("dot_product.metal");

mod utils;

fn read_to_vec<T: Clone>(buffer: &metal::Buffer, n: usize) -> Vec<T> {
    let ptr = buffer.contents() as *const T;
    assert!(!ptr.is_null());
    let slice = unsafe { std::slice::from_raw_parts(ptr, n) };
    slice.to_vec()
}

fn new_buffer<T>(device: &metal::Device, data: &[T]) -> metal::Buffer {
    let options = metal::MTLResourceOptions::StorageModeManaged;
    let ptr = data.as_ptr() as *const std::ffi::c_void;
    let size = std::mem::size_of_val(data) as u64;
    device.new_buffer_with_data(ptr, size, options)
}

pub fn dot_product() -> anyhow::Result<()> {
    use metal::Device;

    println!("hello world");
    let device = Device::system_default().expect("No device found");
    let lib = device.new_library_with_source(DOT, &metal::CompileOptions::new()).unwrap();
    let function = lib.get_function("dot_product", None).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(&function).unwrap();
    let a = new_buffer(&device, &[42f32, 2.0, 3., 3., 7., 9.9999]);
    let b = new_buffer(&device, &[1f32, -3.14, 3., 7., 33., 9.9999]);

    let length = a.length() / std::mem::size_of::<u32>() as u64;
    let size = length * core::mem::size_of::<u32>() as u64;

    let c = device.new_buffer(size, metal::MTLResourceOptions::StorageModeManaged);
    let command_queue = device.new_command_queue();
    let cb = command_queue.new_command_buffer();

    let encoder = cb.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(encoder, (&a, &b, &c));
    encoder.use_resource(&a, metal::MTLResourceUsage::Read);
    encoder.use_resource(&b, metal::MTLResourceUsage::Read);
    encoder.use_resource(&c, metal::MTLResourceUsage::Write);
    let grid_size = metal::MTLSize::new(length, 1, 1);
    let threadgroup_size = metal::MTLSize::new(1, 1, 1);
    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();
    cb.commit();
    cb.wait_until_completed();
    println!("Done!");
    let v: Vec<f32> = read_to_vec(&c, length as usize);
    println!("{v:?}");

    Ok(())
}

pub fn gemm_naive(m: usize, n: usize, k: usize, repeats: usize) -> anyhow::Result<()> {
    use metal::Device;

    let device = Device::system_default().expect("No device found");
    let lib = device.new_library_with_source(DOT, &metal::CompileOptions::new()).unwrap();
    let function = lib.get_function("sgemm_naive", None).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(&function).unwrap();
    let a = new_buffer(&device, &vec![0f32; m * k]);
    let b = new_buffer(&device, &vec![0f32; k * n]);

    let c = device.new_buffer(
        (m * n * std::mem::size_of::<f32>()) as u64,
        metal::MTLResourceOptions::StorageModeManaged,
    );
    let command_queue = device.new_command_queue();
    let start_time = std::time::Instant::now();
    for _ in 0..repeats {
        let cb = command_queue.new_command_buffer();
        let encoder = cb.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        set_params!(encoder, (&a, &b, &c, m, n, k, 1., 0.));
        encoder.use_resource(&a, metal::MTLResourceUsage::Read);
        encoder.use_resource(&b, metal::MTLResourceUsage::Read);
        encoder.use_resource(&c, metal::MTLResourceUsage::Write);
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

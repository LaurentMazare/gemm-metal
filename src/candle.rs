#![allow(unused)]
// The kernels below come from the candle metal gemm implementation.
use anyhow::{Error as E, Result};
use metal::{
    Buffer, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, Function,
    FunctionConstantValues, Library, MTLDataType, MTLSize, NSUInteger,
};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::RwLock;

// Current source: https://github.com/ivarflakstad/metal-flash-attention/tree/candle
const MFA: &[u8] = include_bytes!("libMetalFlashAttention.metallib");
const MLX_GEMM: &str = include_str!("mlx_gemm.metal");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    Gemm,
    Mfa,
}

pub trait EncoderProvider {
    type Encoder<'a>: AsRef<metal::ComputeCommandEncoderRef>
    where
        Self: 'a;
    fn encoder(&self) -> Self::Encoder<'_>;
}

pub struct WrappedEncoder<'a>(&'a ComputeCommandEncoderRef);

impl<'a> Drop for WrappedEncoder<'a> {
    fn drop(&mut self) {
        self.0.end_encoding()
    }
}

impl<'a> AsRef<metal::ComputeCommandEncoderRef> for WrappedEncoder<'a> {
    fn as_ref(&self) -> &metal::ComputeCommandEncoderRef {
        self.0
    }
}

impl EncoderProvider for &metal::CommandBuffer {
    type Encoder<'a> = WrappedEncoder<'a>
    where
        Self: 'a;
    fn encoder(&self) -> Self::Encoder<'_> {
        WrappedEncoder(self.new_compute_command_encoder())
    }
}

impl EncoderProvider for &metal::CommandBufferRef {
    type Encoder<'a> = WrappedEncoder<'a>
    where
        Self: 'a;
    fn encoder(&self) -> Self::Encoder<'_> {
        WrappedEncoder(self.new_compute_command_encoder())
    }
}
type Libraries = HashMap<Source, Library>;
type Pipelines = HashMap<(&'static str, Option<ConstantValues>), ComputePipelineState>;
#[derive(Debug)]
pub struct Kernels {
    libraries: RwLock<Libraries>,
    pipelines: RwLock<Pipelines>,
}

impl Default for Kernels {
    fn default() -> Self {
        Self::new()
    }
}

impl Kernels {
    pub fn new() -> Self {
        let libraries = RwLock::new(Libraries::new());
        let pipelines = RwLock::new(Pipelines::new());
        Self { libraries, pipelines }
    }

    fn get_library_source(&self, source: Source) -> &'static str {
        match source {
            Source::Gemm => MLX_GEMM,
            Source::Mfa => panic!("Invalid lib"),
        }
    }

    /// Load the give library from its [`source`].
    /// If this has been previously loaded it will just fetch it from cache.
    pub fn load_library(&self, device: &Device, source: Source) -> Result<Library> {
        let mut libraries = self.libraries.write().map_err(|_| E::msg("unable to lock"))?;
        if let Some(lib) = libraries.get(&source) {
            Ok(lib.clone())
        } else {
            let lib = match source {
                Source::Mfa => {
                    let source_data = MFA;
                    device.new_library_with_data(source_data).map_err(E::msg)?
                }
                source => {
                    let source_content = self.get_library_source(source);
                    device
                        .new_library_with_source(source_content, &CompileOptions::new())
                        .map_err(E::msg)?
                }
            };
            libraries.insert(source, lib.clone());
            Ok(lib)
        }
    }

    fn load_function(
        &self,
        device: &Device,
        source: Source,
        name: &'static str,
        constants: Option<FunctionConstantValues>,
    ) -> Result<Function> {
        let func =
            self.load_library(device, source)?.get_function(name, constants).map_err(E::msg)?;
        Ok(func)
    }

    /// Load the give pipeline
    /// loads the library from source, then gets the function [`name`] from
    /// that source
    fn load_pipeline_with_constants(
        &self,
        device: &Device,
        source: Source,
        name: &'static str,
        constants: Option<ConstantValues>,
    ) -> Result<ComputePipelineState> {
        let mut pipelines = self.pipelines.write().map_err(|_| E::msg("unable to lock"))?;
        let key = (name, constants);
        if let Some(pipeline) = pipelines.get(&key) {
            Ok(pipeline.clone())
        } else {
            let (name, constants) = key;
            let func = self.load_function(
                device,
                source,
                name,
                constants.as_ref().map(|c| c.function_constant_values()),
            )?;
            let pipeline =
                device.new_compute_pipeline_state_with_function(&func).map_err(E::msg)?;
            pipelines.insert((name, constants), pipeline.clone());

            Ok(pipeline)
        }
    }

    /// Load the give pipeline
    /// loads the library from source, then gets the function [`name`] from
    /// that source (without constants)
    pub fn load_pipeline(
        &self,
        device: &Device,
        source: Source,
        name: &'static str,
    ) -> Result<ComputePipelineState> {
        self.load_pipeline_with_constants(device, source, name, None)
    }
}

fn divide(m: usize, b: usize) -> NSUInteger {
    ((m + b - 1) / b) as NSUInteger
}

#[derive(Debug, PartialEq)]
pub enum Value {
    USize(usize),
    Bool(bool),
    F32(f32),
    U16(u16),
}

impl std::hash::Hash for Value {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            Value::F32(v) => v.to_bits().hash(state),
            Value::USize(v) => v.hash(state),
            Value::U16(v) => v.hash(state),
            Value::Bool(v) => v.hash(state),
        }
    }
}

impl Value {
    fn data_type(&self) -> MTLDataType {
        match self {
            Value::USize(_) => MTLDataType::UInt,
            Value::F32(_) => MTLDataType::Float,
            Value::U16(_) => MTLDataType::UShort,
            Value::Bool(_) => MTLDataType::Bool,
        }
    }
}

/// Not true, good enough for our purposes.
impl Eq for Value {}

#[derive(Debug, Eq, PartialEq, Hash)]
struct ConstantValues(Vec<(usize, Value)>);

impl ConstantValues {
    pub fn new(values: Vec<(usize, Value)>) -> Self {
        Self(values)
    }

    fn function_constant_values(&self) -> FunctionConstantValues {
        let f = FunctionConstantValues::new();
        for (index, value) in &self.0 {
            let ty = value.data_type();
            match value {
                Value::USize(v) => {
                    f.set_constant_value_at_index(
                        v as *const usize as *const c_void,
                        ty,
                        *index as u64,
                    );
                }
                Value::F32(v) => {
                    f.set_constant_value_at_index(
                        v as *const f32 as *const c_void,
                        ty,
                        *index as u64,
                    );
                }
                Value::U16(v) => {
                    f.set_constant_value_at_index(
                        v as *const u16 as *const c_void,
                        ty,
                        *index as u64,
                    );
                }
                Value::Bool(v) => {
                    f.set_constant_value_at_index(
                        v as *const bool as *const c_void,
                        ty,
                        *index as u64,
                    );
                }
            }
        }
        f
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum GemmDType {
    BF16,
    F16,
    F32,
}

#[allow(clippy::too_many_arguments)]
pub fn call_mlx_gemm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GemmDType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_stride: &[usize],
    lhs_offset: usize,
    lhs_buffer: &Buffer,
    rhs_stride: &[usize],
    rhs_offset: usize,
    rhs_buffer: &Buffer,
    output: &Buffer,
) -> Result<()> {
    #[derive(Debug)]
    #[repr(C)]
    struct GemmParams {
        m: i32,
        n: i32,
        k: i32,
        lda: i32,
        ldb: i32,
        ldd: i32,
        tiles_n: i32,
        tiles_m: i32,
        batch_stride_a: isize,
        batch_stride_b: isize,
        batch_stride_d: isize,
        swizzle_log: i32,
        gemm_k_iterations_aligned: i32,
        batch_ndim: i32,
    }
    assert!(rhs_stride.len() >= 2);
    assert!(lhs_stride.len() >= 2);
    let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
    let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
    let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
    let lhs_m2 = lhs_stride[lhs_stride.len() - 2];
    // lhs has shape b, m, k
    // We also allow for the case where the stride on the minor dimension is not as expected but
    // there is a single element.
    let (lda, a_trans) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        (k as i32, false)
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        (m as i32, true)
    } else {
        anyhow::bail!(
            "matmul not contiguous {:?} {:?} {:?}",
            lhs_stride.to_vec(),
            rhs_stride.to_vec(),
            (m, n, k),
        )
    };
    // rhs has shape b, k, n
    let (ldb, b_trans) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
        (n as i32, false)
    } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
        (k as i32, true)
    } else {
        anyhow::bail!(
            "matmul not contiguous {:?} {:?} {:?}",
            lhs_stride.to_vec(),
            rhs_stride.to_vec(),
            (m, n, k),
        )
    };
    let (bm, bn, bk, wn, wm) = (32, 32, 16, 2, 2);
    // https://github.com/ml-explore/mlx/blob/02efb310cac667bc547d1b96f21596c221f84fe7/mlx/backend/metal/matmul.cpp#L422
    let constants = Some(ConstantValues::new(vec![
        (10, Value::Bool(/* has_batch */ b > 1)),
        (100, Value::Bool(/* use_out_source */ false)),
        (110, Value::Bool(/* do_axpby */ false)),
        (200, Value::Bool(/* align_m */ m % bm == 0)),
        (201, Value::Bool(/* align_n */ n % bn == 0)),
        (202, Value::Bool(/* align_k */ k % bk == 0)),
        (300, Value::Bool(/* do_gather */ false)),
    ]));

    let swizzle_log = 0;
    let tile = 1 << swizzle_log;
    let tn = n.div_ceil(bn);
    let tm = m.div_ceil(bm);
    let tn = tn * tile;
    let tm = tm.div_ceil(tile);

    let batch_stride_a =
        if lhs_stride.len() > 2 { lhs_stride[lhs_stride.len() - 3] } else { m * k };
    let batch_stride_b =
        if rhs_stride.len() > 2 { rhs_stride[rhs_stride.len() - 3] } else { n * k };

    let gemm_params = GemmParams {
        m: m as i32,
        n: n as i32,
        k: k as i32,
        lda,
        ldb,
        ldd: n as i32,
        tiles_n: tn as i32,
        tiles_m: tm as i32,
        swizzle_log,
        batch_stride_a: batch_stride_a as isize,
        batch_stride_b: batch_stride_b as isize,
        batch_stride_d: (m * n) as isize,
        batch_ndim: 1i32,
        gemm_k_iterations_aligned: (k / bk) as i32,
    };
    let batch_strides = [gemm_params.batch_stride_a, gemm_params.batch_stride_b];

    // TODO(laurent): generate the name
    // template [[host_name("gemm_" #tname "_"  #iname "_" #oname "_bm" #bm "_bn" #bn "_bk" #bk "_wm" #wm "_wn" #wn)]]
    let name = match (dtype, a_trans, b_trans) {
        (GemmDType::F32, false, false) => "gemm_nn_f32_f32_32_32_16_2_2",
        (GemmDType::F32, true, false) => "gemm_tn_f32_f32_32_32_16_2_2",
        (GemmDType::F32, false, true) => "gemm_nt_f32_f32_32_32_16_2_2",
        (GemmDType::F32, true, true) => "gemm_tt_f32_f32_32_32_16_2_2",
        (GemmDType::BF16, false, false) => "gemm_nn_bf16_bf16_32_32_16_2_2",
        (GemmDType::BF16, true, false) => "gemm_tn_bf16_bf16_32_32_16_2_2",
        (GemmDType::BF16, false, true) => "gemm_nt_bf16_bf16_32_32_16_2_2",
        (GemmDType::BF16, true, true) => "gemm_tt_bf16_bf16_32_32_16_2_2",
        (GemmDType::F16, false, false) => "gemm_nn_f16_f16_32_32_16_2_2",
        (GemmDType::F16, true, false) => "gemm_tn_f16_f16_32_32_16_2_2",
        (GemmDType::F16, false, true) => "gemm_nt_f16_f16_32_32_16_2_2",
        (GemmDType::F16, true, true) => "gemm_tt_f16_f16_32_32_16_2_2",
    };
    let pipeline = kernels.load_pipeline_with_constants(device, Source::Gemm, name, constants)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(lhs_buffer), lhs_offset as NSUInteger);
    encoder.set_buffer(1, Some(rhs_buffer), rhs_offset as NSUInteger);
    encoder.set_buffer(3, Some(output), 0);
    encoder.set_bytes(
        4,
        std::mem::size_of::<GemmParams>() as u64,
        &gemm_params as *const GemmParams as *const c_void,
    );
    encoder.set_bytes(
        6, // batch_shape
        std::mem::size_of::<i32>() as u64,
        &(b as i32) as *const i32 as *const c_void,
    );
    encoder.set_bytes(
        7,
        (std::mem::size_of::<isize>() * batch_strides.len()) as u64,
        batch_strides.as_ptr() as *const c_void,
    );

    let grid_size = MTLSize {
        width: tn as u64,
        height: tm as u64,
        depth: /* batch_size_out */ b as u64,
    };
    let group_size = MTLSize { width: 32, height: wn, depth: wm };
    encoder.use_resource(lhs_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(rhs_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(grid_size, group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_mfa_gemm(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_stride: &[usize],
    lhs_offset: usize,
    lhs_buffer: &Buffer,
    rhs_stride: &[usize],
    rhs_offset: usize,
    rhs_buffer: &Buffer,
    output: &Buffer,
) -> Result<()> {
    assert!(rhs_stride.len() >= 2);
    assert!(lhs_stride.len() >= 2);
    let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
    let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
    let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
    let lhs_m2 = lhs_stride[lhs_stride.len() - 2];
    // lhs has shape b, m, k
    // We also allow for the case where the stride on the minor dimension is not as expected but
    // there is a single element.
    let a_trans = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        false
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        true
    } else {
        anyhow::bail!(
            "matmul not contiguous {:?} {:?} {:?}",
            lhs_stride.to_vec(),
            rhs_stride.to_vec(),
            (m, n, k),
        )
    };
    // rhs has shape b, k, n
    let b_trans = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
        false
    } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
        true
    } else {
        anyhow::bail!(
            "matmul not contiguous {:?} {:?} {:?}",
            lhs_stride.to_vec(),
            rhs_stride.to_vec(),
            (m, n, k),
        )
    };
    let d_trans = false;
    let alpha = 1.0f32;
    let beta = 0.0f32;
    let batched = b > 1;
    let fused_activation = false;
    let fused_bias = false;
    let (m_simd, n_simd, k_simd, m_splits, n_splits) = if m == 1 {
        let m_simd = 8;
        let n_simd = 8;
        let k_simd = 64;
        let m_splits = 1;
        let n_splits = 1;
        (m_simd, n_simd, k_simd, m_splits, n_splits)
    } else {
        let m_simd = 40;
        let n_simd = 40;
        let k_simd = 32;
        let m_splits = 1;
        let n_splits = 1;
        (m_simd, n_simd, k_simd, m_splits, n_splits)
    };
    let constants = Some(ConstantValues::new(vec![
        (0, Value::USize(m)),
        (1, Value::USize(n)),
        (2, Value::USize(k)),
        (10, Value::Bool(a_trans)),
        (11, Value::Bool(b_trans)),
        (13, Value::Bool(d_trans)),
        (20, Value::F32(alpha)),
        (21, Value::F32(beta)),
        (100, Value::Bool(batched)),
        (101, Value::Bool(fused_activation)),
        // Garbage
        (102, Value::Bool(false)),
        (103, Value::Bool(false)),
        (113, Value::Bool(false)),
        (50_000, Value::Bool(false)),
        // End garbage
        (200, Value::U16(m_simd)),
        (201, Value::U16(n_simd)),
        (202, Value::U16(k_simd)),
        (210, Value::U16(m_splits)),
        (211, Value::U16(n_splits)),
        (50_001, Value::Bool(fused_bias)),
    ]));
    let pipeline = kernels.load_pipeline_with_constants(device, Source::Mfa, name, constants)?;
    let m_group = m_simd * m_splits;
    let n_group = n_simd * n_splits;

    let a_block_length = m_group * k_simd;
    let b_block_length = k_simd * n_group;

    let mut block_elements = a_block_length + b_block_length;
    if (m % 8 != 0) && (n % 8 != 0) {
        let c_block_length = m_group * n_group;
        block_elements = std::cmp::max(c_block_length, block_elements)
    }
    if fused_bias {
        if d_trans {
            block_elements = std::cmp::max(block_elements, m_group);
        } else {
            block_elements = std::cmp::max(block_elements, n_group);
        }
    }
    let bytes = match name {
        "sgemm" => 4,
        "hgemm" => 2,
        "bgemm" => 2,
        other => {
            anyhow::bail!("{other} is not a valid kernel for gemm");
        }
    };
    let block_bytes = block_elements * bytes;

    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoderRef = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_threadgroup_memory_length(0, block_bytes.into());
    encoder.set_buffer(0, Some(lhs_buffer), lhs_offset as NSUInteger);
    encoder.set_buffer(1, Some(rhs_buffer), rhs_offset as NSUInteger);
    encoder.set_buffer(2, Some(output), 0);
    // TODO Tensor D

    let grid_z = b;
    if batched {
        let byte_stride_a: usize = lhs_stride[lhs_stride.len() - 3] * bytes as usize;
        let byte_stride_b: usize = rhs_stride[rhs_stride.len() - 3] * bytes as usize;
        let byte_stride_c = m * n * bytes as usize;
        // TODO byte_stride_d
        let byte_stride_d = 0;

        let buffer: Vec<u64> =
            vec![byte_stride_a as _, byte_stride_b as _, byte_stride_c as _, byte_stride_d as _];
        encoder.set_bytes(
            10,
            (buffer.len() * core::mem::size_of::<u64>()) as NSUInteger,
            buffer.as_ptr() as *const NSUInteger as *const c_void,
        );
    }

    let grid_size = MTLSize {
        width: divide(n, n_group.into()),
        height: divide(m, m_group.into()),
        depth: grid_z as NSUInteger,
    };
    let group_size =
        MTLSize { width: 32 * (m_splits as u64) * (n_splits as u64), height: 1, depth: 1 };
    encoder.use_resource(lhs_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(rhs_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(grid_size, group_size);
    Ok(())
}

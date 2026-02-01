//! Host-side state management for the streaming TTT kernel.
//!
//! This module provides:
//! - `TttStreamingState` - manages a running persistent kernel
//! - Global registry for looking up streaming states by key
//! - `StreamingConfig` - configuration including stream key

use std::{
    collections::HashMap,
    sync::{LazyLock, Mutex},
};

use burn_backend::Shape;
use burn_cubecl::{CubeRuntime, FloatElement, ops::numeric::empty_device, tensor::CubeTensor};
use cubecl::prelude::*;
use thundercube::{
    prelude::{D4, D8, D16, D32, D64, LINE_SIZE},
    streaming::{AsyncStream, GpuPtr},
    util::wait_for_sync,
};
use tracing::trace;

use super::{
    forward::{ForwardIntermediatesLaunch, InputsLaunch, OutputsLaunch},
    helpers::Params,
    streaming::{
        CTRL_ARRAY_SIZE, CTRL_DONE, CTRL_READY,
        StreamingKernelConfig, fused_ttt_streaming_kernel,
    },
};
use crate::ttt::cubecl_kernels::FusedTttConfig;

/// Dispatch macro for streaming kernel.
/// Matches on (mini_batch_len, head_dim, threads) to select tile config.
macro_rules! impl_streaming_dispatch {
    (
        $client:expr, $cube_count:expr, $cube_dim_client:expr,
        $inputs:expr, $outputs:expr, $control_arg:expr, $fwd_intermediates:expr,
        $kernel_config:expr,
        $mini_batch_len:expr, $head_dim:expr, $threads:expr;
        $(($s:literal, $h:literal, $t:literal, $CS:ty, $F:ty, $CSR:ty, $FR:ty)),* $(,)?
    ) => {
        match ($mini_batch_len, $head_dim, $threads) {
            $(
                ($s, $h, $t) => {
                    type P<E> = Params<E, $CS, $F, $CSR, $FR>;
                    let cube_dim = CubeDim::new($cube_dim_client, $t);
                    fused_ttt_streaming_kernel::launch::<P<_>, _>(
                        $client, $cube_count, cube_dim,
                        $inputs, $outputs, $control_arg, $fwd_intermediates, $kernel_config,
                    ).unwrap()
                }
            )*
            _ => {
                let supported = [$((stringify!($s), stringify!($h), stringify!($t))),*];
                let supported_str: Vec<_> = supported.iter()
                    .map(|(s, h, t)| format!("{}×{}@{}", s, h, t))
                    .collect();
                panic!(
                    "Unsupported streaming config: mini_batch_len={}, head_dim={}, threads={}. Supported: {}",
                    $mini_batch_len, $head_dim, $threads, supported_str.join(", ")
                )
            }
        }
    };
}

macro_rules! dispatch_streaming_kernel {
    ($client:expr, $cube_count:expr, $cube_dim_client:expr,
     $inputs:expr, $outputs:expr, $control_arg:expr, $fwd_intermediates:expr,
     $kernel_config:expr,
     $mini_batch_len:expr, $head_dim:expr, $threads:expr) => {
        supported_tile_configs!(impl_streaming_dispatch!(
            $client, $cube_count, $cube_dim_client,
            $inputs, $outputs, $control_arg, $fwd_intermediates,
            $kernel_config,
            $mini_batch_len, $head_dim, $threads;
        ))
    };
}

/// Key for the streaming state registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StreamKey {
    pub stream_id: u64,
}

impl StreamKey {
    pub fn new(stream_id: u64) -> Self {
        Self { stream_id }
    }
}

/// Configuration for the streaming kernel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StreamingConfig {
    /// Unique identifier for this streaming session
    pub stream_id: u64,
    /// Batch size
    pub batch_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Mini-batch sequence length
    pub mini_batch_len: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Layer norm epsilon (scaled)
    pub epsilon_scaled: u32,
    /// Number of threads per cube
    pub threads: usize,
    /// Enable debug output in kernel
    pub debug: bool,
}

impl StreamingConfig {
    pub fn new(
        stream_id: u64,
        batch_size: usize,
        num_heads: usize,
        mini_batch_len: usize,
        head_dim: usize,
        epsilon: f32,
        threads: usize,
    ) -> Self {
        Self {
            stream_id,
            batch_size,
            num_heads,
            mini_batch_len,
            head_dim,
            epsilon_scaled: (epsilon / 1e-9) as u32,
            threads,
            debug: false,
        }
    }

    /// Set debug mode
    pub fn with_debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }

    pub fn epsilon(&self) -> f32 {
        self.epsilon_scaled as f32 * 1e-9
    }

    pub fn kernel_config(&self) -> StreamingKernelConfig {
        StreamingKernelConfig::new(
            FusedTttConfig::new(
                self.mini_batch_len,
                self.head_dim,
                self.epsilon(),
                self.threads,
            ),
            self.debug,
        )
    }

    pub fn key(&self) -> StreamKey {
        StreamKey::new(self.stream_id)
    }

    /// Total number of cubes (batch * heads)
    pub fn num_cubes(&self) -> usize {
        self.batch_size * self.num_heads
    }

    /// Size of control array
    pub fn ctrl_array_len(&self) -> usize {
        self.num_cubes() * CTRL_ARRAY_SIZE as usize
    }

    /// Shape for QKV buffers: [batch, heads, mini_batch_len, head_dim]
    pub fn qkv_shape(&self) -> [usize; 4] {
        [
            self.batch_size,
            self.num_heads,
            self.mini_batch_len,
            self.head_dim,
        ]
    }

    /// Number of f32 elements in a QKV buffer
    pub fn qkv_len(&self) -> usize {
        self.batch_size * self.num_heads * self.mini_batch_len * self.head_dim
    }

    /// Shape for ttt_lr_eta: [batch, heads, mini_batch_len]
    pub fn eta_shape(&self) -> [usize; 3] {
        [self.batch_size, self.num_heads, self.mini_batch_len]
    }

    /// Number of f32 elements in eta buffer
    pub fn eta_len(&self) -> usize {
        self.batch_size * self.num_heads * self.mini_batch_len
    }

    /// Shape for weight: [batch, heads, head_dim, head_dim]
    pub fn weight_shape(&self) -> [usize; 4] {
        [self.batch_size, self.num_heads, self.head_dim, self.head_dim]
    }

    /// Shape for bias: [batch, heads, head_dim]
    pub fn bias_shape(&self) -> [usize; 3] {
        [self.batch_size, self.num_heads, self.head_dim]
    }
}

/// GPU buffer tensors for the streaming kernel.
pub struct StreamingBufferTensors<R: CubeRuntime> {
    // Input/output buffers (single mini-batch sized)
    pub xq: CubeTensor<R>,
    pub xk: CubeTensor<R>,
    pub xv: CubeTensor<R>,
    pub ttt_lr_eta: CubeTensor<R>,
    pub output: CubeTensor<R>,
    /// Separate output buffer for returning results without blocking on persistent kernel
    pub result_output: CubeTensor<R>,

    // Control array
    pub control: CubeTensor<R>,

    // Weight and bias (updated in-place by kernel)
    pub weight: CubeTensor<R>,
    pub bias: CubeTensor<R>,
    /// Copy of weight for returning results
    pub result_weight: CubeTensor<R>,
    /// Copy of bias for returning results
    pub result_bias: CubeTensor<R>,

    // Constant tensors
    pub token_eta: CubeTensor<R>,
    pub ln_weight: CubeTensor<R>,
    pub ln_bias: CubeTensor<R>,

    // Forward intermediates (single mini-batch for now)
    pub x_hat_fused: CubeTensor<R>,
    pub std_fused: CubeTensor<R>,
    pub grad_output_fused: CubeTensor<R>,
    pub grad_x_hat_fused: CubeTensor<R>,
    pub grad_l_wrt_Z1: CubeTensor<R>,
    pub x_hat_ln: CubeTensor<R>,
    pub std_ln: CubeTensor<R>,
}

use super::next_persistent_kernel_stream_id;

/// State for a running streaming TTT kernel.
///
/// Note: This struct is not Send because AsyncStream contains a raw HIP stream pointer.
/// It should only be used from the thread that created it.
pub struct TttStreamingState<R: CubeRuntime> {
    pub config: StreamingConfig,
    pub stream: AsyncStream,
    pub tensors: StreamingBufferTensors<R>,
    /// Client for normal operations (D2D copies, reading results, etc.)
    pub client: ComputeClient<R>,
    /// Client with a separate stream for the persistent kernel.
    /// This prevents the persistent kernel from blocking normal operations.
    pub kernel_client: ComputeClient<R>,
    pub is_initialized: bool,
    // Cached GPU pointers - obtained BEFORE kernel launch to avoid
    // triggering cross-stream synchronization when accessing kernel-written buffers
    cached_xq_ptr: GpuPtr<'static, f32>,
    cached_xk_ptr: GpuPtr<'static, f32>,
    cached_xv_ptr: GpuPtr<'static, f32>,
    cached_eta_ptr: GpuPtr<'static, f32>,
    cached_ctrl_ptr: GpuPtr<'static, u32>,
    cached_output_ptr: GpuPtr<'static, f32>,
    cached_weight_ptr: GpuPtr<'static, f32>,
    cached_bias_ptr: GpuPtr<'static, f32>,
    cached_result_output_ptr: GpuPtr<'static, f32>,
    cached_result_weight_ptr: GpuPtr<'static, f32>,
    cached_result_bias_ptr: GpuPtr<'static, f32>,
}

// SAFETY: TttStreamingState is only accessed from the same thread via the registry lock.
// The AsyncStream's raw pointer is only used for HIP API calls which are thread-safe
// when called from the thread that created the stream.
unsafe impl<R: CubeRuntime> Send for TttStreamingState<R> {}

/// Type-erased wrapper for storing in the global registry.
/// Uses Box<dyn Any> pattern for type erasure.
struct AnyStreamingState(Box<dyn std::any::Any + Send>);

impl AnyStreamingState {
    fn new<R: CubeRuntime + 'static>(state: TttStreamingState<R>) -> Self {
        Self(Box::new(state))
    }

    fn downcast_mut<R: CubeRuntime + 'static>(&mut self) -> Option<&mut TttStreamingState<R>> {
        self.0.downcast_mut()
    }
}

/// Global registry of streaming states.
static STREAMING_REGISTRY: LazyLock<Mutex<HashMap<StreamKey, AnyStreamingState>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Get or create a streaming state from the global registry.
pub fn get_or_create_streaming_state<R: CubeRuntime + 'static, F: FloatElement>(
    config: StreamingConfig,
    client: ComputeClient<R>,
    device: R::Device,
    initial_weight: CubeTensor<R>,
    initial_bias: CubeTensor<R>,
    token_eta: CubeTensor<R>,
    ln_weight: CubeTensor<R>,
    ln_bias: CubeTensor<R>,
) -> &'static mut TttStreamingState<R> {
    let key = config.key();

    let mut registry = STREAMING_REGISTRY.lock().unwrap();

    if !registry.contains_key(&key) {
        let state = TttStreamingState::new::<F>(
            config,
            client,
            device,
            initial_weight,
            initial_bias,
            token_eta,
            ln_weight,
            ln_bias,
        );
        registry.insert(key, AnyStreamingState::new(state));
    }

    // SAFETY: We hold the lock and the state exists
    let state = registry.get_mut(&key).unwrap();
    let state_ptr = state.downcast_mut::<R>().unwrap() as *mut TttStreamingState<R>;

    // SAFETY: The registry is static and we're returning a reference that
    // will be used within the same forward_launch call
    unsafe { &mut *state_ptr }
}

/// Remove a streaming state from the registry.
pub fn remove_streaming_state<R: CubeRuntime + 'static>(stream_id: u64) -> Option<TttStreamingState<R>> {
    let key = StreamKey::new(stream_id);
    let mut registry = STREAMING_REGISTRY.lock().unwrap();
    registry.remove(&key).and_then(|any| {
        // Try to downcast and extract
        match any.0.downcast::<TttStreamingState<R>>() {
            Ok(boxed) => Some(*boxed),
            Err(_) => None,
        }
    })
}

/// Remove a streaming state by ID, triggering its Drop impl for cleanup.
/// Use this when you don't need the state back - just need to clean up.
pub fn remove_streaming_state_by_id(stream_id: u64) {
    let key = StreamKey::new(stream_id);
    let mut registry = STREAMING_REGISTRY.lock().unwrap();
    // Remove triggers AnyStreamingState drop -> TttStreamingState::drop() -> signal_shutdown()
    registry.remove(&key);
}

/// Shutdown and remove a streaming state, returning final weight/bias.
pub fn shutdown_streaming_state<R: CubeRuntime + 'static>(
    stream_id: u64,
) -> Option<(Vec<f32>, Vec<f32>)> {
    remove_streaming_state::<R>(stream_id).map(|state| state.shutdown())
}

impl<R: CubeRuntime> TttStreamingState<R> {
    /// Create a new streaming state and launch the persistent kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn new<F: FloatElement>(
        config: StreamingConfig,
        client: ComputeClient<R>,
        device: R::Device,
        initial_weight: CubeTensor<R>,
        initial_bias: CubeTensor<R>,
        token_eta: CubeTensor<R>,
        ln_weight: CubeTensor<R>,
        ln_bias: CubeTensor<R>,
    ) -> Self {
        // Allocate streaming buffers (single mini-batch sized)
        let xq = empty_device::<R, F>(
            client.clone(),
            device.clone(),
            Shape::from(config.qkv_shape()),
        );
        let xk = empty_device::<R, F>(
            client.clone(),
            device.clone(),
            Shape::from(config.qkv_shape()),
        );
        let xv = empty_device::<R, F>(
            client.clone(),
            device.clone(),
            Shape::from(config.qkv_shape()),
        );
        let ttt_lr_eta_buf = empty_device::<R, F>(
            client.clone(),
            device.clone(),
            Shape::from(config.eta_shape()),
        );
        let output = empty_device::<R, F>(
            client.clone(),
            device.clone(),
            Shape::from(config.qkv_shape()),
        );

        // Control array as a 1D tensor of u32 (we'll treat it as Atomic<u32> in kernel)
        let ctrl_len = config.ctrl_array_len();
        let control = empty_device::<R, u32>(
            client.clone(),
            device.clone(),
            Shape::from([ctrl_len]),
        );

        // Forward intermediates (single mini-batch)
        let fwd_shape = config.qkv_shape();
        let fwd_seq_shape = config.eta_shape();

        let x_hat_fused =
            empty_device::<R, F>(client.clone(), device.clone(), Shape::from(fwd_shape));
        let std_fused =
            empty_device::<R, F>(client.clone(), device.clone(), Shape::from(fwd_seq_shape));
        let grad_output_fused =
            empty_device::<R, F>(client.clone(), device.clone(), Shape::from(fwd_shape));
        let grad_x_hat_fused =
            empty_device::<R, F>(client.clone(), device.clone(), Shape::from(fwd_shape));
        let grad_l_wrt_Z1 =
            empty_device::<R, F>(client.clone(), device.clone(), Shape::from(fwd_shape));
        let x_hat_ln =
            empty_device::<R, F>(client.clone(), device.clone(), Shape::from(fwd_shape));
        let std_ln =
            empty_device::<R, F>(client.clone(), device.clone(), Shape::from(fwd_seq_shape));

        // Allocate result buffers for returning data without blocking on persistent kernel
        let result_output = empty_device::<R, F>(
            client.clone(),
            device.clone(),
            Shape::from(config.qkv_shape()),
        );
        let result_weight = empty_device::<R, F>(
            client.clone(),
            device.clone(),
            Shape::from(config.weight_shape()),
        );
        let result_bias = empty_device::<R, F>(
            client.clone(),
            device.clone(),
            Shape::from(config.bias_shape()),
        );

        // Allocate proper weight/bias buffers with full 4D shape
        // The initial_weight/bias might be broadcast views without actual memory for each batch
        let weight = empty_device::<R, F>(
            client.clone(),
            device.clone(),
            Shape::from(config.weight_shape()),
        );
        let bias = empty_device::<R, F>(
            client.clone(),
            device.clone(),
            Shape::from(config.bias_shape()),
        );

        let tensors = StreamingBufferTensors {
            xq,
            xk,
            xv,
            ttt_lr_eta: ttt_lr_eta_buf,
            output,
            result_output,
            control,
            weight,
            bias,
            result_weight,
            result_bias,
            token_eta,
            ln_weight,
            ln_bias,
            x_hat_fused,
            std_fused,
            grad_output_fused,
            grad_x_hat_fused,
            grad_l_wrt_Z1,
            x_hat_ln,
            std_ln,
        };

        // Create async stream for memory transfers
        let stream = AsyncStream::new();

        // Create a separate client for the persistent kernel with its own stream ID.
        // This prevents the persistent kernel from blocking normal operations on the main client.
        let mut kernel_client = client.clone();
        let kernel_stream_id = next_persistent_kernel_stream_id();
        trace!("[HOST] using kernel stream ID: {}", kernel_stream_id);
        // SAFETY: We're setting a unique stream ID that won't conflict with normal operations.
        // The kernel will run on this separate stream.
        unsafe {
            kernel_client.set_stream(kernel_stream_id);
        }

        // Get all cached pointers BEFORE launching the kernel.
        // This is critical: get_resource() triggers cross-stream synchronization,
        // so we must obtain all pointers before the persistent kernel starts.
        // After the kernel launches, accessing kernel-written buffers via get_resource()
        // would block forever waiting for the (never-ending) kernel to complete.
        let cached_xq_ptr: GpuPtr<'static, f32> =
            unsafe { std::mem::transmute(stream.ptr::<f32, R>(&client, &tensors.xq.handle)) };
        let cached_xk_ptr: GpuPtr<'static, f32> =
            unsafe { std::mem::transmute(stream.ptr::<f32, R>(&client, &tensors.xk.handle)) };
        let cached_xv_ptr: GpuPtr<'static, f32> =
            unsafe { std::mem::transmute(stream.ptr::<f32, R>(&client, &tensors.xv.handle)) };
        let cached_eta_ptr: GpuPtr<'static, f32> =
            unsafe { std::mem::transmute(stream.ptr::<f32, R>(&client, &tensors.ttt_lr_eta.handle)) };
        let cached_ctrl_ptr: GpuPtr<'static, u32> =
            unsafe { std::mem::transmute(stream.ptr::<u32, R>(&client, &tensors.control.handle)) };
        let cached_output_ptr: GpuPtr<'static, f32> =
            unsafe { std::mem::transmute(stream.ptr::<f32, R>(&client, &tensors.output.handle)) };
        let cached_weight_ptr: GpuPtr<'static, f32> =
            unsafe { std::mem::transmute(stream.ptr::<f32, R>(&client, &tensors.weight.handle)) };
        let cached_bias_ptr: GpuPtr<'static, f32> =
            unsafe { std::mem::transmute(stream.ptr::<f32, R>(&client, &tensors.bias.handle)) };
        let cached_result_output_ptr: GpuPtr<'static, f32> =
            unsafe { std::mem::transmute(stream.ptr::<f32, R>(&client, &tensors.result_output.handle)) };
        let cached_result_weight_ptr: GpuPtr<'static, f32> =
            unsafe { std::mem::transmute(stream.ptr::<f32, R>(&client, &tensors.result_weight.handle)) };
        let cached_result_bias_ptr: GpuPtr<'static, f32> =
            unsafe { std::mem::transmute(stream.ptr::<f32, R>(&client, &tensors.result_bias.handle)) };

        let mut state = Self {
            config,
            stream,
            tensors,
            client: client.clone(),
            kernel_client,
            is_initialized: false,
            cached_xq_ptr,
            cached_xk_ptr,
            cached_xv_ptr,
            cached_eta_ptr,
            cached_ctrl_ptr,
            cached_output_ptr,
            cached_weight_ptr,
            cached_bias_ptr,
            cached_result_output_ptr,
            cached_result_weight_ptr,
            cached_result_bias_ptr,
        };

        // Initialize weight/bias buffers from initial values
        // The initial values might be broadcast views, so we replicate for each batch
        trace!("[HOST] initializing weight/bias from initial values...");
        let src_weight: GpuPtr<f32> = state.stream.ptr(&client, &initial_weight.handle);
        let src_bias: GpuPtr<f32> = state.stream.ptr(&client, &initial_bias.handle);
        // Use cached pointers for destination
        let dst_weight = state.cached_weight_ptr;
        let dst_bias = state.cached_bias_ptr;

        // Check if source is already the full size or needs replication
        let per_head_weight_size = config.head_dim * config.head_dim;
        let per_head_bias_size = config.head_dim;

        if src_weight.len() == config.num_heads * per_head_weight_size {
            // Source is [num_heads, head_dim, head_dim] - need to replicate for each batch
            for batch in 0..config.batch_size {
                let batch_offset = batch * config.num_heads * per_head_weight_size;
                state.stream.copy_d2d(dst_weight, batch_offset, src_weight, 0, config.num_heads * per_head_weight_size);
            }
            for batch in 0..config.batch_size {
                let batch_offset = batch * config.num_heads * per_head_bias_size;
                state.stream.copy_d2d(dst_bias, batch_offset, src_bias, 0, config.num_heads * per_head_bias_size);
            }
        } else {
            // Source already has batch dimension - just copy
            let full_weight_len = config.batch_size * config.num_heads * per_head_weight_size;
            let full_bias_len = config.batch_size * config.num_heads * per_head_bias_size;
            state.stream.copy_d2d(dst_weight, 0, src_weight, 0, full_weight_len.min(src_weight.len()));
            state.stream.copy_d2d(dst_bias, 0, src_bias, 0, full_bias_len.min(src_bias.len()));
        }
        state.stream.sync();

        // Initialize control array to IDLE
        trace!("[HOST] reset_control...");
        state.reset_control();

        // Launch the persistent kernel
        trace!("[HOST] launch_kernel...");
        state.launch_kernel::<F>();
        trace!("[HOST] kernel launched!");
        state.is_initialized = true;

        state
    }

    /// Reset all control flags to zero (idle state).
    fn reset_control(&mut self) {
        let zeros = vec![0u32; self.config.ctrl_array_len()];
        self.stream.write(self.cached_ctrl_ptr, 0, &zeros);
    }

    /// Launch the persistent streaming kernel.
    fn launch_kernel<F: FloatElement>(&self) {
        let kernel_config = self.config.kernel_config();
        let batch_size = self.config.batch_size as u32;
        let num_heads = self.config.num_heads as u32;
        let mini_batch_len = self.config.mini_batch_len;
        let head_dim = self.config.head_dim;
        let threads = self.config.threads;

        let cube_count = CubeCount::Static(batch_size, num_heads, 1);
        let vectorization = LINE_SIZE;

        // Get handle refs with longer lifetime
        let xq_ref = self.tensors.xq.as_handle_ref();
        let xk_ref = self.tensors.xk.as_handle_ref();
        let xv_ref = self.tensors.xv.as_handle_ref();
        let weight_ref = self.tensors.weight.as_handle_ref();
        let bias_ref = self.tensors.bias.as_handle_ref();
        let token_eta_ref = self.tensors.token_eta.as_handle_ref();
        let ttt_lr_eta_ref = self.tensors.ttt_lr_eta.as_handle_ref();
        let ln_weight_ref = self.tensors.ln_weight.as_handle_ref();
        let ln_bias_ref = self.tensors.ln_bias.as_handle_ref();
        let output_ref = self.tensors.output.as_handle_ref();
        let x_hat_fused_ref = self.tensors.x_hat_fused.as_handle_ref();
        let std_fused_ref = self.tensors.std_fused.as_handle_ref();
        let grad_output_fused_ref = self.tensors.grad_output_fused.as_handle_ref();
        let grad_x_hat_fused_ref = self.tensors.grad_x_hat_fused.as_handle_ref();
        let grad_l_wrt_Z1_ref = self.tensors.grad_l_wrt_Z1.as_handle_ref();
        let x_hat_ln_ref = self.tensors.x_hat_ln.as_handle_ref();
        let std_ln_ref = self.tensors.std_ln.as_handle_ref();

        // Create InputsLaunch
        let inputs = InputsLaunch::<F, R>::new(
            xq_ref.as_tensor_arg(vectorization),
            xk_ref.as_tensor_arg(vectorization),
            xv_ref.as_tensor_arg(vectorization),
            weight_ref.as_tensor_arg(vectorization),
            bias_ref.as_tensor_arg(vectorization),
            token_eta_ref.as_tensor_arg(vectorization),
            ttt_lr_eta_ref.as_tensor_arg(vectorization),
            ln_weight_ref.as_tensor_arg(vectorization),
            ln_bias_ref.as_tensor_arg(vectorization),
        );

        // Create OutputsLaunch
        let outputs = OutputsLaunch::<F, R>::new(
            output_ref.as_tensor_arg(vectorization),
            weight_ref.as_tensor_arg(vectorization),
            bias_ref.as_tensor_arg(vectorization),
        );

        // Control array as ArrayArg for Array<u32> kernel parameter
        let control_arg = unsafe {
            ArrayArg::from_raw_parts::<u32>(&self.tensors.control.handle, self.config.ctrl_array_len(), 1)
        };

        let fwd_intermediates = ForwardIntermediatesLaunch::<F, R>::new(
            x_hat_fused_ref.as_tensor_arg(vectorization),
            std_fused_ref.as_tensor_arg(vectorization),
            grad_output_fused_ref.as_tensor_arg(vectorization),
            grad_x_hat_fused_ref.as_tensor_arg(vectorization),
            grad_l_wrt_Z1_ref.as_tensor_arg(vectorization),
            x_hat_ln_ref.as_tensor_arg(vectorization),
            std_ln_ref.as_tensor_arg(vectorization),
        );

        // Dispatch based on tile configuration
        // Use kernel_client which has a separate stream for the persistent kernel
        dispatch_streaming_kernel!(
            &self.kernel_client, cube_count, &self.kernel_client,
            inputs, outputs, control_arg, fwd_intermediates,
            kernel_config,
            mini_batch_len, head_dim, threads
        );
    }

    /// Feed a mini-batch to the kernel using D2D copies and wait for output.
    ///
    /// This uses GPU-to-GPU copies to transfer data from the input tensors
    /// to the streaming buffers, avoiding CPU round-trips.
    pub fn forward_d2d(
        &mut self,
        xq: &CubeTensor<R>,
        xk: &CubeTensor<R>,
        xv: &CubeTensor<R>,
        ttt_lr_eta: &CubeTensor<R>,
    ) -> &CubeTensor<R> {
        trace!("[HOST] forward_d2d start");
        // Get GPU pointers for source tensors (these are new each call, not kernel-written)
        let src_xq: GpuPtr<f32> = self.stream.ptr(&self.client, &xq.handle);
        let src_xk: GpuPtr<f32> = self.stream.ptr(&self.client, &xk.handle);
        let src_xv: GpuPtr<f32> = self.stream.ptr(&self.client, &xv.handle);
        let src_eta: GpuPtr<f32> = self.stream.ptr(&self.client, &ttt_lr_eta.handle);

        // Use cached pointers for destination buffers (avoid get_resource on kernel-written buffers)
        let dst_xq = self.cached_xq_ptr;
        let dst_xk = self.cached_xk_ptr;
        let dst_xv = self.cached_xv_ptr;
        let dst_eta = self.cached_eta_ptr;
        let ctrl_ptr = self.cached_ctrl_ptr;

        trace!("[HOST] D2D copy...");
        // D2D copy input data to streaming buffers
        self.stream.copy_d2d(dst_xq, 0, src_xq, 0, self.config.qkv_len());
        self.stream.copy_d2d(dst_xk, 0, src_xk, 0, self.config.qkv_len());
        self.stream.copy_d2d(dst_xv, 0, src_xv, 0, self.config.qkv_len());
        self.stream.copy_d2d(dst_eta, 0, src_eta, 0, self.config.eta_len());

        // Sync to ensure copies are complete before signaling kernel
        self.stream.sync();
        trace!("[HOST] D2D done, setting READY for {} cubes", self.config.num_cubes());

        // Set READY flag for all cubes (write full control block like working example)
        for cube in 0..self.config.num_cubes() {
            let base = cube * CTRL_ARRAY_SIZE;
            // Write [READY=1, DONE=0, SHUTDOWN=0] like the working example
            self.stream.write(ctrl_ptr, base, &[1u32, 0, 0]);
        }
        trace!("[HOST] READY set, polling DONE...");

        // Wait for all cubes to complete (poll DONE flag, read all 3 like working example)
        for cube in 0..self.config.num_cubes() {
            trace!("[HOST] polling cube {}", cube);
            let base = cube * CTRL_ARRAY_SIZE;
            loop {
                let flags = self.stream.read(ctrl_ptr, base, 3);
                if flags[CTRL_DONE] != 0 {
                    break;
                }
                std::hint::spin_loop();
            }
            trace!("[HOST] cube {} done", cube);
        }

        trace!("[HOST] all cubes done, clearing DONE flags");
        // Reset control flags to idle (write all 3 like working example)
        for cube in 0..self.config.num_cubes() {
            let base = cube * CTRL_ARRAY_SIZE;
            self.stream.write(ctrl_ptr, base, &[0u32, 0, 0]);
        }

        // Copy results to separate buffers that can be read without blocking on persistent kernel
        // Use cached pointers to avoid get_resource() which would sync with kernel stream
        let output_ptr = self.cached_output_ptr;
        let result_output_ptr = self.cached_result_output_ptr;
        let weight_ptr = self.cached_weight_ptr;
        let result_weight_ptr = self.cached_result_weight_ptr;
        let bias_ptr = self.cached_bias_ptr;
        let result_bias_ptr = self.cached_result_bias_ptr;

        trace!("[HOST] weight tensor shape: {:?}, ptr capacity: {}", &self.tensors.weight.shape, weight_ptr.len());
        trace!("[HOST] result_weight ptr capacity: {}", result_weight_ptr.len());

        self.stream.copy_d2d(result_output_ptr, 0, output_ptr, 0, self.config.qkv_len());
        // Copy only what the source buffer actually has
        let weight_src_len = weight_ptr.len();
        let bias_src_len = bias_ptr.len();
        trace!("[HOST] copying weight: {} elements, bias: {} elements", weight_src_len, bias_src_len);
        self.stream.copy_d2d(result_weight_ptr, 0, weight_ptr, 0, weight_src_len);
        self.stream.copy_d2d(result_bias_ptr, 0, bias_ptr, 0, bias_src_len);
        self.stream.sync();

        trace!("[HOST] forward_d2d complete, returning result_output");
        &self.tensors.result_output
    }

    /// Feed a mini-batch to the kernel and wait for output (CPU round-trip version).
    ///
    /// This is slower than `forward_d2d` but works when you have data on the CPU.
    pub fn forward(&mut self, xq: &[f32], xk: &[f32], xv: &[f32], ttt_lr_eta: &[f32]) -> Vec<f32> {
        // Use cached pointers to avoid get_resource() which would sync with kernel stream
        let xq_ptr = self.cached_xq_ptr;
        let xk_ptr = self.cached_xk_ptr;
        let xv_ptr = self.cached_xv_ptr;
        let eta_ptr = self.cached_eta_ptr;
        let ctrl_ptr = self.cached_ctrl_ptr;
        let output_ptr = self.cached_output_ptr;

        // Write input data
        self.stream.write(xq_ptr, 0, xq);
        self.stream.write(xk_ptr, 0, xk);
        self.stream.write(xv_ptr, 0, xv);
        self.stream.write(eta_ptr, 0, ttt_lr_eta);

        // Set READY flag for all cubes
        for cube in 0..self.config.num_cubes() {
            let base = cube * CTRL_ARRAY_SIZE;
            self.stream.write(ctrl_ptr, base + CTRL_READY, &[1u32]);
        }

        // Wait for all cubes to complete (poll DONE flag)
        for cube in 0..self.config.num_cubes() {
            let done_offset = cube * CTRL_ARRAY_SIZE + CTRL_DONE;
            loop {
                let status = self.stream.read(ctrl_ptr, done_offset, 1);
                if status[0] != 0 {
                    break;
                }
                std::hint::spin_loop();
            }
        }

        // Read output
        let output = self.stream.read(output_ptr, 0, self.config.qkv_len());

        // Reset control flags to idle (clear DONE)
        for cube in 0..self.config.num_cubes() {
            let base = cube * CTRL_ARRAY_SIZE;
            self.stream.write(ctrl_ptr, base + CTRL_DONE, &[0u32]);
        }

        output
    }

    /// Shutdown the kernel and return final weight/bias.
    pub fn shutdown(self) -> (Vec<f32>, Vec<f32>) {
        trace!("[HOST] shutdown: signaling SHUTDOWN to {} cubes", self.config.num_cubes());
        // Use cached pointers to avoid get_resource() which would sync with kernel stream
        let ctrl_ptr = self.cached_ctrl_ptr;
        let weight_ptr = self.cached_weight_ptr;
        let bias_ptr = self.cached_bias_ptr;

        // Read current control state before shutdown
        let ctrl_before = self.stream.read(ctrl_ptr, 0, self.config.ctrl_array_len());
        trace!("[HOST] shutdown: control before = {:?}", ctrl_before);

        // Signal shutdown to all cubes by writing full control array
        // Format: [READY=0, DONE=0, SHUTDOWN=1] for each cube
        for cube in 0..self.config.num_cubes() {
            let base = cube * CTRL_ARRAY_SIZE;
            // Write entire control block at once like the working example
            self.stream.write(ctrl_ptr, base, &[0u32, 0, 1]);
        }

        // Read control state after writing shutdown
        let ctrl_after = self.stream.read(ctrl_ptr, 0, self.config.ctrl_array_len());
        trace!("[HOST] shutdown: control after = {:?}", ctrl_after);

        // Poll for kernel to acknowledge shutdown (READY should become MAX)
        trace!("[HOST] shutdown: polling for kernel to acknowledge (READY=MAX)...");
        let start = std::time::Instant::now();
        let timeout = std::time::Duration::from_secs(5);
        let mut all_acknowledged = false;

        while start.elapsed() < timeout {
            let ctrl = self.stream.read(ctrl_ptr, 0, self.config.ctrl_array_len());
            // Check if all cubes have set READY to MAX
            all_acknowledged = (0..self.config.num_cubes()).all(|cube| {
                let base = cube * CTRL_ARRAY_SIZE;
                ctrl[base + CTRL_READY] == u32::MAX
            });
            if all_acknowledged {
                trace!("[HOST] shutdown: all cubes acknowledged, control = {:?}", ctrl);
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }

        if !all_acknowledged {
            let ctrl = self.stream.read(ctrl_ptr, 0, self.config.ctrl_array_len());
            trace!("[HOST] shutdown: TIMEOUT! kernel did not acknowledge. control = {:?}", ctrl);
            // Try reading with longer interval
            for i in 0..5 {
                std::thread::sleep(std::time::Duration::from_millis(100));
                let ctrl = self.stream.read(ctrl_ptr, 0, self.config.ctrl_array_len());
                trace!("[HOST] shutdown: after {}ms more, control = {:?}", (i+1)*100, ctrl);
            }
        }

        // Wait for kernel to finish on the kernel_client's stream
        trace!("[HOST] shutdown: waiting for kernel to finish via wait_for_sync");
        wait_for_sync(&self.kernel_client).expect("sync failed");
        trace!("[HOST] shutdown: kernel finished");

        // Read final weight and bias
        let weight_len = self.config.batch_size
            * self.config.num_heads
            * self.config.head_dim
            * self.config.head_dim;
        let bias_len = self.config.batch_size * self.config.num_heads * self.config.head_dim;

        let weight = self.stream.read(weight_ptr, 0, weight_len);
        let bias = self.stream.read(bias_ptr, 0, bias_len);

        (weight, bias)
    }

    /// Signal shutdown to the kernel without waiting for final state.
    /// Called by Drop to ensure the kernel is stopped.
    fn signal_shutdown(&self) {
        if !self.is_initialized {
            return;
        }

        trace!("[HOST] signal_shutdown: signaling SHUTDOWN to {} cubes", self.config.num_cubes());
        // Use cached pointer to avoid get_resource() which would sync with kernel stream
        let ctrl_ptr = self.cached_ctrl_ptr;

        // Signal shutdown to all cubes
        for cube in 0..self.config.num_cubes() {
            let base = cube * CTRL_ARRAY_SIZE;
            self.stream.write(ctrl_ptr, base, &[0u32, 0, 1]);
        }

        // Wait for kernel to finish
        trace!("[HOST] signal_shutdown: waiting for kernel to finish");
        if let Err(e) = wait_for_sync(&self.kernel_client) {
            trace!("[HOST] signal_shutdown: sync error (may be expected): {:?}", e);
        }
        trace!("[HOST] signal_shutdown: done");
    }
}

impl<R: CubeRuntime> Drop for TttStreamingState<R> {
    fn drop(&mut self) {
        self.signal_shutdown();
    }
}

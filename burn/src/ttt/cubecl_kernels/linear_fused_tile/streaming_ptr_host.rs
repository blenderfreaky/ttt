//! Host-side state management for pointer-based streaming kernel.
//!
//! This provides true zero-copy input by writing tensor addresses to a pointer table
//! that the kernel reads directly.

use burn::tensor::Shape;
use burn_cubecl::{CubeRuntime, FloatElement, ops::numeric::empty_device, tensor::CubeTensor};
use cubecl::prelude::*;
use cubecl::frontend::ArrayArg;
use thundercube::{
    prelude::{D4, D8, D16, D32, D64, LINE_SIZE},
    streaming::{AsyncStream, GpuPtr},
    util::wait_for_sync,
};
use super::{
    helpers::Params,
    forward::{InputsLaunch, OutputsLaunch, ForwardIntermediatesLaunch},
    streaming_ptr::{
        CTRL_ARRAY_SIZE, PTR_OUTPUT, PTR_TABLE_SIZE,
        STATUS_DONE, STATUS_IDLE, STATUS_READY, STATUS_SHUTDOWN,
        fused_ttt_streaming_ptr_kernel,
    },
};
use crate::ttt::cubecl_kernels::FusedTttConfig;
use tracing::trace;

/// Configuration for pointer-based streaming.
#[derive(Debug, Clone, Copy)]
pub struct PtrStreamingConfig {
    pub batch_size: usize,
    pub num_heads: usize,
    pub mini_batch_len: usize,
    pub head_dim: usize,
    pub epsilon: f32,
    pub threads: usize,
    pub debug: bool,
}

impl PtrStreamingConfig {
    pub fn new(
        batch_size: usize,
        num_heads: usize,
        mini_batch_len: usize,
        head_dim: usize,
        epsilon: f32,
        threads: usize,
    ) -> Self {
        Self {
            batch_size,
            num_heads,
            mini_batch_len,
            head_dim,
            epsilon,
            threads,
            debug: false,
        }
    }

    pub fn with_debug(mut self, debug: bool) -> Self {
        self.debug = debug;
        self
    }

    pub fn num_cubes(&self) -> usize {
        self.batch_size * self.num_heads
    }
}

/// All tensor handles for the streaming kernel.
/// Most are only accessed at launch; kept alive for kernel duration.
pub struct PtrStreamingTensors<R: CubeRuntime> {
    // --- Host-accessed tensors ---
    pub ptr_table: CubeTensor<R>,
    pub control: CubeTensor<R>,
    pub output: CubeTensor<R>,
    pub weight_out: CubeTensor<R>,
    pub bias_out: CubeTensor<R>,
    // --- Array buffers (for HIP pointer loads) ---
    pub xq_buf: CubeTensor<R>,
    pub xk_buf: CubeTensor<R>,
    pub xv_buf: CubeTensor<R>,
    pub eta_buf: CubeTensor<R>,
    // --- Inputs struct tensors ---
    pub xq_scratch: CubeTensor<R>,
    pub xk_scratch: CubeTensor<R>,
    pub xv_scratch: CubeTensor<R>,
    pub weight: CubeTensor<R>,
    pub bias: CubeTensor<R>,
    pub token_eta: CubeTensor<R>,
    pub ttt_lr_eta_scratch: CubeTensor<R>,
    pub ln_weight: CubeTensor<R>,
    pub ln_bias: CubeTensor<R>,
    // --- ForwardIntermediates tensors ---
    pub x_hat_fused: CubeTensor<R>,
    pub std_fused: CubeTensor<R>,
    pub grad_output_fused: CubeTensor<R>,
    pub grad_x_hat_fused: CubeTensor<R>,
    pub grad_l_wrt_Z1: CubeTensor<R>,
    pub x_hat_ln: CubeTensor<R>,
    pub std_ln: CubeTensor<R>,
}

/// State for pointer-based streaming execution.
pub struct TttPtrStreamingState<R: CubeRuntime> {
    pub config: PtrStreamingConfig,
    pub stream: AsyncStream,
    pub tensors: PtrStreamingTensors<R>,
    /// Raw GPU pointers for async access
    pub ptr_table_ptr: GpuPtr<'static, u64>,
    pub control_ptr: GpuPtr<'static, u32>,
    pub output_ptr: GpuPtr<'static, f32>,
    /// Client for GPU operations
    client: ComputeClient<R>,
}

impl<R: CubeRuntime + 'static> TttPtrStreamingState<R> {
    /// Create a new streaming state and launch the persistent kernel.
    #[allow(unused_variables)]
    pub fn new<F: FloatElement>(
        config: PtrStreamingConfig,
        client: ComputeClient<R>,
        device: R::Device,
        // Initial state tensors
        initial_weight: CubeTensor<R>,
        initial_bias: CubeTensor<R>,
        token_eta: CubeTensor<R>,
        ln_weight: CubeTensor<R>,
        ln_bias: CubeTensor<R>,
    ) -> Self {
        let stream = AsyncStream::new();
        let num_cubes = config.num_cubes();
        let mini_batch_len = config.mini_batch_len;
        let head_dim = config.head_dim;

        // Helper to allocate a tensor
        let alloc = |shape: Vec<usize>| {
            empty_device::<R, F>(client.clone(), device.clone(), Shape::from(shape))
        };
        let alloc_u64 = |shape: Vec<usize>| {
            empty_device::<R, u64>(client.clone(), device.clone(), Shape::from(shape))
        };
        let alloc_u32 = |shape: Vec<usize>| {
            empty_device::<R, u32>(client.clone(), device.clone(), Shape::from(shape))
        };

        // --- Allocate all tensors ---
        let ptr_table = alloc_u64(vec![PTR_TABLE_SIZE]);
        let control = alloc_u32(vec![num_cubes * CTRL_ARRAY_SIZE]);

        // Array buffers for HIP pointer loads (per-cube sized)
        let qkv_buf_size = mini_batch_len * head_dim;
        let eta_buf_size = mini_batch_len;
        let xq_buf = alloc(vec![qkv_buf_size]);
        let xk_buf = alloc(vec![qkv_buf_size]);
        let xv_buf = alloc(vec![qkv_buf_size]);
        let eta_buf = alloc(vec![eta_buf_size]);

        // Scratch tensors for Inputs (single mini-batch)
        let xq_scratch = alloc(vec![mini_batch_len, head_dim]);
        let xk_scratch = alloc(vec![mini_batch_len, head_dim]);
        let xv_scratch = alloc(vec![mini_batch_len, head_dim]);
        let ttt_lr_eta_scratch = alloc(vec![mini_batch_len]);

        // Outputs
        let output = alloc(vec![mini_batch_len, head_dim]);
        let weight_out = alloc(vec![head_dim, head_dim]);
        let bias_out = alloc(vec![head_dim]);

        // ForwardIntermediates (single mini-batch)
        let x_hat_fused = alloc(vec![mini_batch_len, head_dim]);
        let std_fused = alloc(vec![mini_batch_len]);
        let grad_output_fused = alloc(vec![mini_batch_len, head_dim]);
        let grad_x_hat_fused = alloc(vec![mini_batch_len, head_dim]);
        let grad_l_wrt_Z1 = alloc(vec![mini_batch_len, head_dim]);
        let x_hat_ln = alloc(vec![mini_batch_len, head_dim]);
        let std_ln = alloc(vec![mini_batch_len]);

        // Get raw pointers for host access
        let ptr_table_ptr: GpuPtr<'static, u64> =
            unsafe { std::mem::transmute(stream.ptr::<u64, R>(&client, &ptr_table.handle)) };
        let control_ptr: GpuPtr<'static, u32> =
            unsafe { std::mem::transmute(stream.ptr::<u32, R>(&client, &control.handle)) };
        let output_ptr: GpuPtr<'static, f32> =
            unsafe { std::mem::transmute(stream.ptr::<f32, R>(&client, &output.handle)) };

        // Initialize control to IDLE
        let zeros = vec![STATUS_IDLE; num_cubes * CTRL_ARRAY_SIZE];
        stream.write(control_ptr, 0, &zeros);

        // Write output address to pointer table
        let output_addr = output_ptr.address();
        stream.write(ptr_table_ptr, PTR_OUTPUT, &[output_addr]);

        let tensors = PtrStreamingTensors {
            ptr_table,
            control,
            output,
            weight_out,
            bias_out,
            xq_buf,
            xk_buf,
            xv_buf,
            eta_buf,
            xq_scratch,
            xk_scratch,
            xv_scratch,
            weight: initial_weight,
            bias: initial_bias,
            token_eta,
            ttt_lr_eta_scratch,
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

        let mut state = Self {
            config,
            stream,
            tensors,
            ptr_table_ptr,
            control_ptr,
            output_ptr,
            client,
        };

        // Launch the persistent kernel
        trace!("ptr_stream: launching persistent kernel");
        state.launch_kernel::<F>();
        trace!("ptr_stream: kernel launched");

        state
    }

    /// Launch the persistent streaming kernel with pointer indirection.
    fn launch_kernel<F: FloatElement>(&self) {
        let fused_config = FusedTttConfig::new(
            self.config.mini_batch_len,
            self.config.head_dim,
            self.config.epsilon,
            self.config.threads,
        );
        let debug = self.config.debug;
        let batch_size = self.config.batch_size as u32;
        let num_heads = self.config.num_heads as u32;
        let mini_batch_len = self.config.mini_batch_len;
        let head_dim = self.config.head_dim;
        let threads = self.config.threads;

        let cube_count = CubeCount::Static(batch_size, num_heads, 1);
        let vectorization = LINE_SIZE;

        // Array sizes (in Line<F> units)
        let qkv_arr_len = mini_batch_len * head_dim / LINE_SIZE;
        let eta_arr_len = mini_batch_len / LINE_SIZE;

        // Create ArrayArgs
        let xq_arg = unsafe { ArrayArg::from_raw_parts::<F>(&self.tensors.xq_buf.handle, qkv_arr_len, vectorization) };
        let xk_arg = unsafe { ArrayArg::from_raw_parts::<F>(&self.tensors.xk_buf.handle, qkv_arr_len, vectorization) };
        let xv_arg = unsafe { ArrayArg::from_raw_parts::<F>(&self.tensors.xv_buf.handle, qkv_arr_len, vectorization) };
        let eta_arg = unsafe { ArrayArg::from_raw_parts::<F>(&self.tensors.eta_buf.handle, eta_arr_len, vectorization) };

        // Get all handle refs (must outlive the Launch structs)
        let xq_scratch_ref = self.tensors.xq_scratch.as_handle_ref();
        let xk_scratch_ref = self.tensors.xk_scratch.as_handle_ref();
        let xv_scratch_ref = self.tensors.xv_scratch.as_handle_ref();
        let weight_ref = self.tensors.weight.as_handle_ref();
        let bias_ref = self.tensors.bias.as_handle_ref();
        let token_eta_ref = self.tensors.token_eta.as_handle_ref();
        let ttt_lr_eta_ref = self.tensors.ttt_lr_eta_scratch.as_handle_ref();
        let ln_weight_ref = self.tensors.ln_weight.as_handle_ref();
        let ln_bias_ref = self.tensors.ln_bias.as_handle_ref();
        let output_ref = self.tensors.output.as_handle_ref();
        let weight_out_ref = self.tensors.weight_out.as_handle_ref();
        let bias_out_ref = self.tensors.bias_out.as_handle_ref();
        let x_hat_fused_ref = self.tensors.x_hat_fused.as_handle_ref();
        let std_fused_ref = self.tensors.std_fused.as_handle_ref();
        let grad_output_fused_ref = self.tensors.grad_output_fused.as_handle_ref();
        let grad_x_hat_fused_ref = self.tensors.grad_x_hat_fused.as_handle_ref();
        let grad_l_wrt_Z1_ref = self.tensors.grad_l_wrt_Z1.as_handle_ref();
        let x_hat_ln_ref = self.tensors.x_hat_ln.as_handle_ref();
        let std_ln_ref = self.tensors.std_ln.as_handle_ref();
        let ptr_table_ref = self.tensors.ptr_table.as_handle_ref();
        let control_ref = self.tensors.control.as_handle_ref();

        // Build InputsLaunch
        let inputs = InputsLaunch::<F, R>::new(
            xq_scratch_ref.as_tensor_arg(vectorization),
            xk_scratch_ref.as_tensor_arg(vectorization),
            xv_scratch_ref.as_tensor_arg(vectorization),
            weight_ref.as_tensor_arg(vectorization),
            bias_ref.as_tensor_arg(vectorization),
            token_eta_ref.as_tensor_arg(vectorization),
            ttt_lr_eta_ref.as_tensor_arg(vectorization),
            ln_weight_ref.as_tensor_arg(vectorization),
            ln_bias_ref.as_tensor_arg(vectorization),
        );

        // Build OutputsLaunch
        let outputs = OutputsLaunch::<F, R>::new(
            output_ref.as_tensor_arg(vectorization),
            weight_out_ref.as_tensor_arg(vectorization),
            bias_out_ref.as_tensor_arg(vectorization),
        );

        // Build ForwardIntermediatesLaunch
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
        match (mini_batch_len, head_dim, threads) {
            (8, 32, 8) => {
                type P<E> = Params<E, D8, D32, D4, D8>;
                let cube_dim = CubeDim::new(&self.client, threads);
                fused_ttt_streaming_ptr_kernel::launch::<P<F>, _>(
                    &self.client, cube_count, cube_dim,
                    ptr_table_ref.as_tensor_arg(1),
                    control_ref.as_tensor_arg(1),
                    xq_arg, xk_arg, xv_arg, eta_arg,
                    inputs, outputs, fwd_intermediates,
                    fused_config,
                    debug,
                ).unwrap();
            }
            (16, 64, 64) => {
                type P<E> = Params<E, D16, D64, D4, D4>;
                let cube_dim = CubeDim::new(&self.client, threads);
                fused_ttt_streaming_ptr_kernel::launch::<P<F>, _>(
                    &self.client, cube_count, cube_dim,
                    ptr_table_ref.as_tensor_arg(1),
                    control_ref.as_tensor_arg(1),
                    xq_arg, xk_arg, xv_arg, eta_arg,
                    inputs, outputs, fwd_intermediates,
                    fused_config,
                    debug,
                ).unwrap();
            }
            (64, 64, 64) => {
                type P<E> = Params<E, D64, D64, D8, D8>;
                let cube_dim = CubeDim::new(&self.client, threads);
                fused_ttt_streaming_ptr_kernel::launch::<P<F>, _>(
                    &self.client, cube_count, cube_dim,
                    ptr_table_ref.as_tensor_arg(1),
                    control_ref.as_tensor_arg(1),
                    xq_arg, xk_arg, xv_arg, eta_arg,
                    inputs, outputs, fwd_intermediates,
                    fused_config,
                    debug,
                ).unwrap();
            }
            _ => panic!(
                "Unsupported streaming config: mini_batch_len={}, head_dim={}, threads={}",
                mini_batch_len, head_dim, threads
            ),
        }
    }

    /// Feed a mini-batch by writing tensor addresses to the pointer table.
    ///
    /// This is true zero-copy - we just write the addresses, no data is copied.
    pub fn feed_mini_batch(
        &mut self,
        xq: &CubeTensor<R>,
        xk: &CubeTensor<R>,
        xv: &CubeTensor<R>,
        ttt_lr_eta: &CubeTensor<R>,
    ) {
        trace!("ptr_stream: feed_mini_batch start");

        // Get addresses of input tensors
        let xq_ptr: GpuPtr<f32> = self.stream.ptr(&self.client, &xq.handle);
        let xk_ptr: GpuPtr<f32> = self.stream.ptr(&self.client, &xk.handle);
        let xv_ptr: GpuPtr<f32> = self.stream.ptr(&self.client, &xv.handle);
        let eta_ptr: GpuPtr<f32> = self.stream.ptr(&self.client, &ttt_lr_eta.handle);

        // Write addresses to pointer table
        let addrs = [
            xq_ptr.address(),
            xk_ptr.address(),
            xv_ptr.address(),
            eta_ptr.address(),
            self.output_ptr.address(), // Keep output address updated
        ];
        self.stream.write(self.ptr_table_ptr, 0, &addrs);

        trace!("ptr_stream: writing READY to {} cubes", self.config.num_cubes());

        // Signal READY to all cubes
        let num_cubes = self.config.num_cubes();
        let ready_signals: Vec<u32> = (0..num_cubes).map(|_| STATUS_READY).collect();
        self.stream.write(self.control_ptr, 0, &ready_signals);
    }

    /// Wait for processing to complete and return output data.
    pub fn wait_for_done(&self) -> Vec<f32> {
        trace!("ptr_stream: wait_for_done start");
        let num_cubes = self.config.num_cubes();

        // Poll until all cubes report DONE
        loop {
            let status = self.stream.read(self.control_ptr, 0, num_cubes);
            if status.iter().all(|&s| s == STATUS_DONE) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_micros(10));
        }

        trace!("ptr_stream: all cubes DONE, resetting to IDLE");

        // Reset control to IDLE
        let idle_signals: Vec<u32> = (0..num_cubes).map(|_| STATUS_IDLE).collect();
        self.stream.write(self.control_ptr, 0, &idle_signals);

        // Read output
        let output_len = self.config.batch_size
            * self.config.num_heads
            * self.config.mini_batch_len
            * self.config.head_dim;
        trace!("ptr_stream: reading output ({} elements)", output_len);
        self.stream.read(self.output_ptr, 0, output_len)
    }

    /// Convenience method: feed and wait.
    pub fn forward(
        &mut self,
        xq: &CubeTensor<R>,
        xk: &CubeTensor<R>,
        xv: &CubeTensor<R>,
        ttt_lr_eta: &CubeTensor<R>,
    ) -> Vec<f32> {
        self.feed_mini_batch(xq, xk, xv, ttt_lr_eta);
        self.wait_for_done()
    }

    /// Signal shutdown and retrieve final weight/bias.
    pub fn shutdown(self) -> (Vec<f32>, Vec<f32>) {
        trace!("ptr_stream: shutdown start");
        let num_cubes = self.config.num_cubes();

        // Signal SHUTDOWN to all cubes
        trace!("ptr_stream: writing SHUTDOWN to {} cubes", num_cubes);
        let shutdown_signals: Vec<u32> = (0..num_cubes).map(|_| STATUS_SHUTDOWN).collect();
        self.stream.write(self.control_ptr, 0, &shutdown_signals);

        // Sync to ensure the SHUTDOWN signal is written
        self.stream.sync();

        // Wait for kernel to exit and write final state
        trace!("ptr_stream: waiting for kernel exit");
        std::thread::sleep(std::time::Duration::from_millis(50));

        // Sync GPU to ensure all kernel operations complete
        wait_for_sync(&self.client).expect("GPU sync failed");

        // Read final weight and bias
        let weight_len =
            self.config.batch_size * self.config.num_heads * self.config.head_dim * self.config.head_dim;
        let bias_len = self.config.batch_size * self.config.num_heads * self.config.head_dim;

        trace!("ptr_stream: reading final weight ({}) and bias ({})", weight_len, bias_len);
        let weight_ptr: GpuPtr<f32> = self.stream.ptr(&self.client, &self.tensors.weight_out.handle);
        let bias_ptr: GpuPtr<f32> = self.stream.ptr(&self.client, &self.tensors.bias_out.handle);

        let weight = self.stream.read(weight_ptr, 0, weight_len);
        let bias = self.stream.read(bias_ptr, 0, bias_len);

        trace!("ptr_stream: shutdown complete");
        (weight, bias)
    }
}

// Safety: The streaming state is designed to be used from a single thread
// but the underlying handles are Send
unsafe impl<R: CubeRuntime> Send for TttPtrStreamingState<R> {}

#[cfg(all(test, feature = "rocm"))]
mod tests {
    use std::sync::Arc;
    use super::*;
    use burn::tensor::{Tensor, TensorData};
    use burn::module::Param;
    use burn_cubecl::ops::numeric::empty_device;
    use cubecl::hip::HipRuntime;
    use crate::ttt::{
        GpuBackend,
        layer::{Qkv, TTTInnerModel, TTTInputsInner},
        linear::{TTTLinear, TTTLinearConfig},
    };

    type R = HipRuntime;
    type RefBackend = GpuBackend;

    fn assert_data_close(a: &[f32], b: &[f32], rtol: f32, atol: f32, name: &str) {
        assert_eq!(a.len(), b.len(), "{name}: Data sizes don't match: {} vs {}", a.len(), b.len());

        let mut max_diff = 0.0f32;
        let mut max_idx = 0;
        let mut max_av = 0.0f32;
        let mut max_bv = 0.0f32;

        for (i, (&av, &bv)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (av - bv).abs();
            if diff > max_diff {
                max_diff = diff;
                max_idx = i;
                max_av = av;
                max_bv = bv;
            }
        }

        let tolerance = atol + rtol * max_bv.abs();
        assert!(
            max_diff <= tolerance,
            "{name}: Max mismatch at index {max_idx}: {max_av} vs {max_bv} (diff: {max_diff}, tolerance: {tolerance})",
        );
    }

    /// Test ptr streaming kernel against CPU reference implementation.
    #[test]
    fn test_ptr_streaming_vs_cpu() {
        let batch_size = 1usize;
        let num_heads = 1usize;
        let head_dim = 32usize;
        let seq_len = 8usize; // mini_batch_size
        let hidden_size = num_heads * head_dim;
        let epsilon = 1e-6f64;
        let threads = 8;

        let config = Arc::new(crate::ttt::TTTConfig {
            num_heads,
            hidden_size,
            token_size: hidden_size,
            mini_batch_size: seq_len,
            base_lr: 1.0,
            epsilon,
            threads,
            ..crate::ttt::TTTConfig::new(crate::ttt::TEST_VOCAB_SIZE)
        });
        let linear_config = Arc::new(TTTLinearConfig::new());

        // Use GPU for both reference and streaming kernel
        let gpu_device = <R as cubecl::Runtime>::Device::default();
        let ref_device = Default::default();

        // Create random input tensors on GPU
        let xq_ref: Tensor<RefBackend, 4> = Tensor::random(
            [batch_size, num_heads, seq_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &ref_device,
        );
        let xk_ref: Tensor<RefBackend, 4> = Tensor::random(
            [batch_size, num_heads, seq_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &ref_device,
        );
        let xv_ref: Tensor<RefBackend, 4> = Tensor::random(
            [batch_size, num_heads, seq_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &ref_device,
        );
        let token_eta_ref: Tensor<RefBackend, 1> =
            Tensor::arange(1..(seq_len as i64 + 1), &ref_device)
                .float()
                .recip();
        let ttt_lr_eta_ref: Tensor<RefBackend, 3> = Tensor::random(
            [batch_size, num_heads, seq_len],
            burn::tensor::Distribution::Uniform(0.01, 0.05),
            &ref_device,
        );

        // Get data as vectors
        let xq_data: Vec<f32> = xq_ref.to_data().to_vec().unwrap();
        let xk_data: Vec<f32> = xk_ref.to_data().to_vec().unwrap();
        let xv_data: Vec<f32> = xv_ref.to_data().to_vec().unwrap();
        let token_eta_data: Vec<f32> = token_eta_ref.to_data().to_vec().unwrap();
        let ttt_lr_eta_data: Vec<f32> = ttt_lr_eta_ref.to_data().to_vec().unwrap();

        // Create GPU reference implementation
        let ttt_linear_ref: TTTLinear<RefBackend> =
            TTTLinear::new(&config, &linear_config, &ref_device);
        let mut state_ref = ttt_linear_ref.init_state(batch_size);

        // Get weight/bias/ln params
        let weight_init_data: Vec<f32> =
            ttt_linear_ref.weight_init.val().to_data().to_vec().unwrap();
        let bias_init_data: Vec<f32> = ttt_linear_ref.bias_init.val().to_data().to_vec().unwrap();
        let ln_weight_data: Vec<f32> = ttt_linear_ref
            .layer_norm
            .weight
            .val()
            .to_data()
            .to_vec()
            .unwrap();
        let ln_bias_data: Vec<f32> = ttt_linear_ref
            .layer_norm
            .bias
            .val()
            .to_data()
            .to_vec()
            .unwrap();

        // Run GPU reference
        let inputs_ref = TTTInputsInner {
            qkv: Qkv {
                xq: xq_ref.clone(),
                xk: xk_ref.clone(),
                xv: xv_ref.clone(),
            },
            token_eta: token_eta_ref.clone(),
            ttt_lr_eta: ttt_lr_eta_ref.clone(),
            start_idx: 0,
        };

        let output_ref = ttt_linear_ref.forward_mini_batch(&mut state_ref, &inputs_ref, 0..seq_len);
        let output_ref_data: Vec<f32> = output_ref.to_data().to_vec().unwrap();
        let weight_ref_data: Vec<f32> = state_ref.weight.to_data().to_vec().unwrap();
        let bias_ref_data: Vec<f32> = state_ref.bias.to_data().to_vec().unwrap();

        // --- Now run the ptr streaming kernel ---
        let client = R::client(&gpu_device);
        let stream = AsyncStream::new();

        let ptr_config = PtrStreamingConfig::new(
            batch_size, num_heads, seq_len, head_dim, epsilon as f32, threads,
        ).with_debug(true);

        // Allocate GPU tensors and write data
        let xq_gpu = empty_device::<R, f32>(client.clone(), gpu_device.clone(),
            Shape::from([batch_size, num_heads, seq_len, head_dim]));
        let xk_gpu = empty_device::<R, f32>(client.clone(), gpu_device.clone(),
            Shape::from([batch_size, num_heads, seq_len, head_dim]));
        let xv_gpu = empty_device::<R, f32>(client.clone(), gpu_device.clone(),
            Shape::from([batch_size, num_heads, seq_len, head_dim]));
        let ttt_lr_eta_gpu = empty_device::<R, f32>(client.clone(), gpu_device.clone(),
            Shape::from([batch_size, num_heads, seq_len]));

        stream.write(stream.ptr(&client, &xq_gpu.handle), 0, &xq_data);
        stream.write(stream.ptr(&client, &xk_gpu.handle), 0, &xk_data);
        stream.write(stream.ptr(&client, &xv_gpu.handle), 0, &xv_data);
        stream.write(stream.ptr(&client, &ttt_lr_eta_gpu.handle), 0, &ttt_lr_eta_data);

        // Allocate state tensors
        let weight_gpu = empty_device::<R, f32>(client.clone(), gpu_device.clone(),
            Shape::from([num_heads, head_dim, head_dim]));
        let bias_gpu = empty_device::<R, f32>(client.clone(), gpu_device.clone(),
            Shape::from([num_heads, head_dim]));
        let token_eta_gpu = empty_device::<R, f32>(client.clone(), gpu_device.clone(),
            Shape::from([seq_len]));
        let ln_weight_gpu = empty_device::<R, f32>(client.clone(), gpu_device.clone(),
            Shape::from([num_heads, head_dim]));
        let ln_bias_gpu = empty_device::<R, f32>(client.clone(), gpu_device.clone(),
            Shape::from([num_heads, head_dim]));

        stream.write(stream.ptr(&client, &weight_gpu.handle), 0, &weight_init_data);
        stream.write(stream.ptr(&client, &bias_gpu.handle), 0, &bias_init_data);
        stream.write(stream.ptr(&client, &token_eta_gpu.handle), 0, &token_eta_data);
        stream.write(stream.ptr(&client, &ln_weight_gpu.handle), 0, &ln_weight_data);
        stream.write(stream.ptr(&client, &ln_bias_gpu.handle), 0, &ln_bias_data);
        stream.sync();

        // Create streaming state and run
        let mut state = TttPtrStreamingState::new::<f32>(
            ptr_config,
            client.clone(),
            gpu_device.clone(),
            weight_gpu,
            bias_gpu,
            token_eta_gpu,
            ln_weight_gpu,
            ln_bias_gpu,
        );

        let output_streaming = state.forward(&xq_gpu, &xk_gpu, &xv_gpu, &ttt_lr_eta_gpu);
        let (weight_streaming, bias_streaming) = state.shutdown();

        // Debug output
        eprintln!("Output streaming (first 8): {:?}", &output_streaming[..8.min(output_streaming.len())]);
        eprintln!("Output ref (first 8):       {:?}", &output_ref_data[..8.min(output_ref_data.len())]);

        // Compare results
        assert_data_close(&output_streaming, &output_ref_data, 1e-3, 1e-4, "output");
        assert_data_close(&weight_streaming, &weight_ref_data, 1e-3, 1e-4, "weight");
        assert_data_close(&bias_streaming, &bias_ref_data, 1e-3, 1e-4, "bias");

        eprintln!("Ptr streaming vs CPU test passed!");
    }
}

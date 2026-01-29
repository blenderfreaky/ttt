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
};
use super::{
    helpers::Params,
    streaming_ptr::{
        CTRL_ARRAY_SIZE, PTR_OUTPUT, PTR_TABLE_SIZE,
        STATUS_DONE, STATUS_IDLE, STATUS_READY, STATUS_SHUTDOWN,
        fused_ttt_streaming_ptr_kernel,
    },
};
use crate::ttt::cubecl_kernels::FusedTttConfig;

/// Configuration for pointer-based streaming.
#[derive(Debug, Clone, Copy)]
pub struct PtrStreamingConfig {
    pub batch_size: usize,
    pub num_heads: usize,
    pub mini_batch_len: usize,
    pub head_dim: usize,
    pub epsilon: f32,
    pub threads: usize,
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
        }
    }

    pub fn num_cubes(&self) -> usize {
        self.batch_size * self.num_heads
    }
}

/// Buffer handles for the pointer-based streaming kernel.
pub struct PtrStreamingTensors<R: CubeRuntime> {
    /// Pointer table [PTR_TABLE_SIZE] - holds u64 addresses
    pub ptr_table: CubeTensor<R>,
    /// Control array [batch * heads] - atomic u32 status flags
    pub control: CubeTensor<R>,
    /// Output weight [batch, heads, head_dim, head_dim]
    pub weight_out: CubeTensor<R>,
    /// Output bias [batch, heads, head_dim]
    pub bias_out: CubeTensor<R>,
    /// Output tensor [batch, heads, mini_batch_len, head_dim]
    pub output: CubeTensor<R>,
    /// Weight tensor (kernel state)
    pub weight: CubeTensor<R>,
    /// Bias tensor (kernel state)
    pub bias: CubeTensor<R>,
    /// Token eta tensor
    pub token_eta: CubeTensor<R>,
    /// Layer norm weight
    pub ln_weight: CubeTensor<R>,
    /// Layer norm bias
    pub ln_bias: CubeTensor<R>,
    /// Dummy buffer for xq Array parameter [batch * heads * mini_batch_len * head_dim / 4]
    pub xq_buf: CubeTensor<R>,
    /// Dummy buffer for xk Array parameter
    pub xk_buf: CubeTensor<R>,
    /// Dummy buffer for xv Array parameter
    pub xv_buf: CubeTensor<R>,
    /// Dummy buffer for eta Array parameter [batch * heads * mini_batch_len / 4]
    pub eta_buf: CubeTensor<R>,
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
        // Initial state tensors (passed to kernel)
        initial_weight: CubeTensor<R>,
        initial_bias: CubeTensor<R>,
        token_eta: CubeTensor<R>,
        ln_weight: CubeTensor<R>,
        ln_bias: CubeTensor<R>,
    ) -> Self {

        let stream = AsyncStream::new();
        let num_cubes = config.num_cubes();

        // Allocate pointer table as CubeTensor
        let ptr_table = empty_device::<R, u64>(
            client.clone(),
            device.clone(),
            Shape::from([PTR_TABLE_SIZE]),
        );

        // Allocate control array
        let control = empty_device::<R, u32>(
            client.clone(),
            device.clone(),
            Shape::from([num_cubes * CTRL_ARRAY_SIZE]),
        );

        // Allocate output tensors
        let weight_out = empty_device::<R, F>(
            client.clone(),
            device.clone(),
            Shape::from([
                config.batch_size,
                config.num_heads,
                config.head_dim,
                config.head_dim,
            ]),
        );
        let bias_out = empty_device::<R, F>(
            client.clone(),
            device.clone(),
            Shape::from([config.batch_size, config.num_heads, config.head_dim]),
        );
        let output = empty_device::<R, F>(
            client.clone(),
            device.clone(),
            Shape::from([
                config.batch_size,
                config.num_heads,
                config.mini_batch_len,
                config.head_dim,
            ]),
        );

        // Allocate dummy Array buffers for kernel parameters
        // These get predictable buffer_N names that injected HIP code can reference
        // Size: batch * heads * mini_batch_len * head_dim (for qkv)
        // Size: batch * heads * mini_batch_len (for eta)
        let qkv_buf_size = config.batch_size * config.num_heads * config.mini_batch_len * config.head_dim;
        let eta_buf_size = config.batch_size * config.num_heads * config.mini_batch_len;
        let xq_buf = empty_device::<R, F>(
            client.clone(),
            device.clone(),
            Shape::from([qkv_buf_size]),
        );
        let xk_buf = empty_device::<R, F>(
            client.clone(),
            device.clone(),
            Shape::from([qkv_buf_size]),
        );
        let xv_buf = empty_device::<R, F>(
            client.clone(),
            device.clone(),
            Shape::from([qkv_buf_size]),
        );
        let eta_buf = empty_device::<R, F>(
            client.clone(),
            device.clone(),
            Shape::from([eta_buf_size]),
        );

        // Get raw pointers BEFORE launching the kernel
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
            weight_out,
            bias_out,
            output,
            weight: initial_weight,
            bias: initial_bias,
            token_eta,
            ln_weight,
            ln_bias,
            xq_buf,
            xk_buf,
            xv_buf,
            eta_buf,
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
        state.launch_kernel::<F>();

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
        let batch_size = self.config.batch_size as u32;
        let num_heads = self.config.num_heads as u32;
        let mini_batch_len = self.config.mini_batch_len;
        let head_dim = self.config.head_dim;
        let threads = self.config.threads;

        let cube_count = CubeCount::Static(batch_size, num_heads, 1);
        let vectorization = LINE_SIZE;

        // Get handle refs
        let ptr_table_ref = self.tensors.ptr_table.as_handle_ref();
        let control_ref = self.tensors.control.as_handle_ref();
        let weight_ref = self.tensors.weight.as_handle_ref();
        let bias_ref = self.tensors.bias.as_handle_ref();
        let token_eta_ref = self.tensors.token_eta.as_handle_ref();
        let ln_weight_ref = self.tensors.ln_weight.as_handle_ref();
        let ln_bias_ref = self.tensors.ln_bias.as_handle_ref();
        let weight_out_ref = self.tensors.weight_out.as_handle_ref();
        let bias_out_ref = self.tensors.bias_out.as_handle_ref();

        // Array sizes for ArrayArgs (in Line<F> units = float4)
        let qkv_arr_len = self.config.batch_size * self.config.num_heads * mini_batch_len * head_dim / 4;
        let eta_arr_len = self.config.batch_size * self.config.num_heads * mini_batch_len / 4;

        // Create ArrayArgs for the Array parameters (only one match arm executes, no need to clone)
        let xq_arg = unsafe { ArrayArg::from_raw_parts::<F>(&self.tensors.xq_buf.handle, qkv_arr_len, vectorization) };
        let xk_arg = unsafe { ArrayArg::from_raw_parts::<F>(&self.tensors.xk_buf.handle, qkv_arr_len, vectorization) };
        let xv_arg = unsafe { ArrayArg::from_raw_parts::<F>(&self.tensors.xv_buf.handle, qkv_arr_len, vectorization) };
        let eta_arg = unsafe { ArrayArg::from_raw_parts::<F>(&self.tensors.eta_buf.handle, eta_arr_len, vectorization) };

        // Dispatch based on tile configuration
        match (mini_batch_len, head_dim, threads) {
            (8, 32, 8) => {
                type P<E> = Params<E, D8, D32, D4, D8>;
                let cube_dim = CubeDim::new(&self.client, threads);
                fused_ttt_streaming_ptr_kernel::launch::<P<F>, _>(
                    &self.client, cube_count, cube_dim,
                    ptr_table_ref.as_tensor_arg(1), // u64, no vectorization
                    control_ref.as_tensor_arg(1),   // atomic u32
                    weight_ref.as_tensor_arg(vectorization),
                    bias_ref.as_tensor_arg(vectorization),
                    token_eta_ref.as_tensor_arg(vectorization),
                    ln_weight_ref.as_tensor_arg(vectorization),
                    ln_bias_ref.as_tensor_arg(vectorization),
                    weight_out_ref.as_tensor_arg(vectorization),
                    bias_out_ref.as_tensor_arg(vectorization),
                    xq_arg, xk_arg, xv_arg, eta_arg,
                    fused_config,
                ).unwrap();
            }
            (16, 32, 16) => {
                type P<E> = Params<E, D16, D32, D4, D8>;
                let cube_dim = CubeDim::new(&self.client, threads);
                fused_ttt_streaming_ptr_kernel::launch::<P<F>, _>(
                    &self.client, cube_count, cube_dim,
                    ptr_table_ref.as_tensor_arg(1),
                    control_ref.as_tensor_arg(1),
                    weight_ref.as_tensor_arg(vectorization),
                    bias_ref.as_tensor_arg(vectorization),
                    token_eta_ref.as_tensor_arg(vectorization),
                    ln_weight_ref.as_tensor_arg(vectorization),
                    ln_bias_ref.as_tensor_arg(vectorization),
                    weight_out_ref.as_tensor_arg(vectorization),
                    bias_out_ref.as_tensor_arg(vectorization),
                    xq_arg, xk_arg, xv_arg, eta_arg,
                    fused_config,
                ).unwrap();
            }
            (16, 64, 64) => {
                type P<E> = Params<E, D16, D64, D4, D4>;
                let cube_dim = CubeDim::new(&self.client, threads);
                fused_ttt_streaming_ptr_kernel::launch::<P<F>, _>(
                    &self.client, cube_count, cube_dim,
                    ptr_table_ref.as_tensor_arg(1),
                    control_ref.as_tensor_arg(1),
                    weight_ref.as_tensor_arg(vectorization),
                    bias_ref.as_tensor_arg(vectorization),
                    token_eta_ref.as_tensor_arg(vectorization),
                    ln_weight_ref.as_tensor_arg(vectorization),
                    ln_bias_ref.as_tensor_arg(vectorization),
                    weight_out_ref.as_tensor_arg(vectorization),
                    bias_out_ref.as_tensor_arg(vectorization),
                    xq_arg, xk_arg, xv_arg, eta_arg,
                    fused_config,
                ).unwrap();
            }
            (64, 64, 64) => {
                type P<E> = Params<E, D64, D64, D8, D8>;
                let cube_dim = CubeDim::new(&self.client, threads);
                fused_ttt_streaming_ptr_kernel::launch::<P<F>, _>(
                    &self.client, cube_count, cube_dim,
                    ptr_table_ref.as_tensor_arg(1),
                    control_ref.as_tensor_arg(1),
                    weight_ref.as_tensor_arg(vectorization),
                    bias_ref.as_tensor_arg(vectorization),
                    token_eta_ref.as_tensor_arg(vectorization),
                    ln_weight_ref.as_tensor_arg(vectorization),
                    ln_bias_ref.as_tensor_arg(vectorization),
                    weight_out_ref.as_tensor_arg(vectorization),
                    bias_out_ref.as_tensor_arg(vectorization),
                    xq_arg, xk_arg, xv_arg, eta_arg,
                    fused_config,
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

        // Signal READY to all cubes
        let num_cubes = self.config.num_cubes();
        let ready_signals: Vec<u32> = (0..num_cubes).map(|_| STATUS_READY).collect();
        self.stream.write(self.control_ptr, 0, &ready_signals);
    }

    /// Wait for processing to complete and return output data.
    pub fn wait_for_done(&self) -> Vec<f32> {
        let num_cubes = self.config.num_cubes();

        // Poll until all cubes report DONE
        loop {
            let status = self.stream.read(self.control_ptr, 0, num_cubes);
            if status.iter().all(|&s| s == STATUS_DONE) {
                break;
            }
            std::thread::sleep(std::time::Duration::from_micros(10));
        }

        // Reset control to IDLE
        let idle_signals: Vec<u32> = (0..num_cubes).map(|_| STATUS_IDLE).collect();
        self.stream.write(self.control_ptr, 0, &idle_signals);

        // Read output
        let output_len = self.config.batch_size
            * self.config.num_heads
            * self.config.mini_batch_len
            * self.config.head_dim;
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
        let num_cubes = self.config.num_cubes();

        // Signal SHUTDOWN to all cubes
        let shutdown_signals: Vec<u32> = (0..num_cubes).map(|_| STATUS_SHUTDOWN).collect();
        self.stream.write(self.control_ptr, 0, &shutdown_signals);

        // Wait a bit for kernel to exit and write final state
        std::thread::sleep(std::time::Duration::from_millis(10));

        // Read final weight and bias
        let weight_len =
            self.config.batch_size * self.config.num_heads * self.config.head_dim * self.config.head_dim;
        let bias_len = self.config.batch_size * self.config.num_heads * self.config.head_dim;

        let weight_ptr: GpuPtr<f32> = self.stream.ptr(&self.client, &self.tensors.weight_out.handle);
        let bias_ptr: GpuPtr<f32> = self.stream.ptr(&self.client, &self.tensors.bias_out.handle);

        let weight = self.stream.read(weight_ptr, 0, weight_len);
        let bias = self.stream.read(bias_ptr, 0, bias_len);

        (weight, bias)
    }
}

// Safety: The streaming state is designed to be used from a single thread
// but the underlying handles are Send
unsafe impl<R: CubeRuntime> Send for TttPtrStreamingState<R> {}

#[cfg(all(test, feature = "rocm"))]
mod tests {
    use super::*;
    use burn_cubecl::ops::numeric::empty_device;
    use cubecl::hip::HipRuntime;

    type R = HipRuntime;

    /// Test that pointer indirection works: xq data should be copied to output.
    #[test]
    fn test_ptr_streaming_passthrough() {
        let device = <R as cubecl::Runtime>::Device::default();
        let client = R::client(&device);

        let batch_size = 1;
        let num_heads = 1;
        let mini_batch_len = 16;
        let head_dim = 64;
        let epsilon = 1e-6;
        let threads = 64;

        let config = PtrStreamingConfig::new(
            batch_size, num_heads, mini_batch_len, head_dim, epsilon, threads,
        );

        // Create test tensors with known values
        let qkv_shape = Shape::from([batch_size, num_heads, mini_batch_len, head_dim]);
        let weight_shape = Shape::from([num_heads, head_dim, head_dim]);
        let ln_shape = Shape::from([num_heads, head_dim]);
        let eta_shape = Shape::from([mini_batch_len]);

        // Fill xq with sequential values for easy verification
        let xq_data: Vec<f32> = (0..batch_size * num_heads * mini_batch_len * head_dim)
            .map(|i| i as f32 * 0.01)
            .collect();
        let xk_data: Vec<f32> = vec![0.1; batch_size * num_heads * mini_batch_len * head_dim];
        let xv_data: Vec<f32> = vec![0.2; batch_size * num_heads * mini_batch_len * head_dim];
        let eta_data: Vec<f32> = vec![0.001; batch_size * num_heads * mini_batch_len];

        // Create input tensors using empty_device and write
        let stream = AsyncStream::new();

        let xq = empty_device::<R, f32>(client.clone(), device.clone(), qkv_shape.clone());
        let xk = empty_device::<R, f32>(client.clone(), device.clone(), qkv_shape.clone());
        let xv = empty_device::<R, f32>(client.clone(), device.clone(), qkv_shape.clone());
        let ttt_lr_eta = empty_device::<R, f32>(
            client.clone(), device.clone(),
            Shape::from([batch_size, num_heads, mini_batch_len]),
        );

        // Write data to tensors
        let xq_ptr: GpuPtr<f32> = stream.ptr(&client, &xq.handle);
        let xk_ptr: GpuPtr<f32> = stream.ptr(&client, &xk.handle);
        let xv_ptr: GpuPtr<f32> = stream.ptr(&client, &xv.handle);
        let eta_ptr: GpuPtr<f32> = stream.ptr(&client, &ttt_lr_eta.handle);

        stream.write(xq_ptr, 0, &xq_data);
        stream.write(xk_ptr, 0, &xk_data);
        stream.write(xv_ptr, 0, &xv_data);
        stream.write(eta_ptr, 0, &eta_data);
        stream.sync();

        // Create initial state tensors (identity weight, zero bias)
        let weight_data: Vec<f32> = (0..num_heads * head_dim * head_dim)
            .map(|i| {
                let r = (i / head_dim) % head_dim;
                let c = i % head_dim;
                if r == c { 1.0 } else { 0.0 }
            })
            .collect();
        let bias_data: Vec<f32> = vec![0.0; num_heads * head_dim];
        let token_eta_data: Vec<f32> = vec![1.0; mini_batch_len];
        let ln_weight_data: Vec<f32> = vec![1.0; num_heads * head_dim];
        let ln_bias_data: Vec<f32> = vec![0.0; num_heads * head_dim];

        let initial_weight = empty_device::<R, f32>(client.clone(), device.clone(), weight_shape);
        let initial_bias = empty_device::<R, f32>(client.clone(), device.clone(), Shape::from([num_heads, head_dim]));
        let token_eta = empty_device::<R, f32>(client.clone(), device.clone(), eta_shape);
        let ln_weight = empty_device::<R, f32>(client.clone(), device.clone(), ln_shape.clone());
        let ln_bias = empty_device::<R, f32>(client.clone(), device.clone(), ln_shape);

        // Write state data
        let weight_ptr: GpuPtr<f32> = stream.ptr(&client, &initial_weight.handle);
        let bias_ptr: GpuPtr<f32> = stream.ptr(&client, &initial_bias.handle);
        let token_eta_ptr: GpuPtr<f32> = stream.ptr(&client, &token_eta.handle);
        let ln_weight_ptr: GpuPtr<f32> = stream.ptr(&client, &ln_weight.handle);
        let ln_bias_ptr: GpuPtr<f32> = stream.ptr(&client, &ln_bias.handle);

        stream.write(weight_ptr, 0, &weight_data);
        stream.write(bias_ptr, 0, &bias_data);
        stream.write(token_eta_ptr, 0, &token_eta_data);
        stream.write(ln_weight_ptr, 0, &ln_weight_data);
        stream.write(ln_bias_ptr, 0, &ln_bias_data);
        stream.sync();

        // Create streaming state (this launches the kernel)
        let mut state = TttPtrStreamingState::new::<f32>(
            config,
            client.clone(),
            device.clone(),
            initial_weight,
            initial_bias,
            token_eta,
            ln_weight,
            ln_bias,
        );

        // Feed mini-batch and wait for output
        let output = state.forward(&xq, &xk, &xv, &ttt_lr_eta);

        // The kernel currently just copies xq to output, so they should match
        assert_eq!(output.len(), xq_data.len(), "Output length mismatch");

        // Check that output matches xq (within floating point tolerance)
        let max_diff = output.iter().zip(xq_data.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        assert!(max_diff < 1e-5, "Output differs from xq by {}, expected < 1e-5", max_diff);
        eprintln!("Pointer indirection test passed! Max diff: {}", max_diff);

        // Shutdown
        let (final_weight, final_bias) = state.shutdown();
        eprintln!("Final weight len: {}, bias len: {}", final_weight.len(), final_bias.len());
    }
}

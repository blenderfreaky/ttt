use ttt_kernels::tensor_bundle;

tensor_bundle! {
    /// Input tensors for the TTT fused kernel.
    pub struct TttInputs[9] {
        xq, xk, xv, weight, bias, token_eta, ttt_lr_eta, ln_weight, ln_bias
    }
}

tensor_bundle! {
    /// Output tensors from the TTT fused kernel.
    pub struct TttOutputs[3] { output, weight, bias }
}

/// Marker type for the TTT fused kernel.
#[derive(Debug, Clone, Copy)]
pub struct TttKernel;

use cubecl::prelude::*;
use thundercube::prelude::*;

use crate::ttt::cubecl_kernels::FusedTttConfig;

/// Forward pass input tensors grouped into a struct.
#[derive(CubeType, CubeLaunch)]
pub struct ForwardInputs<F: Float> {
    pub xq: Tensor<F>,
    pub xk: Tensor<F>,
    pub xv: Tensor<F>,
    pub weight: Tensor<F>,
    pub bias: Tensor<F>,
    pub token_eta: Tensor<F>,
    pub ttt_lr_eta: Tensor<F>,
    pub ln_weight: Tensor<F>,
    pub ln_bias: Tensor<F>,
}

/// Gradient output tensors grouped into a struct.
#[derive(CubeType, CubeLaunch)]
pub struct GradOutputs<F: Float> {
    pub grad_xq: Tensor<F>,
    pub grad_xk: Tensor<F>,
    pub grad_xv: Tensor<F>,
    pub grad_weight: Tensor<F>,
    pub grad_bias: Tensor<F>,
    pub grad_ttt_lr_eta: Tensor<F>,
    pub grad_ln_weight: Tensor<F>,
    pub grad_ln_bias: Tensor<F>,
}

/// Fused TTT-Linear backward pass kernel.
///
/// Each CUBE handles one (batch, head) pair.
/// Thread layout: (head_dim, seq_len)
#[cube(launch)]
pub fn fused_ttt_backward_kernel<F: Float,
CS: Dim,
F: Dim,
>(
    // Inputs
    xq: &Tensor<Line<F>>,
    xk: &Tensor<Line<F>>,
    xv: &Tensor<Line<F>>,
    token_eta: &Tensor<Line<F>>,
    ttt_lr_eta: &Tensor<Line<F>>,
    ln_weight: &Tensor<Line<F>>,
    ln_bias: &Tensor<Line<F>>,
    // State (rw)
    weight: &mut Tensor<Line<F>>,
    bias: &mut Tensor<Line<F>>,
    // Output (write-only)
    output: &mut Tensor<Line<F>>,
    #[comptime] config: FusedTttConfig,
) {
    let B = xq.shape(0);
    let NH = xq.shape(1);
    let CS = comptime!(config.mini_batch_len);
    let F = comptime!(config.head_dim);

    let batch_idx = CUBE_POS_X as usize;
    let head_idx = CUBE_POS_Y as usize;

    let epsilon = comptime!(config.epsilon());

    // let mut weight_t = Rt::<F>::new(F, F);
    // let mut bias_v = Rv::<F>::new(F);
    //
    // plane::mm_AB(&mut weight_t, &slice_2d(weight, batch_idx, head_idx));

    type CS_F_Tile = St<>;

    let mut weight_t = St
}

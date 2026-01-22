#![allow(non_camel_case_types, non_snake_case)]
#![allow(dead_code)]

use cubecl::prelude::*;
use thundercube::{binary_ops::*, impl_reduction_ops, prelude::*, reduction_ops::*, unary_ops::*, util::index_2d};

impl_reduction_ops! {
    SumSq<F> {
        identity => Line::<F>::empty(LINE_SIZE).fill(F::new(0.0));
        combine(a, b) => a + b * b;
        finalize(line) => line[0] + line[1] + line[2] + line[3];
        plane_reduce(val) => plane_sum(val);
    }
}

use crate::ttt::cubecl_kernels::FusedTttConfig;

#[derive(CubeType, CubeLaunch)]
pub struct Inputs<F: Float> {
    pub xq: Tensor<Line<F>>,
    pub xk: Tensor<Line<F>>,
    pub xv: Tensor<Line<F>>,
    pub weight: Tensor<Line<F>>,
    pub bias: Tensor<Line<F>>,
    pub token_eta: Tensor<Line<F>>,
    pub ttt_lr_eta: Tensor<Line<F>>,
    pub ln_weight: Tensor<Line<F>>,
    pub ln_bias: Tensor<Line<F>>,
}

// We use a trait here because
// inherent assoc types are unstable
#[cube]
pub trait ParamsTrait: Send + Sync + 'static {
    type E: Float;
    type CS: Dim;
    type F: Dim;

    type CS_Reg: Dim;
    type F_Reg: Dim;

    // CubeCL won't let us do default impls
    fn cs_f_tile() -> St<Self::E, Self::CS, Self::F>;
    fn f_cs_tile() -> St<Self::E, Self::F, Self::CS>;
    fn cs_cs_tile() -> St<Self::E, Self::CS, Self::CS>;
    fn cs_vec() -> Sv<Self::E, Self::CS>;
    fn f_f_tile() -> St<Self::E, Self::F, Self::F>;
    fn f_vec() -> Sv<Self::E, Self::F>;

    fn cs_cs_reg() -> Rt<Self::E, Self::CS_Reg, Self::CS_Reg>;
    fn cs_f_reg() -> Rt<Self::E, Self::CS_Reg, Self::F_Reg>;
    fn cs_reg() -> Rv<Self::E, Self::CS_Reg>;
}

pub struct Params<E: Float, CS: Dim, F: Dim, CS_Reg: Dim, F_Reg: Dim> {
    _phantom: std::marker::PhantomData<(E, CS, F, CS_Reg, F_Reg)>,
}

#[cube]
impl<E: Float, CS: Dim, F: Dim, CS_Reg: Dim, F_Reg: Dim> ParamsTrait
    for Params<E, CS, F, CS_Reg, F_Reg>
{
    type E = E;
    type CS = CS;
    type F = F;
    type CS_Reg = CS_Reg;
    type F_Reg = F_Reg;

    fn cs_f_tile() -> St<Self::E, Self::CS, Self::F> {
        St::new()
    }
    fn f_cs_tile() -> St<Self::E, Self::F, Self::CS> {
        St::new()
    }
    fn cs_cs_tile() -> St<Self::E, Self::CS, Self::CS> {
        St::new()
    }
    fn cs_vec() -> Sv<Self::E, Self::CS> {
        Sv::new()
    }
    fn f_f_tile() -> St<Self::E, Self::F, Self::F> {
        St::new()
    }
    fn f_vec() -> Sv<Self::E, Self::F> {
        Sv::new()
    }

    fn cs_cs_reg() -> Rt<Self::E, Self::CS_Reg, Self::CS_Reg> {
        Rt::new()
    }
    fn cs_f_reg() -> Rt<Self::E, Self::CS_Reg, Self::F_Reg> {
        Rt::new()
    }

    fn cs_reg() -> Rv<Self::E, Self::CS_Reg> {
        Rv::new()
    }
}

/// Computes layer norm forward and L2 loss gradient backpropagated through layer norm.
/// Uses shared memory tiles with plane-level cooperative operations.
///
/// Given input `x` of shape [R, C] (R rows, C columns), target of same shape,
/// and layer norm parameters weight/bias of shape [C]:
///
/// Forward:
///   mean[r] = sum(x[r, :]) / C
///   var[r] = sum((x[r, :] - mean[r])^2) / C
///   std[r] = sqrt(var[r] + epsilon)
///   norm[r, c] = (x[r, c] - mean[r]) / std[r]
///   out[r, c] = weight[c] * norm[r, c] + bias[c]
///
/// L2 gradient backprop:
///   dl_dout = out - target
///   dl_dnorm = dl_dout * weight
///   dl_dx = (dl_dnorm * C - sum(dl_dnorm) - norm * sum(dl_dnorm * norm)) / (std * C)
///
/// # Arguments
/// * `x` - Input tile [R, C], will be modified to contain dl_dx (the gradient)
/// * `target` - Target tile [R, C] for L2 loss
/// * `ln_weight` - Layer norm weight vector [C]
/// * `ln_bias` - Layer norm bias vector [C]
/// * `temp` - Scratch tile [R, C] for intermediate computations
/// * `epsilon` - Small constant for numerical stability
#[cube]
pub fn layer_norm_l2_grad<F: Float, R: Dim, C: Dim>(
    x: &mut St<F, R, C>,
    target: &St<F, R, C>,
    ln_weight: &Rv<F, C>,
    ln_bias: &Rv<F, C>,
    temp: &mut St<F, R, C>,
    #[comptime] epsilon: f32,
) {
    let c_inv = F::cast_from(1.0f32 / (C::VALUE as f32));
    let c_f = F::cast_from(C::VALUE as f32);

    // Step 1: mean = sum_rows(x) / C
    let mut mean: Rv<F, R> = plane::sum_st_rows(x);
    mean.mul_scalar(c_inv);

    // Step 2: x -= mean (col broadcast)
    x.sub_col(&mean);

    // Step 3: var = sum_rows(x^2) / C using SumSqOp
    let mut std: Rv<F, R> = plane::reduce_st_rows::<F, R, C, SumSqOp>(x);
    std.mul_scalar(c_inv);

    // Step 4: std = sqrt(var + epsilon)
    std.add_scalar(F::cast_from(epsilon));
    std.sqrt();

    // Step 5: x /= std → x now contains norm
    x.div_col(&std);

    // Step 6: temp = x (copy norm), then temp = weight * temp + bias
    temp.copy_from(x);
    temp.mul_row(ln_weight);
    temp.add_row(ln_bias);

    // Step 7: temp -= target → temp = dl_dout
    temp.sub(target);

    // Step 8: temp *= weight → temp = dl_dnorm
    temp.mul_row(ln_weight);

    // Step 9: Compute reduction terms
    let sum_dl_dnorm: Rv<F, R> = plane::sum_st_rows(temp);

    // temp = dl_dnorm currently, we need sum(dl_dnorm * norm)
    // Multiply temp by x (norm), sum, then rebuild dl_dnorm
    temp.mul(x);
    let sum_dl_dnorm_norm: Rv<F, R> = plane::sum_st_rows(temp);

    // Recompute dl_dnorm in temp (x still has norm)
    temp.copy_from(x);
    temp.mul_row(ln_weight);
    temp.add_row(ln_bias);
    temp.sub(target);
    temp.mul_row(ln_weight);

    // Step 10: dl_dx = (dl_dnorm * C - sum_dl_dnorm - norm * sum_dl_dnorm_norm) / (std * C)

    // temp = dl_dnorm * C
    temp.mul_scalar(c_f);

    // temp -= sum_dl_dnorm
    temp.sub_col(&sum_dl_dnorm);

    // x = norm * sum_dl_dnorm_norm (col broadcast), then temp -= x
    x.mul_col(&sum_dl_dnorm_norm);
    temp.sub(x);

    // temp /= (std * C)
    std.mul_scalar(c_f);
    temp.div_col(&std);

    // Copy result to x
    x.copy_from(temp);
}

/// Fused TTT-Linear backward pass kernel.
///
/// Each CUBE handles one (batch, head) pair.
/// Thread layout: (head_dim, seq_len)
#[cube(launch)]
pub fn fused_ttt_backward_kernel<P: ParamsTrait>(
    inputs: Inputs<P::E>,
    #[comptime] config: FusedTttConfig,
) {
    let B = inputs.xq.shape(0);
    let NH = inputs.xq.shape(1);
    let CS = P::CS::VALUE;
    let F = P::F::VALUE;

    let batch_idx = CUBE_POS_X as usize;
    let head_idx = CUBE_POS_Y as usize;

    let epsilon = comptime!(config.epsilon());

    let mut weight_smem = P::f_f_tile();
    let mut bias_smem = P::f_vec();

    let mut q_smem = P::f_cs_tile();
    let mut k_smem = P::f_cs_tile();
    let mut v_smem = P::f_cs_tile();

    let mut token_eta_smem = P::cs_vec();
    let mut ttt_lr_eta_smem = P::cs_vec();

    let mut ln_weight_smem = P::f_vec();
    let mut ln_bias_smem = P::f_vec();

    let mut z1_smem = P::cs_f_tile();

    let mut grad_l_wrt_z1_smem = P::cs_f_tile();

    let mut bias_bar_smem = P::cs_f_tile();
    let mut z1_bar_smem = P::cs_f_tile();

    let mut eta_matrix_smem = P::cs_cs_tile();

    // // why 16?
    // let mut ln_smem = Sv::<P::E, D16>::new();

    // let mut matmul_smem = F::cs_f_tile();
    // let mut b_acc_smem = F::cs_f_tile();

    plane::load_st_direct(
        &inputs.weight,
        &mut weight_smem,
        index_2d(&inputs.weight, batch_idx, head_idx),
        0,
        0,
    );
    plane::load_st_direct(
        &inputs.bias,
        &mut bias_smem,
        index_2d(&inputs.bias, batch_idx, head_idx),
        0,
        0,
    );
    plane::load_st_transpose(
        &inputs.xq,
        &mut q_smem,
        index_2d(&inputs.xq, batch_idx, head_idx),
        0,
        0,
    );
    plane::load_st_transpose(
        &inputs.xk,
        &mut k_smem,
        index_2d(&inputs.xk, batch_idx, head_idx),
        0,
        0,
    );
    plane::load_st_transpose(
        &inputs.xv,
        &mut v_smem,
        index_2d(&inputs.xv, batch_idx, head_idx),
        0,
        0,
    );
    plane::load_st_direct(
        &inputs.token_eta,
        &mut token_eta_smem,
        index_2d(&inputs.token_eta, 0, 0),
        0,
        0,
    );
    plane::load_st_direct(
        &inputs.ttt_lr_eta,
        &mut ttt_lr_eta_smem,
        index_2d(&inputs.ttt_lr_eta, batch_idx, head_idx),
        0,
        0,
    );
    plane::load_st_direct(
        &inputs.ln_weight,
        &mut ln_weight_smem,
        index_2d(&inputs.ln_weight, 0, head_idx),
        0,
        0,
    );
    plane::load_st_direct(
        &inputs.ln_bias,
        &mut ln_bias_smem,
        index_2d(&inputs.ln_bias, 0, head_idx),
        0,
        0,
    );

    sync_plane();

    let cs_cs_reg = P::cs_cs_reg();
    let cs_f_reg = P::cs_f_reg();
    // let mut cs_cs_reg_2 = P::cs_cs_reg();
    let cs_reg = P::cs_reg();

    // z1 = x1 @ W + b
    let mut z1_reg = cs_cs_reg;
    z1_reg.zero();
    plane::mma_AtB(&mut z1_reg, &k_smem, &weight_smem);

    let mut bias_reg = cs_reg;
    plane::load_rt_from_st(&bias_smem, &mut bias_reg);
    z1_reg.add_col(&bias_reg);
    let cs_reg = bias_reg;
    plane::store_rt_to_st(&z1_reg, &mut z1_smem);
    let cs_cs_reg = z1_reg;

    // reconstruction_target = v - k
    v_smem.sub(&k_smem);
    let reconstruction_target = v_smem;

    // TODO: ln grad

    let mut eta_matrix_reg = cs_cs_reg;

    eta_matrix_reg.zero();
    let mut token_eta_reg = cs_reg;
    plane::load_rt_from_st(&token_eta_smem, &mut token_eta_reg);
    // TODO: May be transposed
    eta_matrix_reg.add_row(&token_eta_reg);
    let cs_reg = token_eta_reg;

    let mut ttt_lr_eta_reg = cs_reg;
    plane::load_rt_from_st(&ttt_lr_eta_smem, &mut ttt_lr_eta_reg);
    // TODO: May be transposed
    eta_matrix_reg.mul_col(&ttt_lr_eta_reg);
    let cs_reg = ttt_lr_eta_reg;

    plane::store_rt_to_st(&eta_matrix_reg, &mut eta_matrix_smem);

    // TODO: eta_matrix.tril

    let mut attn_scores = P::cs_cs_reg();

    // TODO, may need a transpose on one side
    plane::mma_AtB(&mut attn_scores, &q_smem, &k_smem);

    // TODO: attn_scores.tril

    let mut bias_bar_reg = cs_f_reg;
    bias_bar_reg.zero();
    plane::mma_AtB(&mut bias_bar_reg, &eta_matrix_smem, &grad_l_wrt_z1_smem);
    bias_bar_reg.neg();

    let mut bias_reg = cs_reg;
    plane::load_rt_from_st(&bias_smem, &mut bias_reg);
    bias_bar_reg.add_col(&bias_reg);
    let cs_reg = bias_reg;

    plane::store_rt_to_st(&bias_bar_reg, &mut bias_bar_smem);
    let cs_f_reg = bias_bar_reg;

    // let mut z1_bar_reg = cs_f_reg;
}

#[cfg(test)]
mod tests {
    use super::*;
    use thundercube::test_kernel;
    use thundercube::test_utils::TestFloat;
    use thundercube::util::sync_planes;

    const ROWS: usize = 8;
    const COLS: usize = 8;
    const EPSILON: f32 = 1e-6;

    #[cube(launch)]
    fn test_layer_norm_l2_grad_kernel<F: Float + CubeElement>(
        input: &Tensor<Line<F>>,
        target_tensor: &Tensor<Line<F>>,
        ln_weight_arr: &Array<Line<F>>,
        ln_bias_arr: &Array<Line<F>>,
        output: &mut Tensor<Line<F>>,
    ) {
        // Shared memory tiles
        let mut x = St::<F, D8, D8>::new();
        let mut tgt = St::<F, D8, D8>::new();
        let mut temp = St::<F, D8, D8>::new();

        // Register vectors for ln params (all threads get same values)
        let mut w = Rv::<F, D8>::new();
        let mut b = Rv::<F, D8>::new();

        // Load into shared memory with proper swizzle handling
        plane::load_st_direct(input, &mut x, 0, 0, 0);
        plane::load_st_direct(target_tensor, &mut tgt, 0, 0, 0);

        // Load ln params into register vectors (each thread loads same data)
        w.copy_from_array(ln_weight_arr);
        b.copy_from_array(ln_bias_arr);

        sync_planes();

        layer_norm_l2_grad::<F, D8, D8>(&mut x, &tgt, &w, &b, &mut temp, EPSILON);

        sync_planes();

        // Store result with proper swizzle handling
        plane::store_st_direct(&x, output, 0, 0, 0);
    }

    /// Reference implementation of layer_norm_l2_grad
    fn ref_layer_norm_l2_grad<F: TestFloat>(
        x: &[F],
        target: &[F],
        ln_weight: &[F],
        ln_bias: &[F],
        output: &mut [F],
        rows: usize,
        cols: usize,
        epsilon: f64,
    ) {
        // Allocate intermediate buffers
        let mut mean = vec![0.0f64; rows];
        let mut std = vec![0.0f64; rows];
        let mut norm = vec![0.0f64; rows * cols];

        // Step 1: Compute mean per row
        for r in 0..rows {
            let mut sum = 0.0f64;
            for c in 0..cols {
                sum += x[r * cols + c].into_f64();
            }
            mean[r] = sum / cols as f64;
        }

        // Step 2-3: Center and compute variance
        for r in 0..rows {
            let mut var_sum = 0.0f64;
            for c in 0..cols {
                let centered = x[r * cols + c].into_f64() - mean[r];
                var_sum += centered * centered;
            }
            let var = var_sum / cols as f64;
            std[r] = (var + epsilon).sqrt();
        }

        // Step 4-5: Normalize
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                norm[idx] = (x[idx].into_f64() - mean[r]) / std[r];
            }
        }

        // Step 6: out = weight * norm + bias
        let mut out = vec![0.0f64; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                out[idx] = ln_weight[c].into_f64() * norm[idx] + ln_bias[c].into_f64();
            }
        }

        // Step 7: dl_dout = out - target
        let mut dl_dout = vec![0.0f64; rows * cols];
        for i in 0..rows * cols {
            dl_dout[i] = out[i] - target[i].into_f64();
        }

        // Step 8: dl_dnorm = dl_dout * weight
        let mut dl_dnorm = vec![0.0f64; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                dl_dnorm[idx] = dl_dout[idx] * ln_weight[c].into_f64();
            }
        }

        // Step 9: Compute reduction terms
        let mut sum_dl_dnorm = vec![0.0f64; rows];
        let mut sum_dl_dnorm_norm = vec![0.0f64; rows];
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                sum_dl_dnorm[r] += dl_dnorm[idx];
                sum_dl_dnorm_norm[r] += dl_dnorm[idx] * norm[idx];
            }
        }

        // Step 10: dl_dx = (dl_dnorm * C - sum_dl_dnorm - norm * sum_dl_dnorm_norm) / (std * C)
        for r in 0..rows {
            for c in 0..cols {
                let idx = r * cols + c;
                let term1 = dl_dnorm[idx] * cols as f64;
                let term2 = sum_dl_dnorm[r];
                let term3 = norm[idx] * sum_dl_dnorm_norm[r];
                output[idx] = F::from_f64((term1 - term2 - term3) / (std[r] * cols as f64));
            }
        }
    }

    test_kernel! {
        #[test]
        fn test_layer_norm_l2_grad() for F in [f32, f64] {
            let input: Tensor = [ROWS, COLS] as Uniform(-1.0, 1.0);
            let target: Tensor = [ROWS, COLS] as Uniform(-1.0, 1.0);
            let ln_weight: Array = [COLS] as Uniform(0.5, 1.5);
            let ln_bias: Array = [COLS] as Uniform(-0.5, 0.5);
            let output: Tensor = [ROWS, COLS] as Range;

            assert_eq!(
                test_layer_norm_l2_grad_kernel(input(), target(), ln_weight(), ln_bias(), output())
                    for (1, 1, 1) @ max(32),
                {
                    ref_layer_norm_l2_grad(
                        &input, &target, &ln_weight, &ln_bias, &mut output,
                        ROWS, COLS, EPSILON as f64
                    );
                }
            );
        }
    }
}

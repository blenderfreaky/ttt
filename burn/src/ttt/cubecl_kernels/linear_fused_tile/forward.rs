#![allow(non_camel_case_types, non_snake_case)]
#![allow(dead_code)]

use cubecl::prelude::*;
use thundercube::{impl_reduction_ops, prelude::*, reduction_ops::*, util::index_2d};

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

#[derive(CubeType, CubeLaunch)]
pub struct Outputs<F: Float> {
    pub output: Tensor<Line<F>>,
    pub weight_out: Tensor<Line<F>>,
    pub bias_out: Tensor<Line<F>>,
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
    fn f_f_reg() -> Rt<Self::E, Self::F_Reg, Self::F_Reg>;
    fn cs_reg() -> Rv<Self::E, Self::CS_Reg>;
    fn f_reg() -> Rv<Self::E, Self::F_Reg>;
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
    fn f_f_reg() -> Rt<Self::E, Self::F_Reg, Self::F_Reg> {
        Rt::new()
    }

    fn cs_reg() -> Rv<Self::E, Self::CS_Reg> {
        Rv::new()
    }
    fn f_reg() -> Rv<Self::E, Self::F_Reg> {
        Rv::new()
    }
}

/// Computes layer norm forward pass only.
/// Uses shared memory tiles with plane-level cooperative operations.
///
/// Given input `x` of shape [R, C] (R rows, C columns),
/// and layer norm parameters weight/bias of shape [C]:
///
/// Forward:
///   mean[r] = sum(x[r, :]) / C
///   var[r] = sum((x[r, :] - mean[r])^2) / C
///   std[r] = sqrt(var[r] + epsilon)
///   norm[r, c] = (x[r, c] - mean[r]) / std[r]
///   out[r, c] = weight[c] * norm[r, c] + bias[c]
///
/// # Arguments
/// * `x` - Input tile [R, C], will be modified to contain the normalized output
/// * `ln_weight` - Layer norm weight vector [C]
/// * `ln_bias` - Layer norm bias vector [C]
/// * `epsilon` - Small constant for numerical stability
#[cube]
pub fn layer_norm_forward<F: Float, R: Dim, C: Dim>(
    x: &mut St<F, R, C>,
    ln_weight: &Rv<F, C>,
    ln_bias: &Rv<F, C>,
    #[comptime] epsilon: f32,
) {
    let c_inv = F::cast_from(1.0f32 / (C::VALUE as f32));

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

    // Step 6: x = weight * x + bias
    x.mul_row(ln_weight);
    x.add_row(ln_bias);
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

/// Extract the last row of a shared memory tile into a register vector.
/// Cooperative: all threads participate, result is broadcast to all.
#[cube]
pub fn extract_last_row<F: Float, R: Dim, C: Dim>(st: &St<F, R, C>) -> Rv<F, C> {
    // The last row is at index R-1
    // We read it using plane::sum with a mask that selects only that row
    // Actually, simpler: each thread can just read the last row
    let mut result = Rv::<F, C>::new();
    let last_row = R::VALUE - 1;
    let vec_stride = C::LINES;
    let mask = vec_stride - 1;

    #[unroll]
    for c_line in 0..C::LINES {
        let phys_col = plane::swizzle(last_row, c_line, mask);
        let s_idx = last_row * vec_stride + phys_col;
        result.data[c_line] = st.data[s_idx];
    }
    result
}

/// Fused TTT-Linear forward pass kernel.
///
/// Each CUBE handles one (batch, head) pair.
/// Computes the TTT-Linear forward pass with online weight updates.
///
/// Algorithm:
/// 1. z1 = xk @ W + b
/// 2. reconstruction_target = xv - xk
/// 3. grad_l_wrt_z1 = layer_norm_l2_grad(z1, reconstruction_target)
/// 4. eta_matrix = outer(token_eta, ttt_lr_eta).tril()
/// 5. attn_scores = xq @ xk^T, attn1 = attn_scores.tril()
/// 6. b1_bar = bias - eta_matrix @ grad_l_wrt_z1
/// 7. z1_bar = xq @ W - (eta_matrix * attn1) @ grad_l_wrt_z1 + b1_bar
/// 8. output = xq + layer_norm(z1_bar)
/// 9. weight_out = weight - (last_eta_col * xk).T @ grad
/// 10. bias_out = bias - sum_rows(last_eta_col * grad)
#[cube(launch)]
pub fn fused_ttt_forward_kernel<P: ParamsTrait>(
    inputs: Inputs<P::E>,
    outputs: &mut Outputs<P::E>,
    #[comptime] config: FusedTttConfig,
) {
    let batch_idx = CUBE_POS_X as usize;
    let head_idx = CUBE_POS_Y as usize;

    let epsilon = comptime!(config.epsilon());

    // === Allocate shared memory tiles ===

    // Weight: [F, F]
    let mut weight_smem = P::f_f_tile();

    // Q, K transposed for matmuls: [F, CS]
    let mut q_smem = P::f_cs_tile();
    let mut k_smem = P::f_cs_tile();

    // K and V in original layout for layer norm: [CS, F]
    let mut k_direct_smem = P::cs_f_tile();
    let mut v_direct_smem = P::cs_f_tile();

    // Eta vectors in shared memory: [CS]
    let mut token_eta_smem = P::cs_vec();
    let mut ttt_lr_eta_smem = P::cs_vec();

    // Working tiles
    let mut z1_smem = P::cs_f_tile();           // z1, then grad_l_wrt_z1 after layer_norm_l2_grad
    let mut temp_cs_f_smem = P::cs_f_tile();    // Scratch space
    let mut eta_matrix_smem = P::cs_cs_tile();  // eta outer product (stored transposed)
    let mut attn_smem = P::cs_cs_tile();        // attention scores

    // === Load inputs ===

    let base_qkv = index_2d(&inputs.xq, batch_idx, head_idx);
    let base_weight = index_2d(&inputs.weight, batch_idx, head_idx);
    let base_ttt_lr = index_2d(&inputs.ttt_lr_eta, batch_idx, head_idx);

    // Load weight [F, F]
    plane::load_st_direct(&inputs.weight, &mut weight_smem, base_weight, 0, 0);

    // Load Q, K transposed [F, CS] for matmuls
    plane::load_st_transpose(&inputs.xq, &mut q_smem, base_qkv, 0, 0);
    plane::load_st_transpose(&inputs.xk, &mut k_smem, base_qkv, 0, 0);

    // Load K, V direct [CS, F] for layer norm operations
    plane::load_st_direct(&inputs.xk, &mut k_direct_smem, base_qkv, 0, 0);
    plane::load_st_direct(&inputs.xv, &mut v_direct_smem, base_qkv, 0, 0);

    // Load eta vectors [CS] - treat as [CS, 1] tiles
    plane::load_st_direct(&inputs.token_eta, &mut token_eta_smem, 0, 0, 0);
    plane::load_st_direct(&inputs.ttt_lr_eta, &mut ttt_lr_eta_smem, base_ttt_lr, 0, 0);

    sync_plane();

    // Load bias into register vectors
    // Each thread loads the portion relevant to its sub-tile
    let base_bias = index_2d(&inputs.bias, batch_idx, head_idx);

    // Compute thread's column offset for loading bias
    let threads_n = P::F::VALUE / P::F_Reg::VALUE;
    let thread_n = (UNIT_POS as usize) % threads_n;
    let offset_n = thread_n * P::F_Reg::VALUE;

    let mut bias_reg = P::f_reg();
    plane::load_rt_direct(&inputs.bias, &mut bias_reg, base_bias, 0, offset_n);

    // Load layer norm params into register vectors
    // Each thread loads the full ln params for use in row broadcasts
    let base_ln = index_2d(&inputs.ln_weight, 0, head_idx);
    let mut ln_weight_rv = Rv::<P::E, P::F>::new();
    let mut ln_bias_rv = Rv::<P::E, P::F>::new();
    plane::load_rt_direct(&inputs.ln_weight, &mut ln_weight_rv, base_ln, 0, 0);
    plane::load_rt_direct(&inputs.ln_bias, &mut ln_bias_rv, base_ln, 0, 0);

    // === Step 1: z1 = xk @ W + b ===
    // k_smem is [F, CS], weight_smem is [F, F]
    // mma_AtB: k^T @ W = [CS, F] @ [F, F] = [CS, F]
    let mut z1_reg = P::cs_f_reg();
    z1_reg.zero();
    plane::mma_AtB(&mut z1_reg, &k_smem, &weight_smem);

    // Add bias (broadcast along rows)
    z1_reg.add_row(&bias_reg);

    plane::store_rt_to_st(&z1_reg, &mut z1_smem);

    sync_plane();

    // === Step 2: reconstruction_target = xv - xk ===
    // Both are [CS, F], result stored in v_direct_smem
    v_direct_smem.sub(&k_direct_smem);
    // v_direct_smem now contains reconstruction_target

    sync_plane();

    // === Step 3: grad_l_wrt_z1 = layer_norm_l2_grad(z1, reconstruction_target) ===
    // This overwrites z1_smem with the gradient
    layer_norm_l2_grad::<P::E, P::CS, P::F>(
        &mut z1_smem,
        &v_direct_smem,  // reconstruction_target
        &ln_weight_rv,
        &ln_bias_rv,
        &mut temp_cs_f_smem,
        epsilon,
    );
    // z1_smem now contains grad_l_wrt_z1

    sync_plane();

    // === Step 4: eta_matrix = outer(token_eta, ttt_lr_eta).tril() ===
    // eta[i,j] = token_eta[j] * ttt_lr_eta[i] for j <= i, else 0

    let mut eta_reg = P::cs_cs_reg();
    eta_reg.zero();

    // Load eta vectors into registers
    let mut token_eta_reg = P::cs_reg();
    let mut ttt_lr_eta_reg = P::cs_reg();
    plane::load_rt_from_st(&token_eta_smem, &mut token_eta_reg);
    plane::load_rt_from_st(&ttt_lr_eta_smem, &mut ttt_lr_eta_reg);

    // eta[i,j] = token_eta[j] * ttt_lr_eta[i]
    // token_eta broadcasts along rows: eta[i,j] = token_eta[j]
    eta_reg.add_row(&token_eta_reg);
    // Multiply by ttt_lr_eta along cols: eta[i,j] *= ttt_lr_eta[i]
    eta_reg.mul_col(&ttt_lr_eta_reg);

    plane::store_rt_to_st(&eta_reg, &mut eta_matrix_smem);

    sync_plane();

    // Apply lower triangular mask
    eta_matrix_smem.tril();

    sync_plane();

    // === Step 5: attn_scores = xq @ xk^T, attn1 = attn_scores.tril() ===
    // q_smem [F, CS], k_smem [F, CS]
    // mma_AtB: q^T @ k = [CS, F] @ [F, CS] = [CS, CS]
    let mut attn_reg = P::cs_cs_reg();
    attn_reg.zero();
    plane::mma_AtB(&mut attn_reg, &q_smem, &k_smem);

    plane::store_rt_to_st(&attn_reg, &mut attn_smem);

    sync_plane();

    // Apply lower triangular mask
    attn_smem.tril();

    sync_plane();

    // === Step 6 & 7: Compute z1_bar ===
    // b1_bar = bias - eta_matrix @ grad_l_wrt_z1      [CS, F]
    // z1_bar = xq @ W - (eta_matrix * attn1) @ grad + b1_bar

    // Recompute eta in original (non-transposed) lower triangular form
    // eta[i,j] = token_eta[j] * ttt_lr_eta[i] for j <= i
    eta_reg.zero();
    eta_reg.add_row(&token_eta_reg);  // eta[i,j] = token_eta[j]
    eta_reg.mul_col(&ttt_lr_eta_reg); // eta[i,j] *= ttt_lr_eta[i]
    plane::store_rt_to_st(&eta_reg, &mut eta_matrix_smem);

    sync_plane();

    eta_matrix_smem.tril();

    sync_plane();

    // Compute eta @ grad using mma_AB (eta is [CS, CS], grad is [CS, F])
    let mut eta_grad_reg = P::cs_f_reg();
    eta_grad_reg.zero();
    plane::mma_AB(&mut eta_grad_reg, &eta_matrix_smem, &z1_smem);

    // Compute eta * attn (elementwise, both lower triangular)
    eta_matrix_smem.mul(&attn_smem);

    sync_plane();

    // Compute (eta * attn) @ grad using mma_AB
    let mut eta_attn_grad_reg = P::cs_f_reg();
    eta_attn_grad_reg.zero();
    plane::mma_AB(&mut eta_attn_grad_reg, &eta_matrix_smem, &z1_smem);

    // z1_bar = xq @ W
    let mut z1_bar_reg = P::cs_f_reg();
    z1_bar_reg.zero();
    plane::mma_AtB(&mut z1_bar_reg, &q_smem, &weight_smem);

    // z1_bar -= (eta * attn) @ grad
    z1_bar_reg.sub(&eta_attn_grad_reg);

    // z1_bar += bias (broadcast)
    z1_bar_reg.add_row(&bias_reg);

    // z1_bar -= eta @ grad
    z1_bar_reg.sub(&eta_grad_reg);

    // Store z1_bar to shared memory for layer norm
    plane::store_rt_to_st(&z1_bar_reg, &mut temp_cs_f_smem);

    sync_plane();

    // === Step 8: output = xq + layer_norm(z1_bar) ===
    layer_norm_forward::<P::E, P::CS, P::F>(
        &mut temp_cs_f_smem,
        &ln_weight_rv,
        &ln_bias_rv,
        epsilon,
    );
    // temp_cs_f_smem now contains z1_bar_normalized

    sync_plane();

    // Add xq (need to load it in [CS, F] layout)
    // We can reuse k_direct_smem since we're done with it
    plane::load_st_direct(&inputs.xq, &mut k_direct_smem, base_qkv, 0, 0);

    sync_plane();

    // output = xq + z1_bar_normalized
    temp_cs_f_smem.add(&k_direct_smem);

    sync_plane();

    // Store output
    let base_output = index_2d(&outputs.output, batch_idx, head_idx);
    plane::store_st_direct(&temp_cs_f_smem, &mut outputs.output, base_output, 0, 0);

    // === Step 9 & 10: Weight and bias state updates ===
    // weight_out = weight - (last_eta_col * xk).T @ grad
    // bias_out = bias - sum_rows(last_eta_col * grad)

    // Extract last row of eta_matrix (before it was multiplied by attn)
    // We need to recompute eta since we modified eta_matrix_smem
    eta_reg.zero();
    eta_reg.add_row(&token_eta_reg);
    eta_reg.mul_col(&ttt_lr_eta_reg);
    plane::store_rt_to_st(&eta_reg, &mut eta_matrix_smem);

    sync_plane();

    eta_matrix_smem.tril();

    sync_plane();

    // Extract last row of eta as a column vector
    let last_eta_col: Rv<P::E, P::CS> = extract_last_row(&eta_matrix_smem);

    // Reload k_direct for weight update (we overwrote it with xq earlier)
    plane::load_st_direct(&inputs.xk, &mut k_direct_smem, base_qkv, 0, 0);

    sync_plane();

    // Scale k by last_eta_col: k_direct[i, :] *= last_eta_col[i]
    k_direct_smem.mul_col(&last_eta_col);

    sync_plane();

    // weight_update = (scaled_k).T @ grad = [F, CS] @ [CS, F] = [F, F]
    // Use mma_AtB: k_direct is [CS, F], z1_smem (grad) is [CS, F]
    // mma_AtB(result, A, B) computes A.T @ B where A is [K, M], B is [K, N]
    // Here: A = k_direct [CS, F], B = z1_smem [CS, F]
    // Result = k_direct.T @ z1_smem = [F, CS] @ [CS, F] = [F, F]
    let mut weight_update_reg = P::f_f_reg();
    weight_update_reg.zero();
    plane::mma_AtB(&mut weight_update_reg, &k_direct_smem, &z1_smem);

    // Subtract from weight: weight_out = weight - weight_update
    // First load weight into temp tile, then subtract and store
    let mut weight_out_reg = P::f_f_reg();
    plane::load_rt_from_st(&weight_smem, &mut weight_out_reg);
    weight_out_reg.sub(&weight_update_reg);

    // Store updated weight
    plane::store_rt_to_st(&weight_out_reg, &mut weight_smem);

    sync_plane();

    let base_weight_out = index_2d(&outputs.weight_out, batch_idx, head_idx);
    plane::store_st_direct(&weight_smem, &mut outputs.weight_out, base_weight_out, 0, 0);

    // Bias update: bias_out = bias - sum_rows(last_eta_col * grad)
    // Scale grad by last_eta_col: temp[i, :] = grad[i, :] * last_eta_col[i]
    temp_cs_f_smem.copy_from(&z1_smem);
    temp_cs_f_smem.mul_col(&last_eta_col);

    sync_plane();

    // Sum rows to get bias update: [F]
    let bias_update: Rv<P::E, P::F> = plane::sum_st_cols(&temp_cs_f_smem);

    // Compute bias_out = bias - bias_update
    // Each thread has its portion of bias in bias_reg
    // But sum_st_cols gives the full [F] vector to all threads
    // We need to subtract the corresponding portion

    // Reload original bias into a full shared memory vector for subtraction
    let mut bias_out_smem = P::f_vec();
    plane::load_st_direct(&inputs.bias, &mut bias_out_smem, base_bias, 0, 0);

    sync_plane();

    // Subtract bias_update from bias_out_smem
    // For Sv<E, F> = St<E, F, D1>, we use sub_col since the vector is stored as F rows x 1 col
    bias_out_smem.sub_col(&bias_update);

    sync_plane();

    // Store updated bias
    let base_bias_out = index_2d(&outputs.bias_out, batch_idx, head_idx);
    plane::store_st_direct(&bias_out_smem, &mut outputs.bias_out, base_bias_out, 0, 0);
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

#![allow(non_camel_case_types, non_snake_case)]
#![allow(dead_code)]

use cubecl::prelude::*;
use thundercube::{impl_reduction_ops, prelude::*, reduction_ops::*, util::{index_2d, sync_planes}};

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

    // Eta vectors loaded directly into registers (small, all threads need same data)

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


    sync_planes();

    // Load bias into register vectors
    // Each thread loads the portion relevant to its sub-tile
    let base_bias = index_2d(&inputs.bias, batch_idx, head_idx);

    // Compute thread's column offset for loading bias
    let threads_n = P::F::VALUE / P::F_Reg::VALUE;
    let thread_n = (UNIT_POS as usize) % threads_n;
    let offset_n = thread_n * P::F_Reg::VALUE;

    let mut bias_reg = P::f_reg();
    plane::load_rv_direct(&inputs.bias, &mut bias_reg, base_bias + offset_n);

    // Load layer norm params into register vectors
    // Each thread loads the full ln params for use in row broadcasts
    let base_ln = index_2d(&inputs.ln_weight, 0, head_idx);
    let mut ln_weight_rv = Rv::<P::E, P::F>::new();
    let mut ln_bias_rv = Rv::<P::E, P::F>::new();
    plane::load_rv_direct(&inputs.ln_weight, &mut ln_weight_rv, base_ln);
    plane::load_rv_direct(&inputs.ln_bias, &mut ln_bias_rv, base_ln);

    // === Step 1: z1 = xk @ W + b ===
    // k_smem is [F, CS], weight_smem is [F, F]
    // mma_AtB: k^T @ W = [CS, F] @ [F, F] = [CS, F]
    let mut z1_reg = P::cs_f_reg();
    z1_reg.zero();
    plane::mma_AtB(&mut z1_reg, &k_smem, &weight_smem);

    // Add bias (broadcast along rows)
    z1_reg.add_row(&bias_reg);

    plane::store_rt_to_st(&z1_reg, &mut z1_smem);

    sync_planes();

    // === Step 2: reconstruction_target = xv - xk ===
    // Both are [CS, F], result stored in v_direct_smem
    v_direct_smem.sub(&k_direct_smem);
    // v_direct_smem now contains reconstruction_target

    sync_planes();

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

    sync_planes();

    // === Step 4: eta_matrix = outer(token_eta, ttt_lr_eta).tril() ===
    // eta[i,j] = token_eta[i] * ttt_lr_eta[j] for j <= i, else 0

    let mut eta_reg = P::cs_cs_reg();
    eta_reg.zero();

    // CS×CS tiles have (CS/CS_Reg)² sub-tiles, which may be fewer than num_threads
    // Only threads with valid sub-tile indices participate
    let tiles_per_row_eta = P::CS::VALUE / P::CS_Reg::VALUE;
    let num_cs_cs_tiles = tiles_per_row_eta * tiles_per_row_eta;
    let participates_in_cs_cs = (UNIT_POS as usize) < num_cs_cs_tiles;

    // Compute thread's sub-tile position for CS×CS tile
    let tile_row_eta = (UNIT_POS as usize) / tiles_per_row_eta;
    let tile_col_eta = (UNIT_POS as usize) % tiles_per_row_eta;

    // Load eta vectors with offsets for this thread's sub-tile
    // token_eta[i] broadcasts along cols, so load token_eta[tile_row_eta * CS_Reg : ...]
    // ttt_lr_eta[j] broadcasts along rows, so load ttt_lr_eta[tile_col_eta * CS_Reg : ...]
    let mut token_eta_reg = P::cs_reg();
    let mut ttt_lr_eta_reg = P::cs_reg();
    if participates_in_cs_cs {
        let token_eta_offset = tile_row_eta * P::CS_Reg::VALUE;
        let ttt_lr_eta_offset = base_ttt_lr + tile_col_eta * P::CS_Reg::VALUE;
        plane::load_rv_direct(&inputs.token_eta, &mut token_eta_reg, token_eta_offset);
        plane::load_rv_direct(&inputs.ttt_lr_eta, &mut ttt_lr_eta_reg, ttt_lr_eta_offset);

        // eta[i,j] = token_eta[i] * ttt_lr_eta[j]
        // token_eta broadcasts along cols: eta[i,j] = token_eta[i]
        eta_reg.add_col(&token_eta_reg);
        // Multiply by ttt_lr_eta along rows: eta[i,j] *= ttt_lr_eta[j]
        eta_reg.mul_row(&ttt_lr_eta_reg);

        plane::store_rt_to_st(&eta_reg, &mut eta_matrix_smem);
    }

    sync_planes();

    // Apply lower triangular mask
    eta_matrix_smem.tril();

    sync_planes();

    // === Step 5: attn_scores = xq @ xk^T, attn1 = attn_scores.tril() ===
    // q_smem [F, CS], k_smem [F, CS]
    // mma_AtB: q^T @ k = [CS, F] @ [F, CS] = [CS, CS]
    let mut attn_reg = P::cs_cs_reg();
    attn_reg.zero();
    plane::mma_AtB(&mut attn_reg, &q_smem, &k_smem);

    if participates_in_cs_cs {
        plane::store_rt_to_st(&attn_reg, &mut attn_smem);
    }

    sync_planes();

    // Apply lower triangular mask
    attn_smem.tril();

    sync_planes();

    // === Step 6 & 7: Compute z1_bar ===
    // b1_bar = bias - eta_matrix @ grad_l_wrt_z1      [CS, F]
    // z1_bar = xq @ W - (eta_matrix * attn1) @ grad + b1_bar

    // Recompute eta in original lower triangular form
    // eta[i,j] = token_eta[i] * ttt_lr_eta[j] for j <= i
    if participates_in_cs_cs {
        eta_reg.zero();
        eta_reg.add_col(&token_eta_reg);  // eta[i,j] = token_eta[i]
        eta_reg.mul_row(&ttt_lr_eta_reg); // eta[i,j] *= ttt_lr_eta[j]
        plane::store_rt_to_st(&eta_reg, &mut eta_matrix_smem);
    }

    sync_planes();

    eta_matrix_smem.tril();

    sync_planes();

    // Compute eta @ grad using mma_AB (eta is [CS, CS], grad is [CS, F])
    let mut eta_grad_reg = P::cs_f_reg();
    eta_grad_reg.zero();
    plane::mma_AB(&mut eta_grad_reg, &eta_matrix_smem, &z1_smem);

    // Compute eta * attn (elementwise, both lower triangular)
    eta_matrix_smem.mul(&attn_smem);

    sync_planes();

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

    sync_planes();

    // === Step 8: output = xq + layer_norm(z1_bar) ===
    layer_norm_forward::<P::E, P::CS, P::F>(
        &mut temp_cs_f_smem,
        &ln_weight_rv,
        &ln_bias_rv,
        epsilon,
    );
    // temp_cs_f_smem now contains z1_bar_normalized

    sync_planes();

    // Add xq (need to load it in [CS, F] layout)
    // We can reuse k_direct_smem since we're done with it
    plane::load_st_direct(&inputs.xq, &mut k_direct_smem, base_qkv, 0, 0);

    sync_planes();

    // output = xq + z1_bar_normalized
    temp_cs_f_smem.add(&k_direct_smem);

    sync_planes();

    // Store output
    let base_output = index_2d(&outputs.output, batch_idx, head_idx);
    plane::store_st_direct(&temp_cs_f_smem, &mut outputs.output, base_output, 0, 0);

    // === Step 9 & 10: Weight and bias state updates ===
    // weight_out = weight - (last_eta_col * xk).T @ grad
    // bias_out = bias - sum_rows(last_eta_col * grad)

    // Extract last row of eta_matrix (before it was multiplied by attn)
    // We need to recompute eta since we modified eta_matrix_smem
    if participates_in_cs_cs {
        eta_reg.zero();
        eta_reg.add_col(&token_eta_reg);
        eta_reg.mul_row(&ttt_lr_eta_reg);
        plane::store_rt_to_st(&eta_reg, &mut eta_matrix_smem);
    }

    sync_planes();

    eta_matrix_smem.tril();

    sync_planes();

    // Extract last row of eta as a column vector
    let last_eta_col: Rv<P::E, P::CS> = extract_last_row(&eta_matrix_smem);

    // Reload k_direct for weight update (we overwrote it with xq earlier)
    plane::load_st_direct(&inputs.xk, &mut k_direct_smem, base_qkv, 0, 0);

    sync_planes();

    // Scale k by last_eta_col: k_direct[i, :] *= last_eta_col[i]
    k_direct_smem.mul_col(&last_eta_col);

    sync_planes();

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

    sync_planes();

    let base_weight_out = index_2d(&outputs.weight_out, batch_idx, head_idx);
    plane::store_st_direct(&weight_smem, &mut outputs.weight_out, base_weight_out, 0, 0);

    // Bias update: bias_out = bias - sum_rows(last_eta_col * grad)
    // Scale grad by last_eta_col: temp[i, :] = grad[i, :] * last_eta_col[i]
    temp_cs_f_smem.copy_from(&z1_smem);
    temp_cs_f_smem.mul_col(&last_eta_col);

    sync_planes();

    // Sum rows to get bias update: [F]
    let bias_update: Rv<P::E, P::F> = plane::sum_st_cols(&temp_cs_f_smem);

    // Compute bias_out = bias - bias_update
    // bias_update is Rv<E, F> from sum_st_cols (all threads have same value)
    // Load original bias into register vector, subtract, and store

    let mut bias_out_rv = Rv::<P::E, P::F>::new();
    plane::load_rv_direct(&inputs.bias, &mut bias_out_rv, base_bias);

    // Subtract bias_update (elementwise on Rv)
    bias_out_rv.sub(&bias_update);

    // Store updated bias
    let base_bias_out = index_2d(&outputs.bias_out, batch_idx, head_idx);
    plane::store_rv_direct(&bias_out_rv, &mut outputs.bias_out, base_bias_out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use thundercube::test_kernel;
    use thundercube::test_utils::TestFloat;

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

/// Integration tests comparing tile kernel against CPU reference implementation.
#[cfg(test)]
mod integration_tests {
    use crate::ttt::{
        CpuBackend,
        layer::{Qkv, TTTInnerModel, TTTInputsInner},
        linear::{TTTLinear, TTTLinearConfig},
    };
    use burn::tensor::{Tensor, TensorData};
    use burn_backend::Backend;
    use std::sync::Arc;

    fn assert_data_close(a: &[f32], b: &[f32], rtol: f32, atol: f32, name: &str) {
        assert_eq!(a.len(), b.len(), "{name}: Data sizes don't match");

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

    /// Reference implementation of the full TTT-Linear forward pass using pure Rust.
    /// This matches the algorithm implemented in the fused_ttt_forward_tile_kernel.
    fn ref_ttt_linear_forward(
        xq: &[f32],
        xk: &[f32],
        xv: &[f32],
        weight: &[f32],
        bias: &[f32],
        token_eta: &[f32],
        ttt_lr_eta: &[f32],
        ln_weight: &[f32],
        ln_bias: &[f32],
        output: &mut [f32],
        weight_out: &mut [f32],
        bias_out: &mut [f32],
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        head_dim: usize,
        epsilon: f64,
    ) {
        let bh_size = seq_len * head_dim;
        let ff_size = head_dim * head_dim;

        for b in 0..batch_size {
            for h in 0..num_heads {
                let bh_offset = (b * num_heads + h) * bh_size;
                let w_offset = (b * num_heads + h) * ff_size;
                let bias_offset = (b * num_heads + h) * head_dim;
                let eta_offset = (b * num_heads + h) * seq_len;
                let ln_offset = h * head_dim;

                // Local working copies
                let mut w_local: Vec<f64> = weight[w_offset..w_offset + ff_size]
                    .iter()
                    .map(|&x| x as f64)
                    .collect();
                let mut b_local: Vec<f64> = bias[bias_offset..bias_offset + head_dim]
                    .iter()
                    .map(|&x| x as f64)
                    .collect();

                // Step 1: z1 = xk @ W + b
                let mut z1 = vec![0.0f64; seq_len * head_dim];
                for s in 0..seq_len {
                    for d in 0..head_dim {
                        let mut sum = 0.0f64;
                        for k in 0..head_dim {
                            let xk_val = xk[bh_offset + s * head_dim + k] as f64;
                            let w_val = w_local[k * head_dim + d];
                            sum += xk_val * w_val;
                        }
                        z1[s * head_dim + d] = sum + b_local[d];
                    }
                }

                // Step 2: reconstruction_target = xv - xk
                let mut target = vec![0.0f64; seq_len * head_dim];
                for i in 0..bh_size {
                    target[i] = xv[bh_offset + i] as f64 - xk[bh_offset + i] as f64;
                }

                // Step 3: Layer norm + L2 grad (grad_l_wrt_z1)
                let mut grad = vec![0.0f64; seq_len * head_dim];
                for s in 0..seq_len {
                    // Compute mean and std for this row
                    let mut mean = 0.0f64;
                    for d in 0..head_dim {
                        mean += z1[s * head_dim + d];
                    }
                    mean /= head_dim as f64;

                    let mut var = 0.0f64;
                    for d in 0..head_dim {
                        let diff = z1[s * head_dim + d] - mean;
                        var += diff * diff;
                    }
                    var /= head_dim as f64;
                    let std = (var + epsilon).sqrt();

                    // Normalize
                    let mut norm = vec![0.0f64; head_dim];
                    for d in 0..head_dim {
                        norm[d] = (z1[s * head_dim + d] - mean) / std;
                    }

                    // Layer norm output
                    let mut ln_out = vec![0.0f64; head_dim];
                    for d in 0..head_dim {
                        ln_out[d] = ln_weight[ln_offset + d] as f64 * norm[d]
                            + ln_bias[ln_offset + d] as f64;
                    }

                    // L2 gradient through layer norm
                    let mut dl_dnorm = vec![0.0f64; head_dim];
                    for d in 0..head_dim {
                        let dl_dout = ln_out[d] - target[s * head_dim + d];
                        dl_dnorm[d] = dl_dout * ln_weight[ln_offset + d] as f64;
                    }

                    let mut sum_dl_dnorm = 0.0f64;
                    let mut sum_dl_dnorm_norm = 0.0f64;
                    for d in 0..head_dim {
                        sum_dl_dnorm += dl_dnorm[d];
                        sum_dl_dnorm_norm += dl_dnorm[d] * norm[d];
                    }

                    let n = head_dim as f64;
                    for d in 0..head_dim {
                        grad[s * head_dim + d] =
                            (dl_dnorm[d] * n - sum_dl_dnorm - norm[d] * sum_dl_dnorm_norm)
                                / (std * n);
                    }
                }

                // Step 4: eta_matrix = outer(token_eta, ttt_lr_eta).tril()
                // eta[i,j] = token_eta[i] * ttt_lr_eta[j] for j <= i
                let mut eta = vec![0.0f64; seq_len * seq_len];
                for i in 0..seq_len {
                    for j in 0..=i {
                        eta[i * seq_len + j] =
                            token_eta[i] as f64 * ttt_lr_eta[eta_offset + j] as f64;
                    }
                }

                // Step 5: attn1 = tril(xq @ xk^T)
                let mut attn = vec![0.0f64; seq_len * seq_len];
                for i in 0..seq_len {
                    for j in 0..=i {
                        let mut sum = 0.0f64;
                        for k in 0..head_dim {
                            sum += xq[bh_offset + i * head_dim + k] as f64
                                * xk[bh_offset + j * head_dim + k] as f64;
                        }
                        attn[i * seq_len + j] = sum;
                    }
                }

                // Step 6 & 7: z1_bar = xq @ W - (eta * attn) @ grad + b - eta @ grad
                let mut z1_bar = vec![0.0f64; seq_len * head_dim];
                for i in 0..seq_len {
                    for d in 0..head_dim {
                        // xq @ W
                        let mut xq_w = 0.0f64;
                        for k in 0..head_dim {
                            xq_w += xq[bh_offset + i * head_dim + k] as f64 * w_local[k * head_dim + d];
                        }

                        // eta @ grad (for b1_bar)
                        let mut eta_grad = 0.0f64;
                        for j in 0..=i {
                            eta_grad += eta[i * seq_len + j] * grad[j * head_dim + d];
                        }

                        // (eta * attn) @ grad
                        let mut eta_attn_grad = 0.0f64;
                        for j in 0..=i {
                            eta_attn_grad +=
                                eta[i * seq_len + j] * attn[i * seq_len + j] * grad[j * head_dim + d];
                        }

                        z1_bar[i * head_dim + d] =
                            xq_w - eta_attn_grad + b_local[d] - eta_grad;
                    }
                }

                // Step 8: output = xq + layer_norm(z1_bar)
                for s in 0..seq_len {
                    let mut mean = 0.0f64;
                    for d in 0..head_dim {
                        mean += z1_bar[s * head_dim + d];
                    }
                    mean /= head_dim as f64;

                    let mut var = 0.0f64;
                    for d in 0..head_dim {
                        let diff = z1_bar[s * head_dim + d] - mean;
                        var += diff * diff;
                    }
                    var /= head_dim as f64;
                    let std = (var + epsilon).sqrt();

                    for d in 0..head_dim {
                        let norm = (z1_bar[s * head_dim + d] - mean) / std;
                        let ln_out =
                            ln_weight[ln_offset + d] as f64 * norm + ln_bias[ln_offset + d] as f64;
                        output[bh_offset + s * head_dim + d] =
                            (xq[bh_offset + s * head_dim + d] as f64 + ln_out) as f32;
                    }
                }

                // Step 9 & 10: Weight and bias updates
                // weight_out = weight - (last_eta_col * xk)^T @ grad
                // bias_out = bias - sum_cols(last_eta_col * grad)

                let last_i = seq_len - 1;
                for row in 0..head_dim {
                    for col in 0..head_dim {
                        let mut update = 0.0f64;
                        for k in 0..seq_len {
                            let eta_k = eta[last_i * seq_len + k];
                            let xk_kr = xk[bh_offset + k * head_dim + row] as f64;
                            let grad_kc = grad[k * head_dim + col];
                            update += eta_k * xk_kr * grad_kc;
                        }
                        w_local[row * head_dim + col] -= update;
                    }
                }

                for d in 0..head_dim {
                    let mut update = 0.0f64;
                    for k in 0..seq_len {
                        let eta_k = eta[last_i * seq_len + k];
                        update += eta_k * grad[k * head_dim + d];
                    }
                    b_local[d] -= update;
                }

                // Copy outputs
                for i in 0..ff_size {
                    weight_out[w_offset + i] = w_local[i] as f32;
                }
                for i in 0..head_dim {
                    bias_out[bias_offset + i] = b_local[i] as f32;
                }
            }
        }
    }

    #[test]
    fn test_tile_kernel_vs_reference() {
        let batch_size = 2usize;
        let num_heads = 2usize;
        let head_dim = 64usize;
        let seq_len = 16usize;
        let hidden_size = num_heads * head_dim;
        let epsilon = 1e-6f32;

        let config = Arc::new(crate::ttt::TTTConfig {
            num_heads,
            hidden_size,
            token_size: hidden_size,
            mini_batch_size: seq_len,
            base_lr: 1.0,
            epsilon: f64::from(epsilon),
            ..crate::ttt::TTTConfig::new(crate::ttt::TEST_VOCAB_SIZE)
        });
        let linear_config = Arc::new(TTTLinearConfig::new());

        let cpu_device = Default::default();

        // Create random input tensors
        let xq_cpu: Tensor<CpuBackend, 4> = Tensor::random(
            [batch_size, num_heads, seq_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &cpu_device,
        );
        let xk_cpu: Tensor<CpuBackend, 4> = Tensor::random(
            [batch_size, num_heads, seq_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &cpu_device,
        );
        let xv_cpu: Tensor<CpuBackend, 4> = Tensor::random(
            [batch_size, num_heads, seq_len, head_dim],
            burn::tensor::Distribution::Normal(0.0, 0.1),
            &cpu_device,
        );
        let token_eta_cpu: Tensor<CpuBackend, 1> =
            Tensor::arange(1..(seq_len as i64 + 1), &cpu_device)
                .float()
                .recip();
        let ttt_lr_eta_cpu: Tensor<CpuBackend, 3> = Tensor::random(
            [batch_size, num_heads, seq_len],
            burn::tensor::Distribution::Uniform(0.01, 0.05),
            &cpu_device,
        );

        // Get data as vectors
        let xq_data: Vec<f32> = xq_cpu.to_data().to_vec().unwrap();
        let xk_data: Vec<f32> = xk_cpu.to_data().to_vec().unwrap();
        let xv_data: Vec<f32> = xv_cpu.to_data().to_vec().unwrap();
        let token_eta_data: Vec<f32> = token_eta_cpu.to_data().to_vec().unwrap();
        let ttt_lr_eta_data: Vec<f32> = ttt_lr_eta_cpu.to_data().to_vec().unwrap();

        // Create TTTLinear for weight/bias/ln initialization
        let ttt_linear_cpu: TTTLinear<CpuBackend> =
            TTTLinear::new(&config, &linear_config, &cpu_device);
        let mut state_cpu = ttt_linear_cpu.init_state(batch_size);

        let weight_init_data: Vec<f32> =
            ttt_linear_cpu.weight_init.val().to_data().to_vec().unwrap();
        let bias_init_data: Vec<f32> = ttt_linear_cpu.bias_init.val().to_data().to_vec().unwrap();
        let ln_weight_data: Vec<f32> = ttt_linear_cpu
            .layer_norm
            .weight
            .val()
            .to_data()
            .to_vec()
            .unwrap();
        let ln_bias_data: Vec<f32> = ttt_linear_cpu
            .layer_norm
            .bias
            .val()
            .to_data()
            .to_vec()
            .unwrap();

        // Expand weight/bias for batch dimension
        let weight_expanded: Vec<f32> = (0..batch_size)
            .flat_map(|_| weight_init_data.iter().copied())
            .collect();
        let bias_expanded: Vec<f32> = (0..batch_size)
            .flat_map(|_| bias_init_data.iter().copied())
            .collect();

        // Run CPU TTTLinear reference
        let inputs_cpu = TTTInputsInner {
            qkv: Qkv {
                xq: xq_cpu.clone(),
                xk: xk_cpu.clone(),
                xv: xv_cpu.clone(),
            },
            token_eta: token_eta_cpu.clone(),
            ttt_lr_eta: ttt_lr_eta_cpu.clone(),
            start_idx: 0,
        };

        let output_ttt_linear = ttt_linear_cpu.forward_mini_batch(&mut state_cpu, inputs_cpu);
        let output_ttt_linear_data: Vec<f32> = output_ttt_linear.to_data().to_vec().unwrap();

        // Run our reference implementation
        let mut output_ref = vec![0.0f32; batch_size * num_heads * seq_len * head_dim];
        let mut weight_out_ref = vec![0.0f32; batch_size * num_heads * head_dim * head_dim];
        let mut bias_out_ref = vec![0.0f32; batch_size * num_heads * head_dim];

        ref_ttt_linear_forward(
            &xq_data,
            &xk_data,
            &xv_data,
            &weight_expanded,
            &bias_expanded,
            &token_eta_data,
            &ttt_lr_eta_data,
            &ln_weight_data,
            &ln_bias_data,
            &mut output_ref,
            &mut weight_out_ref,
            &mut bias_out_ref,
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            epsilon as f64,
        );

        // Compare reference implementation against TTTLinear output
        assert_data_close(
            &output_ref,
            &output_ttt_linear_data,
            1e-3, // rtol
            1e-4, // atol
            "tile reference vs TTTLinear output",
        );

        // Compare weight updates
        let weight_ttt_linear: Vec<f32> = state_cpu.weight.to_data().to_vec().unwrap();
        assert_data_close(
            &weight_out_ref,
            &weight_ttt_linear,
            1e-3,
            1e-4,
            "tile reference vs TTTLinear weight update",
        );

        // Compare bias updates
        let bias_ttt_linear: Vec<f32> = state_cpu.bias.to_data().to_vec().unwrap();
        assert_data_close(
            &bias_out_ref,
            &bias_ttt_linear,
            1e-3,
            1e-4,
            "tile reference vs TTTLinear bias update",
        );
    }
}

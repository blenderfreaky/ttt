//! Layer normalization kernels and their backward passes.
//!
//! This module contains:
//! - `layer_norm_forward`: Basic layer norm forward pass
//! - `layer_norm_l2_grad`: Layer norm forward + L2 gradient backward (fused)
//! - `layer_norm_backward`: Standard layer norm backward pass
//! - `layer_norm_l2_grad_backward`: Second derivative through LN+L2 (backward-backward)

#![allow(non_camel_case_types, non_snake_case)]

use cubecl::prelude::*;
use thundercube::{
    cube::ReduceBuf,
    impl_reduction_ops,
    prelude::*,
    reduction_ops::{ReductionOp, SumOp},
};

// Sum of squares reduction operation for variance computation.
// Results in SumSqOp struct (macro adds Op suffix).
impl_reduction_ops! {
    SumSq<F> {
        identity => Line::<F>::empty(LINE_SIZE).fill(F::new(0.0));
        combine(a, b) => a + b * b;
        finalize(line) => line[0] + line[1] + line[2] + line[3];
        plane_reduce(val) => plane_sum(val);
        plane_combine(a, b) => a + b;  // Merge partials by addition, not squaring
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
    buf: &mut ReduceBuf<F>,
    #[comptime] epsilon: f32,
) {
    let c_inv = F::cast_from(1.0f32 / (C::VALUE as f32));

    // Step 1: mean = sum_rows(x) / C
    let mut mean = Rv::<F, R>::new();
    cube::sum_st_rows_cube(x, &mut mean, buf);
    mean.mul_scalar(c_inv);

    // Step 2: x -= mean (col broadcast)
    x.sub_col(&mean);

    // Sync after modifying shared memory before reading again
    sync_cube();

    // Step 3: var = sum_rows(x^2) / C using SumSqOp
    let mut std = Rv::<F, R>::new();
    cube::reduce_st_rows_cube::<F, R, C, SumSqOp>(x, &mut std, buf);
    std.mul_scalar(c_inv);

    // Step 4: std = sqrt(var + epsilon)
    std.add_scalar(F::cast_from(epsilon));
    std.sqrt();

    // Step 5: x /= std -> x now contains norm
    x.div_col(&std);

    // Sync after modifying shared memory before reading again
    sync_cube();

    // Step 6: x = weight * x + bias
    x.mul_row(ln_weight);
    x.add_row(ln_bias);
}

/// Computes layer norm forward pass and returns intermediate values needed for backward.
///
/// Returns: (x_hat, std) where x_hat = normalized input (before affine), std = standard deviation
///
/// # Arguments
/// * `x` - Input tile [R, C], will be modified to contain x_hat (normalized, pre-affine)
/// * `ln_weight` - Layer norm weight vector [C]
/// * `ln_bias` - Layer norm bias vector [C]
/// * `output` - Output tile [R, C], will contain the final layer norm output
/// * `std_out` - Output vector [R], will contain std per row
/// * `epsilon` - Small constant for numerical stability
#[cube]
pub fn layer_norm_forward_with_intermediates<F: Float, R: Dim, C: Dim>(
    x: &mut St<F, R, C>,
    ln_weight: &Rv<F, C>,
    ln_bias: &Rv<F, C>,
    output: &mut St<F, R, C>,
    std_out: &mut Rv<F, R>,
    buf: &mut ReduceBuf<F>,
    #[comptime] epsilon: f32,
) {
    let c_inv = F::cast_from(1.0f32 / (C::VALUE as f32));

    // Step 1: mean = sum_rows(x) / C
    let mut mean = Rv::<F, R>::new();
    cube::sum_st_rows_cube(x, &mut mean, buf);
    mean.mul_scalar(c_inv);

    // Step 2: x -= mean (col broadcast)
    x.sub_col(&mean);

    sync_cube();

    // Step 3: var = sum_rows(x^2) / C
    cube::reduce_st_rows_cube::<F, R, C, SumSqOp>(x, std_out, buf);
    std_out.mul_scalar(c_inv);

    // Step 4: std = sqrt(var + epsilon)
    std_out.add_scalar(F::cast_from(epsilon));
    std_out.sqrt();

    // Step 5: x /= std -> x now contains x_hat (normalized)
    x.div_col(std_out);

    sync_cube();

    // Step 6: output = weight * x_hat + bias
    output.copy_from(x);
    output.mul_row(ln_weight);
    output.add_row(ln_bias);
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
    buf: &mut ReduceBuf<F>,
    #[comptime] epsilon: f32,
) {
    let c_inv = F::cast_from(1.0f32 / (C::VALUE as f32));
    let c_f = F::cast_from(C::VALUE as f32);

    // Step 1: mean = sum_rows(x) / C
    let mut mean = Rv::<F, R>::new();
    cube::sum_st_rows_cube(x, &mut mean, buf);
    mean.mul_scalar(c_inv);

    // Step 2: x -= mean (col broadcast)
    x.sub_col(&mean);

    // Sync after modifying x before reading for variance
    sync_cube();

    // Step 3: var = sum_rows(x^2) / C using SumSqOp
    let mut std = Rv::<F, R>::new();
    cube::reduce_st_rows_cube::<F, R, C, SumSqOp>(x, &mut std, buf);
    std.mul_scalar(c_inv);

    // Step 4: std = sqrt(var + epsilon)
    std.add_scalar(F::cast_from(epsilon));
    std.sqrt();

    // Step 5: x /= std -> x now contains norm
    x.div_col(&std);

    // Sync after modifying x before copying to temp
    sync_cube();

    // Step 6: temp = x (copy norm), then temp = weight * temp + bias
    temp.copy_from(x);
    temp.mul_row(ln_weight);
    temp.add_row(ln_bias);

    // Step 7: temp -= target -> temp = dl_dout
    temp.sub(target);

    // Step 8: temp *= weight -> temp = dl_dnorm
    temp.mul_row(ln_weight);

    // Sync after modifying temp before reading for reduction
    sync_cube();

    // Step 9: Compute reduction terms
    let mut sum_dl_dnorm = Rv::<F, R>::new();
    cube::sum_st_rows_cube(temp, &mut sum_dl_dnorm, buf);

    // temp = dl_dnorm currently, we need sum(dl_dnorm * norm)
    // Multiply temp by x (norm), sum, then rebuild dl_dnorm
    temp.mul(x);

    // Sync after modifying temp before reading for reduction
    sync_cube();

    let mut sum_dl_dnorm_norm = Rv::<F, R>::new();
    cube::sum_st_rows_cube(temp, &mut sum_dl_dnorm_norm, buf);

    // Recompute dl_dnorm in temp (x still has norm)
    temp.copy_from(x);
    temp.mul_row(ln_weight);
    temp.add_row(ln_bias);
    temp.sub(target);
    temp.mul_row(ln_weight);

    // Sync after modifying temp before continuing
    sync_cube();

    // Step 10: dl_dx = (dl_dnorm * C - sum_dl_dnorm - norm * sum_dl_dnorm_norm) / (std * C)

    // temp = dl_dnorm * C
    temp.mul_scalar(c_f);

    // temp -= sum_dl_dnorm
    temp.sub_col(&sum_dl_dnorm);

    // x = norm * sum_dl_dnorm_norm (col broadcast), then temp -= x
    x.mul_col(&sum_dl_dnorm_norm);

    // Sync after modifying x before reading
    sync_cube();

    temp.sub(x);

    // temp /= (std * C)
    std.mul_scalar(c_f);
    temp.div_col(&std);

    // Sync after modifying temp before final copy
    sync_cube();

    // Copy result to x
    x.copy_from(temp);
}

/// Fused layer norm + L2 gradient computation that saves intermediates for backward.
///
/// Same as `layer_norm_l2_grad` but saves the intermediate values needed for
/// the backward pass (second derivative computation).
///
/// Saves:
/// - x_hat: normalized input (x - mean) / std [R, C]
/// - std: standard deviation per row [R]
/// - grad_output: y - target [R, C]
/// - grad_x_hat: grad_output * ln_weight [R, C]
///
/// The grad_l result is written to `x` as usual.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn layer_norm_l2_grad_save_intermediates<F: Float, R: Dim, C: Dim>(
    x: &mut St<F, R, C>,
    target: &St<F, R, C>,
    ln_weight: &Rv<F, C>,
    ln_bias: &Rv<F, C>,
    temp: &mut St<F, R, C>,
    buf: &mut ReduceBuf<F>,
    // Output intermediates
    x_hat_out: &mut St<F, R, C>,
    std_out: &mut Rv<F, R>,
    grad_output_out: &mut St<F, R, C>,
    grad_x_hat_out: &mut St<F, R, C>,
    #[comptime] epsilon: f32,
) {
    let c_inv = F::cast_from(1.0f32 / (C::VALUE as f32));
    let c_f = F::cast_from(C::VALUE as f32);

    // Step 1: mean = sum_rows(x) / C
    let mut mean = Rv::<F, R>::new();
    cube::sum_st_rows_cube(x, &mut mean, buf);
    mean.mul_scalar(c_inv);

    // Step 2: x -= mean (col broadcast)
    x.sub_col(&mean);

    sync_cube();

    // Step 3: var = sum_rows(x^2) / C using SumSqOp
    let mut std = Rv::<F, R>::new();
    cube::reduce_st_rows_cube::<F, R, C, SumSqOp>(x, &mut std, buf);
    std.mul_scalar(c_inv);

    // Step 4: std = sqrt(var + epsilon)
    std.add_scalar(F::cast_from(epsilon));
    std.sqrt();

    // Save std for backward
    std_out.set(&std);

    // Step 5: x /= std -> x now contains x_hat (normalized)
    x.div_col(&std);

    sync_cube();

    // Save x_hat for backward
    x_hat_out.copy_from(x);

    // Step 6: temp = x (copy norm), then temp = weight * temp + bias = y
    temp.copy_from(x);
    temp.mul_row(ln_weight);
    temp.add_row(ln_bias);

    // Step 7: temp -= target -> temp = grad_output = y - target
    temp.sub(target);

    sync_cube();

    // Save grad_output for backward
    grad_output_out.copy_from(temp);

    // Step 8: temp *= weight -> temp = grad_x_hat = grad_output * ln_weight
    temp.mul_row(ln_weight);

    sync_cube();

    // Save grad_x_hat for backward
    grad_x_hat_out.copy_from(temp);

    // Step 9: Compute reduction terms for grad_l
    let mut sum_dl_dnorm = Rv::<F, R>::new();
    cube::sum_st_rows_cube(temp, &mut sum_dl_dnorm, buf);

    // temp = dl_dnorm currently, we need sum(dl_dnorm * norm)
    temp.mul(x);

    sync_cube();

    let mut sum_dl_dnorm_norm = Rv::<F, R>::new();
    cube::sum_st_rows_cube(temp, &mut sum_dl_dnorm_norm, buf);

    // Recompute dl_dnorm in temp (x still has x_hat)
    temp.copy_from(x);
    temp.mul_row(ln_weight);
    temp.add_row(ln_bias);
    temp.sub(target);
    temp.mul_row(ln_weight);

    sync_cube();

    // Step 10: dl_dx = (dl_dnorm * C - sum_dl_dnorm - norm * sum_dl_dnorm_norm) / (std * C)
    temp.mul_scalar(c_f);
    temp.sub_col(&sum_dl_dnorm);

    // x = norm * sum_dl_dnorm_norm (col broadcast), then temp -= x
    x.mul_col(&sum_dl_dnorm_norm);

    sync_cube();

    temp.sub(x);

    // temp /= (std * C)
    std.mul_scalar(c_f);
    temp.div_col(&std);

    sync_cube();

    // Copy result to x (grad_l_wrt_Z1)
    x.copy_from(temp);
}

/// Computes layer norm forward and saves x_hat and std for backward.
///
/// Same as `layer_norm_forward` but saves:
/// - x_hat: normalized input before weight/bias [R, C]
/// - std: standard deviation per row [R]
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn layer_norm_forward_save_intermediates<F: Float, R: Dim, C: Dim>(
    x: &mut St<F, R, C>,
    ln_weight: &Rv<F, C>,
    ln_bias: &Rv<F, C>,
    buf: &mut ReduceBuf<F>,
    // Output intermediates
    x_hat_out: &mut St<F, R, C>,
    std_out: &mut Rv<F, R>,
    #[comptime] epsilon: f32,
) {
    let c_inv = F::cast_from(1.0f32 / (C::VALUE as f32));

    // Step 1: mean = sum_rows(x) / C
    let mut mean = Rv::<F, R>::new();
    cube::sum_st_rows_cube(x, &mut mean, buf);
    mean.mul_scalar(c_inv);

    // Step 2: x -= mean
    x.sub_col(&mean);

    sync_cube();

    // Step 3: var = sum_rows(x^2) / C
    let mut std = Rv::<F, R>::new();
    cube::reduce_st_rows_cube::<F, R, C, SumSqOp>(x, &mut std, buf);
    std.mul_scalar(c_inv);

    // Step 4: std = sqrt(var + epsilon)
    std.add_scalar(F::cast_from(epsilon));
    std.sqrt();

    // Save std for backward
    std_out.set(&std);

    // Step 5: x /= std -> x_hat
    x.div_col(&std);

    sync_cube();

    // Save x_hat for backward
    x_hat_out.copy_from(x);

    // Step 6: x = weight * x + bias
    x.mul_row(ln_weight);
    x.add_row(ln_bias);
}

/// Standard layer norm backward pass with temp storage.
/// Computes gradients w.r.t. input, weight, and bias.
///
/// Given:
/// - grad_output: upstream gradient [R, C]
/// - x_hat: normalized input (x - mean) / std [R, C]
/// - std: standard deviation per row [R]
/// - ln_weight: layer norm weight [C]
///
/// Computes:
/// - grad_x: gradient w.r.t. input [R, C]
/// - grad_ln_weight: gradient w.r.t. weight [C]
/// - grad_ln_bias: gradient w.r.t. bias [C]
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn layer_norm_backward<F: Float, R: Dim, C: Dim>(
    grad_output: &St<F, R, C>,
    x_hat: &St<F, R, C>,
    std: &Rv<F, R>,
    ln_weight: &Rv<F, C>,
    temp: &mut St<F, R, C>,
    grad_x: &mut St<F, R, C>,
    grad_ln_weight: &mut Rv<F, C>,
    grad_ln_bias: &mut Rv<F, C>,
    buf: &mut ReduceBuf<F>,
) {
    let c_f = F::cast_from(C::VALUE as f32);
    let c_inv = F::cast_from(1.0f32 / (C::VALUE as f32));

    // grad_ln_bias = sum_rows(grad_output)
    cube::reduce_st_cols_cube::<F, R, C, SumOp>(grad_output, grad_ln_bias, buf);

    // grad_ln_weight = sum_rows(grad_output * x_hat)
    temp.copy_from(grad_output);
    temp.mul(x_hat);

    sync_cube();

    cube::reduce_st_cols_cube::<F, R, C, SumOp>(temp, grad_ln_weight, buf);

    // grad_x_hat = grad_output * weight
    temp.copy_from(grad_output);
    temp.mul_row(ln_weight);

    sync_cube();

    // sum_grad_x_hat = sum_cols(grad_x_hat) per row
    let mut sum_grad_x_hat = Rv::<F, R>::new();
    cube::sum_st_rows_cube(temp, &mut sum_grad_x_hat, buf);

    // Compute sum(grad_x_hat * x_hat) per row
    grad_x.copy_from(temp);
    grad_x.mul(x_hat);

    sync_cube();

    let mut sum_grad_x_hat_x_hat = Rv::<F, R>::new();
    cube::sum_st_rows_cube(grad_x, &mut sum_grad_x_hat_x_hat, buf);

    // Now compute the final gradient:
    // grad_x = (grad_x_hat * C - sum_grad_x_hat - x_hat * sum_grad_x_hat_x_hat) / (std * C)

    // Start with grad_x_hat * C (temp still has grad_x_hat)
    grad_x.copy_from(temp);
    grad_x.mul_scalar(c_f);

    // Subtract sum_grad_x_hat (broadcast across columns)
    grad_x.sub_col(&sum_grad_x_hat);

    sync_cube();

    // Subtract x_hat * sum_grad_x_hat_x_hat
    // temp = x_hat * sum_grad_x_hat_x_hat
    temp.copy_from(x_hat);
    temp.mul_col(&sum_grad_x_hat_x_hat);

    sync_cube();

    grad_x.sub(temp);

    // Divide by (std * C) = divide by std, then divide by C
    grad_x.div_col(std);
    grad_x.mul_scalar(c_inv);
}

/// Backward through the fused layer norm + L2 gradient computation.
/// This is the second derivative (backward-backward) through the LN+L2 forward.
///
/// The forward computed:
///   x_hat = (Z1 - mean) / std
///   y = ln_weight * x_hat + ln_bias
///   grad_output = y - target
///   grad_x_hat = grad_output * ln_weight
///   grad_l = (grad_x_hat * F - sum(grad_x_hat) - x_hat * sum(grad_x_hat * x_hat)) / (std * F)
///
/// Given upstream gradient `grad_L_grad_l` (gradient w.r.t. grad_l),
/// this computes gradients w.r.t. Z1, target, ln_weight, ln_bias.
///
/// # Arguments
/// * `grad_L_grad_l` - Upstream gradient w.r.t. grad_l [R, C]
/// * `x_hat` - Saved x_hat from forward [R, C]
/// * `std` - Saved std from forward [R]
/// * `grad_output` - Saved (y - target) from forward [R, C]
/// * `grad_x_hat` - Saved grad_output * ln_weight from forward [R, C]
/// * `ln_weight` - Layer norm weight [C]
/// * `temp1`, `temp2` - Scratch tiles [R, C]
/// * `grad_L_Z1` - Output: gradient w.r.t. Z1 [R, C]
/// * `grad_L_target` - Output: gradient w.r.t. target [R, C]
/// * `grad_L_ln_weight` - Output: gradient w.r.t. ln_weight [C]
/// * `grad_L_ln_bias` - Output: gradient w.r.t. ln_bias [C]
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn layer_norm_l2_grad_backward<F: Float, R: Dim, C: Dim>(
    // Upstream gradient
    grad_L_grad_l: &St<F, R, C>,
    // Saved from forward
    x_hat: &St<F, R, C>,
    std: &Rv<F, R>,
    grad_output: &St<F, R, C>, // y - target from forward
    grad_x_hat: &St<F, R, C>,  // grad_output * ln_weight from forward
    ln_weight: &Rv<F, C>,
    // Temp storage
    temp1: &mut St<F, R, C>,
    temp2: &mut St<F, R, C>,
    // Outputs (shared memory)
    grad_L_Z1: &mut St<F, R, C>,
    grad_L_target: &mut St<F, R, C>,
    // Outputs (register vectors)
    grad_L_ln_weight: &mut Rv<F, C>,
    grad_L_ln_bias: &mut Rv<F, C>,
    buf: &mut ReduceBuf<F>,
) {
    let f_f = F::cast_from(C::VALUE as f32);
    let f_inv = F::cast_from(1.0f32 / (C::VALUE as f32));

    // From Triton reference:
    // grad_L_grad_x_hat = (1/std) * grad_L_grad_l
    //                   + (1/F) * sum(-grad_L_grad_l / std, axis=1)
    //                   + (1/F) * x_hat * sum(-grad_L_grad_l / std * x_hat, axis=1)

    // First compute -grad_L_grad_l / std
    temp1.copy_from(grad_L_grad_l);
    temp1.neg();
    temp1.div_col(std);

    sync_cube();

    // sum1 = sum(-grad_L_grad_l / std) per row
    let mut sum1 = Rv::<F, R>::new();
    cube::sum_st_rows_cube(temp1, &mut sum1, buf);

    // sum2 = sum(-grad_L_grad_l / std * x_hat) per row
    temp1.mul(x_hat);

    sync_cube();

    let mut sum2 = Rv::<F, R>::new();
    cube::sum_st_rows_cube(temp1, &mut sum2, buf);

    // grad_L_grad_x_hat = (1/std) * grad_L_grad_l + (1/F) * sum1 + (1/F) * x_hat * sum2
    // temp1 = (1/std) * grad_L_grad_l
    temp1.copy_from(grad_L_grad_l);
    temp1.div_col(std);

    // Add (1/F) * sum1 broadcast
    let mut scaled_sum1 = sum1;
    scaled_sum1.mul_scalar(f_inv);
    temp1.add_col(&scaled_sum1);

    sync_cube();

    // Add (1/F) * x_hat * sum2
    // temp2 = x_hat * sum2
    temp2.copy_from(x_hat);
    temp2.mul_col(&sum2);
    temp2.mul_scalar(f_inv);

    sync_cube();

    temp1.add(temp2);

    sync_cube();

    // Now temp1 = grad_L_grad_x_hat

    // grad_L_y = ln_weight * grad_L_grad_x_hat
    temp2.copy_from(temp1);
    temp2.mul_row(ln_weight);

    sync_cube();

    // grad_L_ln_weight = sum(grad_output * grad_L_grad_x_hat + grad_L_y * x_hat)
    // First term: grad_output * grad_L_grad_x_hat
    grad_L_Z1.copy_from(grad_output);
    grad_L_Z1.mul(temp1);

    sync_cube();

    // Second term: grad_L_y * x_hat = temp2 * x_hat
    grad_L_target.copy_from(temp2);
    grad_L_target.mul(x_hat);

    sync_cube();

    grad_L_Z1.add(grad_L_target);

    sync_cube();

    cube::reduce_st_cols_cube::<F, R, C, SumOp>(grad_L_Z1, grad_L_ln_weight, buf);

    // grad_L_ln_bias = sum(grad_L_y) = sum(temp2)
    cube::reduce_st_cols_cube::<F, R, C, SumOp>(temp2, grad_L_ln_bias, buf);

    // grad_L_x_hat = grad_L_y * ln_weight
    //              + (1/F) * grad_x_hat * sum(-grad_L_grad_l / std * x_hat)
    //              + (1/F) * sum(grad_x_hat * x_hat) * (-grad_L_grad_l / std)

    // Start fresh: temp1 = grad_L_y * ln_weight
    temp1.copy_from(temp2);
    temp1.mul_row(ln_weight);

    sync_cube();

    // Compute sum(grad_x_hat * x_hat) per row
    temp2.copy_from(grad_x_hat);
    temp2.mul(x_hat);

    sync_cube();

    let mut sum_gxh_xh = Rv::<F, R>::new();
    cube::sum_st_rows_cube(temp2, &mut sum_gxh_xh, buf);

    // Term 2: (1/F) * grad_x_hat * sum2 (sum2 = sum(-grad_L_grad_l / std * x_hat))
    temp2.copy_from(grad_x_hat);
    temp2.mul_col(&sum2);
    temp2.mul_scalar(f_inv);

    sync_cube();

    temp1.add(temp2);

    sync_cube();

    // Term 3: (1/F) * sum(grad_x_hat * x_hat) * (-grad_L_grad_l / std)
    temp2.copy_from(grad_L_grad_l);
    temp2.neg();
    temp2.div_col(std);
    temp2.mul_col(&sum_gxh_xh);
    temp2.mul_scalar(f_inv);

    sync_cube();

    temp1.add(temp2);

    sync_cube();

    // Now temp1 = grad_L_x_hat

    // We need to compute grad_l for the std gradient
    // grad_l = (grad_x_hat * F - sum(grad_x_hat) - x_hat * sum(grad_x_hat * x_hat)) / (std * F)

    // Compute sum(grad_x_hat) per row
    let mut sum_gxh = Rv::<F, R>::new();
    cube::sum_st_rows_cube(grad_x_hat, &mut sum_gxh, buf);

    // grad_l = (grad_x_hat * F - sum_gxh - x_hat * sum_gxh_xh) / (std * F)
    temp2.copy_from(grad_x_hat);
    temp2.mul_scalar(f_f);
    temp2.sub_col(&sum_gxh);

    sync_cube();

    // Subtract x_hat * sum_gxh_xh
    grad_L_Z1.copy_from(x_hat);
    grad_L_Z1.mul_col(&sum_gxh_xh);

    sync_cube();

    temp2.sub(grad_L_Z1);

    // Divide by std * F = divide by std, then divide by F
    temp2.div_col(std);
    temp2.mul_scalar(f_inv);

    sync_cube();

    // Now temp2 = grad_l (recomputed)

    // grad_L_std = -grad_L_x_hat * (x_hat / std) - grad_L_grad_l * (grad_l / std)
    //            = -(temp1 * x_hat + grad_L_grad_l * temp2) / std
    grad_L_Z1.copy_from(temp1);
    grad_L_Z1.mul(x_hat);

    sync_cube();

    grad_L_target.copy_from(grad_L_grad_l);
    grad_L_target.mul(temp2);

    sync_cube();

    grad_L_Z1.add(grad_L_target);
    grad_L_Z1.neg();
    grad_L_Z1.div_col(std);

    sync_cube();

    // Now grad_L_Z1 = grad_L_std

    // Compute sum(grad_L_std) per row
    let mut sum_grad_L_std = Rv::<F, R>::new();
    cube::sum_st_rows_cube(grad_L_Z1, &mut sum_grad_L_std, buf);

    // Final: grad_L_Z1 = grad_L_x_hat / std - (1/F) * sum(grad_L_x_hat) / std + (1/F) * sum(grad_L_std) * x_hat

    // sum(grad_L_x_hat) per row - temp1 still has grad_L_x_hat
    let mut sum_grad_L_x_hat = Rv::<F, R>::new();
    cube::sum_st_rows_cube(temp1, &mut sum_grad_L_x_hat, buf);

    // grad_L_Z1 = temp1 / std
    grad_L_Z1.copy_from(temp1);
    grad_L_Z1.div_col(std);

    // Subtract (1/F) * sum(grad_L_x_hat) / std
    let mut term2 = sum_grad_L_x_hat;
    term2.div(std);
    term2.mul_scalar(f_inv);
    grad_L_Z1.sub_col(&term2);

    sync_cube();

    // Add (1/F) * sum(grad_L_std) * x_hat
    temp2.copy_from(x_hat);
    let mut scaled_sum_std = sum_grad_L_std;
    scaled_sum_std.mul_scalar(f_inv);
    temp2.mul_col(&scaled_sum_std);

    sync_cube();

    grad_L_Z1.add(temp2);

    sync_cube();

    // grad_L_target = -ln_weight * grad_L_grad_x_hat
    // We need to recompute grad_L_grad_x_hat (was in temp1 earlier, but we overwrote it)

    // Recompute -grad_L_grad_l / std
    temp1.copy_from(grad_L_grad_l);
    temp1.neg();
    temp1.div_col(std);

    sync_cube();

    let mut sum1_recomputed = Rv::<F, R>::new();
    cube::sum_st_rows_cube(temp1, &mut sum1_recomputed, buf);

    temp1.mul(x_hat);

    sync_cube();

    let mut sum2_recomputed = Rv::<F, R>::new();
    cube::sum_st_rows_cube(temp1, &mut sum2_recomputed, buf);

    // grad_L_grad_x_hat = (1/std) * grad_L_grad_l + (1/F) * sum1 + (1/F) * x_hat * sum2
    temp1.copy_from(grad_L_grad_l);
    temp1.div_col(std);

    let mut s1 = sum1_recomputed;
    s1.mul_scalar(f_inv);
    temp1.add_col(&s1);

    sync_cube();

    temp2.copy_from(x_hat);
    temp2.mul_col(&sum2_recomputed);
    temp2.mul_scalar(f_inv);

    sync_cube();

    temp1.add(temp2);

    sync_cube();

    // grad_L_target = -ln_weight * grad_L_grad_x_hat
    grad_L_target.copy_from(temp1);
    grad_L_target.mul_row(ln_weight);
    grad_L_target.neg();
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::prelude::*;
    use test_case::test_matrix;
    use thundercube::{prelude::*, test_kernel, test_utils::TestFloat};

    const ROWS: usize = 8;
    const COLS: usize = 32;
    const EPSILON: f32 = 1e-5;

    // =========================================================================
    // Test kernels that wrap the layer norm functions
    // =========================================================================

    /// Test kernel for layer_norm_forward
    #[cube(launch)]
    fn test_ln_forward_kernel<F: Float>(
        input: &Tensor<Line<F>>,
        ln_weight: &Tensor<Line<F>>,
        ln_bias: &Tensor<Line<F>>,
        output: &mut Tensor<Line<F>>,
    ) {
        let mut x = St::<F, D8, D32>::new();
        let mut weight = Rv::<F, D32>::new();
        let mut bias = Rv::<F, D32>::new();
        let mut buf = ReduceBuf::<F>::new();

        // Load using library functions
        cube::load_st_direct(input, &mut x, 0, 0, 0);
        cube::broadcast::load_rv_direct(ln_weight, &mut weight, 0);
        cube::broadcast::load_rv_direct(ln_bias, &mut bias, 0);

        sync_cube();

        // Run layer norm forward
        layer_norm_forward::<F, D8, D32>(&mut x, &weight, &bias, &mut buf, EPSILON);

        sync_cube();

        // Store using library function
        cube::store_st_direct(&x, output, 0, 0, 0);
    }

    /// Test kernel for layer_norm_l2_grad
    #[cube(launch)]
    fn test_ln_l2_grad_kernel<F: Float>(
        input: &Tensor<Line<F>>,
        target: &Tensor<Line<F>>,
        ln_weight: &Tensor<Line<F>>,
        ln_bias: &Tensor<Line<F>>,
        output: &mut Tensor<Line<F>>,
    ) {
        let mut x = St::<F, D8, D32>::new();
        let mut tgt = St::<F, D8, D32>::new();
        let mut temp = St::<F, D8, D32>::new();
        let mut weight = Rv::<F, D32>::new();
        let mut bias = Rv::<F, D32>::new();
        let mut buf = ReduceBuf::<F>::new();

        // Load using library functions
        cube::load_st_direct(input, &mut x, 0, 0, 0);
        cube::load_st_direct(target, &mut tgt, 0, 0, 0);
        cube::broadcast::load_rv_direct(ln_weight, &mut weight, 0);
        cube::broadcast::load_rv_direct(ln_bias, &mut bias, 0);

        sync_cube();

        // Run layer norm + L2 grad
        layer_norm_l2_grad::<F, D8, D32>(
            &mut x, &tgt, &weight, &bias, &mut temp, &mut buf, EPSILON,
        );

        sync_cube();

        // Store using library function
        cube::store_st_direct(&x, output, 0, 0, 0);
    }

    /// Identity test kernel - just load and store to verify load/store works
    #[cube(launch)]
    fn test_identity_kernel<F: Float>(input: &Tensor<Line<F>>, output: &mut Tensor<Line<F>>) {
        let mut x = St::<F, D8, D32>::new();

        cube::load_st_direct(input, &mut x, 0, 0, 0);
        sync_cube();
        cube::store_st_direct(&x, output, 0, 0, 0);
    }

    /// Test kernel for just computing row means (diagnostic)
    #[cube(launch)]
    fn test_row_mean_kernel<F: Float>(input: &Tensor<Line<F>>, output: &mut Tensor<Line<F>>) {
        let mut x = St::<F, D8, D32>::new();
        let mut buf = ReduceBuf::<F>::new();

        cube::load_st_direct(input, &mut x, 0, 0, 0);
        sync_cube();

        // Compute row means: sum across columns, divide by C
        let mut mean = Rv::<F, D8>::new();
        cube::sum_st_rows_cube(&x, &mut mean, &mut buf);
        mean.mul_scalar(F::cast_from(1.0f32 / 32.0f32));

        // Output is just the row means (broadcast to full row width)
        // Write mean to first row of output
        if UNIT_POS == 0 {
            for i in 0..D8::LINES {
                let line_idx = i;
                output[line_idx] = mean.data[i];
            }
        }
    }

    /// Test kernel for center (subtract mean) - steps 1-4 of layer norm
    #[cube(launch)]
    fn test_center_kernel<F: Float>(input: &Tensor<Line<F>>, output: &mut Tensor<Line<F>>) {
        let mut x = St::<F, D8, D32>::new();
        let mut buf = ReduceBuf::<F>::new();

        cube::load_st_direct(input, &mut x, 0, 0, 0);
        sync_cube();

        // Step 1: mean = sum_rows(x) / C
        let mut mean = Rv::<F, D8>::new();
        cube::sum_st_rows_cube(&x, &mut mean, &mut buf);
        mean.mul_scalar(F::cast_from(1.0f32 / 32.0f32));

        // Step 2: x -= mean
        x.sub_col(&mean);

        sync_cube();

        cube::store_st_direct(&x, output, 0, 0, 0);
    }

    /// Test kernel for sum of squares reduction (diagnostic)
    #[cube(launch)]
    fn test_sum_sq_kernel<F: Float>(input: &Tensor<Line<F>>, output: &mut Tensor<Line<F>>) {
        let mut x = St::<F, D8, D32>::new();
        let mut buf = ReduceBuf::<F>::new();

        cube::load_st_direct(input, &mut x, 0, 0, 0);
        sync_cube();

        // Compute sum of squares per row
        let mut sum_sq = Rv::<F, D8>::new();
        cube::reduce_st_rows_cube::<F, D8, D32, SumSqOp>(&x, &mut sum_sq, &mut buf);

        // Output the sum of squares
        if UNIT_POS == 0 {
            for i in 0..D8::LINES {
                output[i] = sum_sq.data[i];
            }
        }
    }

    /// Test kernel for normalize (center + divide by std)
    #[cube(launch)]
    fn test_normalize_kernel<F: Float>(input: &Tensor<Line<F>>, output: &mut Tensor<Line<F>>) {
        let mut x = St::<F, D8, D32>::new();
        let mut buf = ReduceBuf::<F>::new();

        cube::load_st_direct(input, &mut x, 0, 0, 0);
        sync_cube();

        // Step 1: mean = sum_rows(x) / C
        let mut mean = Rv::<F, D8>::new();
        cube::sum_st_rows_cube(&x, &mut mean, &mut buf);
        mean.mul_scalar(F::cast_from(1.0f32 / 32.0f32));

        // Step 2: x -= mean
        x.sub_col(&mean);
        sync_cube();

        // Step 3: var = sum_rows(x^2) / C
        let mut std = Rv::<F, D8>::new();
        cube::reduce_st_rows_cube::<F, D8, D32, SumSqOp>(&x, &mut std, &mut buf);
        std.mul_scalar(F::cast_from(1.0f32 / 32.0f32));

        // Step 4: std = sqrt(var + epsilon)
        std.add_scalar(F::cast_from(EPSILON));
        std.sqrt();

        // Step 5: x /= std
        x.div_col(&std);
        sync_cube();

        cube::store_st_direct(&x, output, 0, 0, 0);
    }

    /// Diagnostic kernel to check thread indices
    #[cube(launch)]
    fn test_thread_indices<F: Float>(output: &mut Tensor<Line<F>>) {
        let tid = UNIT_POS;
        let cube_dim = CUBE_DIM;
        let plane_dim = PLANE_DIM;

        // Each thread writes its index to its position
        // output[0] = tid for thread 0, output[1] = tid for thread 1, etc.
        if (tid as usize) < 64 {
            let val = Line::<F>::empty(LINE_SIZE).fill(F::cast_from(tid));
            output[tid as usize] = val;
        }

        // Thread 0 also writes cube_dim and plane_dim to slots 64 and 65
        if tid == 0 {
            output[64] = Line::<F>::empty(LINE_SIZE).fill(F::cast_from(cube_dim));
            output[65] = Line::<F>::empty(LINE_SIZE).fill(F::cast_from(plane_dim));
        }
    }

    // =========================================================================
    // Tests using test_kernel! macro
    // =========================================================================

    test_kernel! {
        #[test_matrix([64])]
        fn test_thread_indices_check(threads: usize) for F in [f32] {
            let output: Tensor = [66 * LINE_SIZE] as Range;

            assert_eq!(
                test_thread_indices(output()) for (1, 1, 1) @ (threads),
                {
                    // Each thread should write its tid
                    for i in 0..64 {
                        for j in 0..LINE_SIZE {
                            output[i * LINE_SIZE + j] = F::from_f64(i as f64);
                        }
                    }
                    // Thread 0 writes cube_dim (64) and plane_dim (32 on AMD)
                    for j in 0..LINE_SIZE {
                        output[64 * LINE_SIZE + j] = F::from_f64(64.0);
                        output[65 * LINE_SIZE + j] = F::from_f64(32.0);
                    }
                }
            );
        }

        #[test_matrix([64])]
        fn test_identity(threads: usize) for F in [f32] {
            let input: Tensor = [ROWS, COLS];
            let output: Tensor = [ROWS, COLS] as Range;

            assert_eq!(
                test_identity_kernel(input(), output()) for (1, 1, 1) @ (threads),
                {
                    // Output should equal input
                    for i in 0..(ROWS * COLS) {
                        output[i] = input[i];
                    }
                }
            );
        }

        #[test_matrix([32, 64])]
        fn test_row_mean(threads: usize) for F in [f32] {
            let input: Tensor = [ROWS, COLS];
            let output: Tensor = [ROWS] as Range;

            assert_eq!(
                test_row_mean_kernel(input(), output()) for (1, 1, 1) @ (threads),
                {
                    // Compute row means
                    for r in 0..ROWS {
                        let mut sum = 0.0;
                        for c in 0..COLS {
                            sum += input[r * COLS + c].into_f64();
                        }
                        output[r] = F::from_f64(sum / COLS as f64);
                    }
                }
            );
        }

        #[test_matrix([32, 64])]
        fn test_center(threads: usize) for F in [f32] {
            let input: Tensor = [ROWS, COLS];
            let output: Tensor = [ROWS, COLS] as Range;

            assert_eq!(
                test_center_kernel(input(), output()) for (1, 1, 1) @ (threads),
                {
                    // Compute mean and subtract
                    for r in 0..ROWS {
                        let mut mean = 0.0;
                        for c in 0..COLS {
                            mean += input[r * COLS + c].into_f64();
                        }
                        mean /= COLS as f64;

                        for c in 0..COLS {
                            output[r * COLS + c] = F::from_f64(input[r * COLS + c].into_f64() - mean);
                        }
                    }
                }
            );
        }

        #[test_matrix([32, 64])]
        fn test_sum_sq(threads: usize) for F in [f32] {
            let input: Tensor = [ROWS, COLS];
            let output: Tensor = [ROWS] as Range;

            assert_eq!(
                test_sum_sq_kernel(input(), output()) for (1, 1, 1) @ (threads),
                {
                    // Compute sum of squares per row
                    for r in 0..ROWS {
                        let mut sum_sq = 0.0;
                        for c in 0..COLS {
                            let val = input[r * COLS + c].into_f64();
                            sum_sq += val * val;
                        }
                        output[r] = F::from_f64(sum_sq);
                    }
                }
            );
        }

        #[test_matrix([32, 64])]
        fn test_normalize(threads: usize) for F in [f32] {
            let input: Tensor = [ROWS, COLS];
            let output: Tensor = [ROWS, COLS] as Range;

            assert_eq!(
                test_normalize_kernel(input(), output()) for (1, 1, 1) @ (threads),
                {
                    // Compute x_hat = (x - mean) / std
                    for r in 0..ROWS {
                        let mut mean = 0.0;
                        for c in 0..COLS {
                            mean += input[r * COLS + c].into_f64();
                        }
                        mean /= COLS as f64;

                        let mut var = 0.0;
                        for c in 0..COLS {
                            let diff = input[r * COLS + c].into_f64() - mean;
                            var += diff * diff;
                        }
                        var /= COLS as f64;

                        let std = (var + EPSILON as f64).sqrt();

                        for c in 0..COLS {
                            let x_hat = (input[r * COLS + c].into_f64() - mean) / std;
                            output[r * COLS + c] = F::from_f64(x_hat);
                        }
                    }
                }
            );
        }

        #[test_matrix([32, 64])]
        fn test_layer_norm_forward(threads: usize) for F in [f32] {
            let input: Tensor = [ROWS, COLS];
            let ln_weight: Tensor = [COLS];
            let ln_bias: Tensor = [COLS];
            let output: Tensor = [ROWS, COLS] as Range;

            assert_eq!(
                test_ln_forward_kernel(input(), ln_weight(), ln_bias(), output()) for (1, 1, 1) @ (threads),
                {
                    // Reference implementation
                    for r in 0..ROWS {
                        // Compute mean
                        let mut mean = 0.0;
                        for c in 0..COLS {
                            mean += input[r * COLS + c].into_f64();
                        }
                        mean /= COLS as f64;

                        // Compute variance
                        let mut var = 0.0;
                        for c in 0..COLS {
                            let diff = input[r * COLS + c].into_f64() - mean;
                            var += diff * diff;
                        }
                        var /= COLS as f64;

                        // Compute std
                        let std = (var + EPSILON as f64).sqrt();

                        // Normalize and apply affine
                        for c in 0..COLS {
                            let x_hat = (input[r * COLS + c].into_f64() - mean) / std;
                            let out = ln_weight[c].into_f64() * x_hat + ln_bias[c].into_f64();
                            output[r * COLS + c] = F::from_f64(out);
                        }
                    }
                }
            );
        }

        #[test_matrix([64])]
        fn test_layer_norm_l2_grad(threads: usize) for F in [f32] {
            let input: Tensor = [ROWS, COLS];
            let target: Tensor = [ROWS, COLS];
            let ln_weight: Tensor = [COLS];
            let ln_bias: Tensor = [COLS];
            let output: Tensor = [ROWS, COLS] as Range;

            assert_eq!(
                test_ln_l2_grad_kernel(input(), target(), ln_weight(), ln_bias(), output()) for (1, 1, 1) @ (threads),
                {
                    // Reference implementation
                    for r in 0..ROWS {
                        // Forward pass: compute mean, var, std, x_hat, y
                        let mut mean = 0.0;
                        for c in 0..COLS {
                            mean += input[r * COLS + c].into_f64();
                        }
                        mean /= COLS as f64;

                        let mut var = 0.0;
                        for c in 0..COLS {
                            let diff = input[r * COLS + c].into_f64() - mean;
                            var += diff * diff;
                        }
                        var /= COLS as f64;

                        let std = (var + EPSILON as f64).sqrt();

                        let mut x_hat = vec![0.0f64; COLS];
                        let mut y = vec![0.0f64; COLS];
                        for c in 0..COLS {
                            x_hat[c] = (input[r * COLS + c].into_f64() - mean) / std;
                            y[c] = ln_weight[c].into_f64() * x_hat[c] + ln_bias[c].into_f64();
                        }

                        // L2 grad: dl_dout = y - target
                        let mut dl_dout = vec![0.0f64; COLS];
                        for c in 0..COLS {
                            dl_dout[c] = y[c] - target[r * COLS + c].into_f64();
                        }

                        // dl_dnorm = dl_dout * weight
                        let mut dl_dnorm = vec![0.0f64; COLS];
                        for c in 0..COLS {
                            dl_dnorm[c] = dl_dout[c] * ln_weight[c].into_f64();
                        }

                        // sum(dl_dnorm), sum(dl_dnorm * x_hat)
                        let mut sum_dl_dnorm = 0.0;
                        let mut sum_dl_dnorm_xhat = 0.0;
                        for c in 0..COLS {
                            sum_dl_dnorm += dl_dnorm[c];
                            sum_dl_dnorm_xhat += dl_dnorm[c] * x_hat[c];
                        }

                        // dl_dx = (dl_dnorm * C - sum_dl_dnorm - x_hat * sum_dl_dnorm_xhat) / (std * C)
                        let c_f = COLS as f64;
                        for c in 0..COLS {
                            let grad = (dl_dnorm[c] * c_f - sum_dl_dnorm - x_hat[c] * sum_dl_dnorm_xhat) / (std * c_f);
                            output[r * COLS + c] = F::from_f64(grad);
                        }
                    }
                }
            );
        }
    }
}

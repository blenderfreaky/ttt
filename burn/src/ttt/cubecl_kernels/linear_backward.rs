//! Fused TTT-Linear backward pass kernel.
//!
//! Given dL/d_output (gradient of loss w.r.t. output), computes gradients w.r.t.:
//! - xq, xk, xv (input projections)
//! - W, b (initial weights before TTT update)
//! - ttt_lr_eta (learned learning rate)
//! - ln_weight, ln_bias (layer norm parameters)

use cubecl::prelude::*;

/// Scaling factor for epsilon (see forward kernel).
const EPSILON_SCALE_INV: f32 = 1e-9;

/// Configuration for the fused TTT backward kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FusedTttBackwardConfig {
    pub seq_len: u32,
    pub head_dim: u32,
    pub epsilon_scaled: u32,
}

impl FusedTttBackwardConfig {
    #[must_use]
    pub fn new(seq_len: u32, head_dim: u32, epsilon: f32) -> Self {
        Self {
            seq_len,
            head_dim,
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            epsilon_scaled: (epsilon / EPSILON_SCALE_INV) as u32,
        }
    }

    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    pub fn epsilon(&self) -> f32 {
        self.epsilon_scaled as f32 * EPSILON_SCALE_INV
    }
}

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
pub fn fused_ttt_backward_kernel<F: Float>(
    inputs: &ForwardInputs<F>,
    grad_output: &Tensor<F>,
    outputs: &mut GradOutputs<F>,
    #[comptime] config: FusedTttBackwardConfig,
) {
    let batch_idx = CUBE_POS_X;
    let head_idx = CUBE_POS_Y;

    let batch_size = inputs.xq.shape(0);
    let num_heads = inputs.xq.shape(1);
    let seq_len = config.seq_len;
    let head_dim = config.head_dim;
    let epsilon = config.epsilon();

    let dim_idx = UNIT_POS_X;
    let seq_idx = UNIT_POS_Y;

    // Shared memory for intermediates
    let mut z1_shared = SharedMemory::<F>::new(seq_len * head_dim);
    let mut grad_l_shared = SharedMemory::<F>::new(seq_len * head_dim);
    let mut z1_bar_shared = SharedMemory::<F>::new(seq_len * head_dim);
    let mut dl_dz1_bar_shared = SharedMemory::<F>::new(seq_len * head_dim);
    let mut dl_dgrad_l_shared = SharedMemory::<F>::new(seq_len * head_dim);
    let mut dl_d_z1_shared = SharedMemory::<F>::new(seq_len * head_dim);

    // Layer norm stats
    let mut z1_mean_shared = SharedMemory::<F>::new(seq_len);
    let mut z1_inv_std_shared = SharedMemory::<F>::new(seq_len);
    let mut z1_bar_mean_shared = SharedMemory::<F>::new(seq_len);
    let mut z1_bar_inv_std_shared = SharedMemory::<F>::new(seq_len);

    // Gradients accumulator
    let mut grad_xq_local = F::new(0.0);
    let mut grad_xk_local = F::new(0.0);
    let mut grad_xv_local = F::new(0.0);

    if batch_idx < batch_size && head_idx < num_heads {
        let bh_seq_dim_base =
            batch_idx * num_heads * seq_len * head_dim + head_idx * seq_len * head_dim;
        let bh_dim_dim_base =
            batch_idx * num_heads * head_dim * head_dim + head_idx * head_dim * head_dim;
        let bh_dim_base = batch_idx * num_heads * head_dim + head_idx * head_dim;
        let bh_seq_base = batch_idx * num_heads * seq_len + head_idx * seq_len;
        let h_dim_base = head_idx * head_dim;

        // We need to recompute forward values for the backward pass

        // z1 = xk @ W + b
        if seq_idx < seq_len && dim_idx < head_dim {
            let xk_base = bh_seq_dim_base + seq_idx * head_dim;
            let mut z1_val = F::new(0.0);

            for k in 0..head_dim {
                let xk_val = inputs.xk[xk_base + k];
                let w_val = inputs.weight[bh_dim_dim_base + k * head_dim + dim_idx];
                z1_val += xk_val * w_val;
            }
            z1_val += inputs.bias[bh_dim_base + dim_idx];
            z1_shared[seq_idx * head_dim + dim_idx] = z1_val;
        }
        sync_cube();

        // Layer norm stats for z1
        if seq_idx < seq_len && dim_idx == 0 {
            let mut sum = F::new(0.0);
            for d in 0..head_dim {
                sum += z1_shared[seq_idx * head_dim + d];
            }
            let mean = sum / F::cast_from(head_dim);

            let mut var_sum = F::new(0.0);
            for d in 0..head_dim {
                let diff = z1_shared[seq_idx * head_dim + d] - mean;
                var_sum += diff * diff;
            }
            let variance = var_sum / F::cast_from(head_dim);
            let inv_std = F::new(1.0) / F::sqrt(variance + F::new(epsilon));

            z1_mean_shared[seq_idx] = mean;
            z1_inv_std_shared[seq_idx] = inv_std;
        }
        sync_cube();

        // grad_l = backward of L2 loss through layer norm
        // This is the internal gradient used in the forward pass weight update
        if seq_idx < seq_len && dim_idx < head_dim {
            let idx = bh_seq_dim_base + seq_idx * head_dim + dim_idx;
            let mean = z1_mean_shared[seq_idx];
            let inv_std = z1_inv_std_shared[seq_idx];

            let z1_val = z1_shared[seq_idx * head_dim + dim_idx];
            let z1_norm = (z1_val - mean) * inv_std;

            let ln_w = inputs.ln_weight[h_dim_base + dim_idx];
            let ln_b = inputs.ln_bias[h_dim_base + dim_idx];
            let ln_out = ln_w * z1_norm + ln_b;

            let target = inputs.xv[idx] - inputs.xk[idx];

            let dl_dln_out = ln_out - target;

            let dl_d_z1_norm = dl_dln_out * ln_w;

            grad_l_shared[seq_idx * head_dim + dim_idx] = dl_d_z1_norm;
        }
        sync_cube();

        if seq_idx < seq_len && dim_idx < head_dim {
            let mean = z1_mean_shared[seq_idx];
            let inv_std = z1_inv_std_shared[seq_idx];

            let mut dl_dnorm_sum = F::new(0.0);
            let mut dl_dnorm_norm_sum = F::new(0.0);

            for d in 0..head_dim {
                let z1_d = z1_shared[seq_idx * head_dim + d];
                let z1_norm_d = (z1_d - mean) * inv_std;
                let dl_dnorm_d = grad_l_shared[seq_idx * head_dim + d];
                dl_dnorm_sum += dl_dnorm_d;
                dl_dnorm_norm_sum += dl_dnorm_d * z1_norm_d;
            }

            let z1_val = z1_shared[seq_idx * head_dim + dim_idx];
            let z1_norm = (z1_val - mean) * inv_std;
            let dl_dnorm = grad_l_shared[seq_idx * head_dim + dim_idx];
            let n = F::cast_from(head_dim);

            let grad_l_val =
                (dl_dnorm * n - dl_dnorm_sum - z1_norm * dl_dnorm_norm_sum) * inv_std / n;
            grad_l_shared[seq_idx * head_dim + dim_idx] = grad_l_val;
        }
        sync_cube();

        // z1_bar = xq @ W - (eta * attn1) @ grad_l + b1_bar
        // where b1_bar = b - eta @ grad_l
        if seq_idx < seq_len && dim_idx < head_dim {
            let token_eta_i = inputs.token_eta[seq_idx];
            let xq_base = bh_seq_dim_base + seq_idx * head_dim;

            // b1_bar[i,d] = b[d] - sum_{j<=i}(eta[i,j] * grad_l[j,d])
            let mut eta_grad_sum = F::new(0.0);
            for j in 0..seq_len {
                if j <= seq_idx {
                    let eta_ij = token_eta_i * inputs.ttt_lr_eta[bh_seq_base + j];
                    eta_grad_sum += eta_ij * grad_l_shared[j * head_dim + dim_idx];
                }
            }
            let b1_bar = inputs.bias[bh_dim_base + dim_idx] - eta_grad_sum;

            // xq @ W
            let mut xq_w = F::new(0.0);
            for k in 0..head_dim {
                xq_w += inputs.xq[xq_base + k]
                    * inputs.weight[bh_dim_dim_base + k * head_dim + dim_idx];
            }

            // -(eta * attn1) @ grad_l
            let mut correction = F::new(0.0);
            for j in 0..seq_len {
                if j <= seq_idx {
                    let xk_base_j = bh_seq_dim_base + j * head_dim;
                    let mut attn_ij = F::new(0.0);
                    for k in 0..head_dim {
                        attn_ij += inputs.xq[xq_base + k] * inputs.xk[xk_base_j + k];
                    }
                    let eta_ij = token_eta_i * inputs.ttt_lr_eta[bh_seq_base + j];
                    correction += eta_ij * attn_ij * grad_l_shared[j * head_dim + dim_idx];
                }
            }

            let z1_bar = xq_w - correction + b1_bar;
            z1_bar_shared[seq_idx * head_dim + dim_idx] = z1_bar;
        }
        sync_cube();

        // Layer norm stats for z1_bar
        if seq_idx < seq_len && dim_idx == 0 {
            let mut sum = F::new(0.0);
            for d in 0..head_dim {
                sum += z1_bar_shared[seq_idx * head_dim + d];
            }
            let mean = sum / F::cast_from(head_dim);

            let mut var_sum = F::new(0.0);
            for d in 0..head_dim {
                let diff = z1_bar_shared[seq_idx * head_dim + d] - mean;
                var_sum += diff * diff;
            }
            let variance = var_sum / F::cast_from(head_dim);
            let inv_std = F::new(1.0) / F::sqrt(variance + F::new(epsilon));

            z1_bar_mean_shared[seq_idx] = mean;
            z1_bar_inv_std_shared[seq_idx] = inv_std;
        }
        sync_cube();

        // Backward through output = xq + LN(z1_bar)
        // dL/dxq += grad_output
        // dL/d_z1_bar_norm = grad_output
        if seq_idx < seq_len && dim_idx < head_dim {
            let idx = bh_seq_dim_base + seq_idx * head_dim + dim_idx;
            let grad_out = grad_output[idx];

            // Direct contribution to xq gradient from residual connection
            grad_xq_local = grad_out;

            // Backprop through layer norm affine: LN_out = ln_w * z1_bar_norm + ln_b
            let ln_w = inputs.ln_weight[h_dim_base + dim_idx];
            let dl_d_z1_bar_norm = grad_out * ln_w;

            dl_dz1_bar_shared[seq_idx * head_dim + dim_idx] = dl_d_z1_bar_norm;
        }
        sync_cube();

        // Complete layer norm backward for z1_bar
        if seq_idx < seq_len && dim_idx < head_dim {
            let mean = z1_bar_mean_shared[seq_idx];
            let inv_std = z1_bar_inv_std_shared[seq_idx];

            let mut dl_dnorm_sum = F::new(0.0);
            let mut dl_dnorm_norm_sum = F::new(0.0);

            for d in 0..head_dim {
                let z1_bar_d = z1_bar_shared[seq_idx * head_dim + d];
                let z1_bar_norm_d = (z1_bar_d - mean) * inv_std;
                let dl_dnorm_d = dl_dz1_bar_shared[seq_idx * head_dim + d];
                dl_dnorm_sum += dl_dnorm_d;
                dl_dnorm_norm_sum += dl_dnorm_d * z1_bar_norm_d;
            }

            let z1_bar_val = z1_bar_shared[seq_idx * head_dim + dim_idx];
            let z1_bar_norm = (z1_bar_val - mean) * inv_std;
            let dl_dnorm = dl_dz1_bar_shared[seq_idx * head_dim + dim_idx];
            let n = F::cast_from(head_dim);

            let dl_dz1_bar =
                (dl_dnorm * n - dl_dnorm_sum - z1_bar_norm * dl_dnorm_norm_sum) * inv_std / n;
            dl_dz1_bar_shared[seq_idx * head_dim + dim_idx] = dl_dz1_bar;
        }
        sync_cube();

        // Backward through z1_bar = xq @ W - (eta * attn1) @ grad_l + b1_bar
        // where b1_bar = b - eta @ grad_l

        // From z1_bar = xq @ W: dL/d_xq += dL/d_z1_bar @ W^T
        if seq_idx < seq_len && dim_idx < head_dim {
            for k in 0..head_dim {
                let dl_dz1_bar_k = dl_dz1_bar_shared[seq_idx * head_dim + k];
                let w_val = inputs.weight[bh_dim_dim_base + dim_idx * head_dim + k];
                grad_xq_local += dl_dz1_bar_k * w_val;
            }
        }
        sync_cube();

        // dL/d_grad_l from the -(eta * attn1) @ grad_l and -eta @ grad_l terms
        if seq_idx < seq_len && dim_idx < head_dim {
            let mut dl_dgrad_l_val = F::new(0.0);

            let j = seq_idx;
            for i in j..seq_len {
                let token_eta_i = inputs.token_eta[i];
                let eta_ij = token_eta_i * inputs.ttt_lr_eta[bh_seq_base + j];

                // attn1[i, j] = xq[i] @ xk[j]
                let xq_base_i = bh_seq_dim_base + i * head_dim;
                let xk_base_j = bh_seq_dim_base + j * head_dim;
                let mut attn_ij = F::new(0.0);
                for k in 0..head_dim {
                    attn_ij += inputs.xq[xq_base_i + k] * inputs.xk[xk_base_j + k];
                }

                let dl_dz1_bar_id = dl_dz1_bar_shared[i * head_dim + dim_idx];

                // From b1_bar term
                dl_dgrad_l_val -= eta_ij * dl_dz1_bar_id;

                // From (eta * attn1) @ grad_l term
                dl_dgrad_l_val -= eta_ij * attn_ij * dl_dz1_bar_id;
            }

            dl_dgrad_l_shared[seq_idx * head_dim + dim_idx] = dl_dgrad_l_val;
        }
        sync_cube();

        if seq_idx < seq_len && dim_idx < head_dim {
            let mean = z1_mean_shared[seq_idx];
            let inv_std = z1_inv_std_shared[seq_idx];
            let ln_w = inputs.ln_weight[h_dim_base + dim_idx];

            let mut dl_dgrad_l_sum = F::new(0.0);
            let mut dl_dgrad_l_znorm_sum = F::new(0.0);
            for d in 0..head_dim {
                let z1_d = z1_shared[seq_idx * head_dim + d];
                let z1_norm_d = (z1_d - mean) * inv_std;
                let dl_dgrad_l_d = dl_dgrad_l_shared[seq_idx * head_dim + d];
                dl_dgrad_l_sum += dl_dgrad_l_d;
                dl_dgrad_l_znorm_sum += dl_dgrad_l_d * z1_norm_d;
            }

            let z1_val = z1_shared[seq_idx * head_dim + dim_idx];
            let z1_norm = (z1_val - mean) * inv_std;
            let dl_dgrad_l = dl_dgrad_l_shared[seq_idx * head_dim + dim_idx];
            let n = F::cast_from(head_dim);

            let dl_d_dl_dnorm =
                (dl_dgrad_l * n - dl_dgrad_l_sum - z1_norm * dl_dgrad_l_znorm_sum) * inv_std / n;

            let dl_d_dl_dln_out = dl_d_dl_dnorm * ln_w;

            let dl_d_target = -dl_d_dl_dln_out;

            // target = xv - xk, so:
            grad_xv_local = dl_d_target;
            grad_xk_local = -dl_d_target;

            let dl_d_z1_norm_via_dl_dnorm = dl_d_dl_dln_out * ln_w;

            let mut dl_dnorm_znorm_sum = F::new(0.0);
            let mut dl_dgrad_l_znorm_sum_for_path_b = F::new(0.0);
            for d in 0..head_dim {
                let z1_d = z1_shared[seq_idx * head_dim + d];
                let z1_norm_d = (z1_d - mean) * inv_std;
                let ln_w_d = inputs.ln_weight[h_dim_base + d];
                let ln_b_d = inputs.ln_bias[h_dim_base + d];
                let ln_out_d = ln_w_d * z1_norm_d + ln_b_d;
                let target_d = inputs.xv[bh_seq_dim_base + seq_idx * head_dim + d]
                    - inputs.xk[bh_seq_dim_base + seq_idx * head_dim + d];
                let dl_dnorm_d = (ln_out_d - target_d) * ln_w_d;

                dl_dnorm_znorm_sum += dl_dnorm_d * z1_norm_d;
                dl_dgrad_l_znorm_sum_for_path_b +=
                    dl_dgrad_l_shared[seq_idx * head_dim + d] * z1_norm_d;
            }

            let dl_dgrad_l_here = dl_dgrad_l_shared[seq_idx * head_dim + dim_idx];
            let ln_w_here = inputs.ln_weight[h_dim_base + dim_idx];
            let ln_b_here = inputs.ln_bias[h_dim_base + dim_idx];
            let ln_out_here = ln_w_here * z1_norm + ln_b_here;
            let target_here = inputs.xv[bh_seq_dim_base + seq_idx * head_dim + dim_idx]
                - inputs.xk[bh_seq_dim_base + seq_idx * head_dim + dim_idx];
            let dl_dnorm_here = (ln_out_here - target_here) * ln_w_here;
            let n = F::cast_from(head_dim);

            let dl_d_z1_norm_via_direct = (-dl_dgrad_l_here * dl_dnorm_znorm_sum
                - dl_dgrad_l_znorm_sum_for_path_b * dl_dnorm_here)
                * inv_std
                / n;

            let dl_d_z1_norm = dl_d_z1_norm_via_dl_dnorm + dl_d_z1_norm_via_direct;

            dl_d_z1_shared[seq_idx * head_dim + dim_idx] = dl_d_z1_norm;
        }
        sync_cube();

        if seq_idx < seq_len && dim_idx < head_dim {
            let mean = z1_mean_shared[seq_idx];
            let inv_std = z1_inv_std_shared[seq_idx];

            let mut dl_d_z1_norm_sum = F::new(0.0);
            let mut dl_d_z1_norm_znorm_sum = F::new(0.0);
            let mut dl_dgrad_l_times_grad_l_sum = F::new(0.0);
            for d in 0..head_dim {
                let z1_d = z1_shared[seq_idx * head_dim + d];
                let z1_norm_d = (z1_d - mean) * inv_std;
                let dl_d_z1_norm_d = dl_d_z1_shared[seq_idx * head_dim + d];
                dl_d_z1_norm_sum += dl_d_z1_norm_d;
                dl_d_z1_norm_znorm_sum += dl_d_z1_norm_d * z1_norm_d;

                let dl_dgrad_l_d = dl_dgrad_l_shared[seq_idx * head_dim + d];
                let grad_l_d = grad_l_shared[seq_idx * head_dim + d];
                dl_dgrad_l_times_grad_l_sum += dl_dgrad_l_d * grad_l_d;
            }

            let z1_val = z1_shared[seq_idx * head_dim + dim_idx];
            let z1_norm = (z1_val - mean) * inv_std;
            let dl_d_z1_norm_val = dl_d_z1_shared[seq_idx * head_dim + dim_idx];
            let n = F::cast_from(head_dim);

            let dl_d_z1_from_norm =
                (dl_d_z1_norm_val * n - dl_d_z1_norm_sum - z1_norm * dl_d_z1_norm_znorm_sum)
                    * inv_std
                    / n;

            let dl_d_z1_via_std = -dl_dgrad_l_times_grad_l_sum * z1_norm * inv_std / n;

            let dl_d_z1 = dl_d_z1_from_norm + dl_d_z1_via_std;
            dl_d_z1_shared[seq_idx * head_dim + dim_idx] = dl_d_z1;
        }
        sync_cube();

        if seq_idx < seq_len && dim_idx < head_dim {
            let i = seq_idx;
            let token_eta_i = inputs.token_eta[i];

            for j in 0..seq_len {
                if j <= i {
                    let eta_ij = token_eta_i * inputs.ttt_lr_eta[bh_seq_base + j];

                    // dL/d_attn1[i,j] = -eta[i,j] * sum_d(grad_l[j,d] * dL/d_z1_bar[i,d])
                    let mut grad_attn_contrib = F::new(0.0);
                    for d in 0..head_dim {
                        grad_attn_contrib +=
                            grad_l_shared[j * head_dim + d] * dl_dz1_bar_shared[i * head_dim + d];
                    }
                    let dl_d_attn_ij = -eta_ij * grad_attn_contrib;

                    // attn1[i,j] = sum_k(xq[i,k] * xk[j,k])
                    // dL/d_xq[i, dim_idx] += dL/d_attn1[i,j] * xk[j, dim_idx]
                    let xk_jd = inputs.xk[bh_seq_dim_base + j * head_dim + dim_idx];
                    grad_xq_local += dl_d_attn_ij * xk_jd;
                }
            }
        }
        sync_cube();

        if seq_idx < seq_len && dim_idx < head_dim {
            let j = seq_idx;

            for i in j..seq_len {
                let token_eta_i = inputs.token_eta[i];
                let eta_ij = token_eta_i * inputs.ttt_lr_eta[bh_seq_base + j];

                let mut grad_attn_contrib = F::new(0.0);
                for d in 0..head_dim {
                    grad_attn_contrib +=
                        grad_l_shared[j * head_dim + d] * dl_dz1_bar_shared[i * head_dim + d];
                }
                let dl_d_attn_ij = -eta_ij * grad_attn_contrib;

                let xq_id = inputs.xq[bh_seq_dim_base + i * head_dim + dim_idx];
                grad_xk_local += dl_d_attn_ij * xq_id;
            }
        }
        sync_cube();

        if seq_idx < seq_len && dim_idx < head_dim {
            for k in 0..head_dim {
                let dl_d_z1_k = dl_d_z1_shared[seq_idx * head_dim + k];
                let w_val = inputs.weight[bh_dim_dim_base + dim_idx * head_dim + k];
                grad_xk_local += dl_d_z1_k * w_val;
            }
        }
        sync_cube();

        // Write out gradients
        if seq_idx < seq_len && dim_idx < head_dim {
            let idx = bh_seq_dim_base + seq_idx * head_dim + dim_idx;
            outputs.grad_xq[idx] = grad_xq_local;
            outputs.grad_xk[idx] = grad_xk_local;
            outputs.grad_xv[idx] = grad_xv_local;
        }
        sync_cube();

        if dim_idx < head_dim && seq_idx == 0 {
            for col in 0..head_dim {
                let mut grad_w = F::new(0.0);
                for s in 0..seq_len {
                    let xq_val = inputs.xq[bh_seq_dim_base + s * head_dim + dim_idx];
                    let dl_dz1_bar = dl_dz1_bar_shared[s * head_dim + col];
                    grad_w += xq_val * dl_dz1_bar;
                }
                outputs.grad_weight[bh_dim_dim_base + dim_idx * head_dim + col] = grad_w;
            }

            let mut grad_b = F::new(0.0);
            for s in 0..seq_len {
                grad_b += dl_dz1_bar_shared[s * head_dim + dim_idx];
            }
            outputs.grad_bias[bh_dim_base + dim_idx] = grad_b;
        }
        sync_cube();

        if seq_idx < seq_len && dim_idx == 0 {
            let j = seq_idx;
            let mut grad_lr = F::new(0.0);

            for i in j..seq_len {
                let token_eta_i = inputs.token_eta[i];
                let xq_base_i = bh_seq_dim_base + i * head_dim;
                let xk_base_j = bh_seq_dim_base + j * head_dim;

                let mut attn_ij = F::new(0.0);
                for k in 0..head_dim {
                    attn_ij += inputs.xq[xq_base_i + k] * inputs.xk[xk_base_j + k];
                }

                let mut grad_dot = F::new(0.0);
                for d in 0..head_dim {
                    grad_dot +=
                        grad_l_shared[j * head_dim + d] * dl_dz1_bar_shared[i * head_dim + d];
                }

                let dl_d_eta_ij = -(F::new(1.0) + attn_ij) * grad_dot;

                grad_lr += dl_d_eta_ij * token_eta_i;
            }

            outputs.grad_ttt_lr_eta[bh_seq_base + j] = grad_lr;
        }
        sync_cube();

        if seq_idx == 0 && dim_idx < head_dim {
            let mut grad_ln_w = F::new(0.0);
            let mut grad_ln_b = F::new(0.0);

            for s in 0..seq_len {
                let mean = z1_bar_mean_shared[s];
                let inv_std = z1_bar_inv_std_shared[s];
                let z1_bar_val = z1_bar_shared[s * head_dim + dim_idx];
                let z1_bar_norm = (z1_bar_val - mean) * inv_std;

                let grad_out = grad_output[bh_seq_dim_base + s * head_dim + dim_idx];
                grad_ln_w += grad_out * z1_bar_norm;
                grad_ln_b += grad_out;
            }

            outputs.grad_ln_weight[batch_idx * num_heads * head_dim + h_dim_base + dim_idx] =
                grad_ln_w;
            outputs.grad_ln_bias[batch_idx * num_heads * head_dim + h_dim_base + dim_idx] =
                grad_ln_b;
        }
    }
}

/// Launch the fused TTT backward kernel.
pub fn launch_fused_ttt_backward<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    xq: TensorHandleRef<R>,
    xk: TensorHandleRef<R>,
    xv: TensorHandleRef<R>,
    weight: TensorHandleRef<R>,
    bias: TensorHandleRef<R>,
    token_eta: TensorHandleRef<R>,
    ttt_lr_eta: TensorHandleRef<R>,
    ln_weight: TensorHandleRef<R>,
    ln_bias: TensorHandleRef<R>,
    grad_output: TensorHandleRef<R>,
    grad_xq: TensorHandleRef<R>,
    grad_xk: TensorHandleRef<R>,
    grad_xv: TensorHandleRef<R>,
    grad_weight: TensorHandleRef<R>,
    grad_bias: TensorHandleRef<R>,
    grad_ttt_lr_eta: TensorHandleRef<R>,
    grad_ln_weight: TensorHandleRef<R>,
    grad_ln_bias: TensorHandleRef<R>,
    config: FusedTttBackwardConfig,
) {
    let batch_size = xq.shape[0] as u32;
    let num_heads = xq.shape[1] as u32;
    let seq_len = config.seq_len;
    let head_dim = config.head_dim;

    let cube_dim = CubeDim::new_2d(head_dim, seq_len);

    unsafe {
        fused_ttt_backward_kernel::launch::<F, R>(
            client,
            CubeCount::Static(batch_size, num_heads, 1),
            cube_dim,
            ForwardInputsLaunch::new(
                TensorArg::from_raw_parts::<F>(xq.handle, xq.strides, xq.shape, 1),
                TensorArg::from_raw_parts::<F>(xk.handle, xk.strides, xk.shape, 1),
                TensorArg::from_raw_parts::<F>(xv.handle, xv.strides, xv.shape, 1),
                TensorArg::from_raw_parts::<F>(weight.handle, weight.strides, weight.shape, 1),
                TensorArg::from_raw_parts::<F>(bias.handle, bias.strides, bias.shape, 1),
                TensorArg::from_raw_parts::<F>(
                    token_eta.handle,
                    token_eta.strides,
                    token_eta.shape,
                    1,
                ),
                TensorArg::from_raw_parts::<F>(
                    ttt_lr_eta.handle,
                    ttt_lr_eta.strides,
                    ttt_lr_eta.shape,
                    1,
                ),
                TensorArg::from_raw_parts::<F>(
                    ln_weight.handle,
                    ln_weight.strides,
                    ln_weight.shape,
                    1,
                ),
                TensorArg::from_raw_parts::<F>(ln_bias.handle, ln_bias.strides, ln_bias.shape, 1),
            ),
            TensorArg::from_raw_parts::<F>(
                grad_output.handle,
                grad_output.strides,
                grad_output.shape,
                1,
            ),
            GradOutputsLaunch::new(
                TensorArg::from_raw_parts::<F>(grad_xq.handle, grad_xq.strides, grad_xq.shape, 1),
                TensorArg::from_raw_parts::<F>(grad_xk.handle, grad_xk.strides, grad_xk.shape, 1),
                TensorArg::from_raw_parts::<F>(grad_xv.handle, grad_xv.strides, grad_xv.shape, 1),
                TensorArg::from_raw_parts::<F>(
                    grad_weight.handle,
                    grad_weight.strides,
                    grad_weight.shape,
                    1,
                ),
                TensorArg::from_raw_parts::<F>(
                    grad_bias.handle,
                    grad_bias.strides,
                    grad_bias.shape,
                    1,
                ),
                TensorArg::from_raw_parts::<F>(
                    grad_ttt_lr_eta.handle,
                    grad_ttt_lr_eta.strides,
                    grad_ttt_lr_eta.shape,
                    1,
                ),
                TensorArg::from_raw_parts::<F>(
                    grad_ln_weight.handle,
                    grad_ln_weight.strides,
                    grad_ln_weight.shape,
                    1,
                ),
                TensorArg::from_raw_parts::<F>(
                    grad_ln_bias.handle,
                    grad_ln_bias.strides,
                    grad_ln_bias.shape,
                    1,
                ),
            ),
            config,
        )
        .unwrap();
    }
}

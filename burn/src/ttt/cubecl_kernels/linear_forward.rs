//! Fully fused TTT-Linear mini-batch forward kernel.
//!
//! This kernel performs the entire TTT-Linear forward pass in a single kernel launch,
//! keeping all intermediates in shared memory to minimize global memory traffic.
//!
//! Operations:
//! 1. z1 = xk @ W + b
//! 2. reconstruction_target = xv - xk
//! 3. Layer norm(z1) + L2 gradient computation
//! 4. Eta matrix construction with tril masking
//! 5. attn1 = tril(xq @ xk^T)
//! 6. b1_bar = b - eta @ grad
//! 7. z1_bar = xq @ W - (eta * attn1) @ grad + b1_bar
//! 8. Weight/bias updates with last row of eta
//! 9. output = xq + layer_norm(z1_bar)

use cubecl::prelude::*;

use crate::ttt::cubecl_kernels::FusedTttConfig;

/// Fully fused TTT-Linear forward pass kernel.
///
/// Each CUBE (thread block) handles one (batch, head) pair.
/// Thread layout within cube: 2D grid of (head_dim, seq_len)
/// - UNIT_POS_X: dimension index (0..head_dim)
/// - UNIT_POS_Y: sequence index (0..seq_len)
///
/// Shared memory layout:
/// - z1: [seq_len, head_dim] - forward output before layer norm
/// - grad: [seq_len, head_dim] - gradient of layer norm + L2 loss
/// - z1_bar: [seq_len, head_dim] - output after gradient correction
/// - scratch: [head_dim] - for reductions
///
/// Input shapes:
/// - xq, xk, xv: [batch_size, num_heads, seq_len, head_dim]
/// - weight: [batch_size, num_heads, head_dim, head_dim] (state, modified in place)
/// - bias: [batch_size, num_heads, head_dim] (state, modified in place)
/// - token_eta: [seq_len] - position-based scale (1/k)
/// - ttt_lr_eta: [batch_size, num_heads, seq_len] - learned learning rate
/// - ln_weight, ln_bias: [num_heads, head_dim] - layer norm parameters
///
/// Output:
/// - output: [batch_size, num_heads, seq_len, head_dim]
#[cube(launch)]
pub fn fused_ttt_forward_kernel<F: Float>(
    // Inputs
    xq: &Tensor<F>,
    xk: &Tensor<F>,
    xv: &Tensor<F>,
    token_eta: &Tensor<F>,
    ttt_lr_eta: &Tensor<F>,
    ln_weight: &Tensor<F>,
    ln_bias: &Tensor<F>,
    // State (rw)
    weight: &mut Tensor<F>,
    bias: &mut Tensor<F>,
    // Output (write-only)
    output: &mut Tensor<F>,
    #[comptime] config: FusedTttConfig,
) {
    let batch_idx = CUBE_POS_X as usize;
    let head_idx = CUBE_POS_Y as usize;

    let batch_size = xq.shape(0) as usize;
    let num_heads = xq.shape(1) as usize;
    let seq_len = config.seq_len;
    let head_dim = config.head_dim;
    let epsilon = config.epsilon();

    // Thread indices
    let dim_idx = UNIT_POS_X as usize; // 0..head_dim
    let seq_idx = UNIT_POS_Y as usize; // 0..seq_len

    // Shared memory for intermediates
    // z1[seq, dim], grad[seq, dim]
    let mut z1_shared = SharedMemory::<F>::new(seq_len * head_dim);
    let mut grad_shared = SharedMemory::<F>::new(seq_len * head_dim);
    let mut z1_bar_shared = SharedMemory::<F>::new(seq_len * head_dim);

    // Per-sequence layer norm stats (stored per seq position)
    let mut ln_mean_shared = SharedMemory::<F>::new(seq_len);
    let mut ln_inv_std_shared = SharedMemory::<F>::new(seq_len);

    if batch_idx < batch_size && head_idx < num_heads {
        let bh_seq_dim_base =
            batch_idx * num_heads * seq_len * head_dim + head_idx * seq_len * head_dim;
        let bh_dim_dim_base =
            batch_idx * num_heads * head_dim * head_dim + head_idx * head_dim * head_dim;
        let bh_dim_base = batch_idx * num_heads * head_dim + head_idx * head_dim;
        let bh_seq_base = batch_idx * num_heads * seq_len + head_idx * seq_len;
        let h_dim_base = head_idx * head_dim;

        // Each thread computes z1[seq_idx, dim_idx]

        if seq_idx < seq_len && dim_idx < head_dim {
            let xk_base = bh_seq_dim_base + seq_idx * head_dim;
            let mut z1_val = F::new(0.0);

            for k in 0..head_dim {
                let xk_val = xk[xk_base + k];
                let w_val = weight[bh_dim_dim_base + k * head_dim + dim_idx];
                z1_val += xk_val * w_val;
            }
            z1_val += bias[bh_dim_base + dim_idx];

            z1_shared[seq_idx * head_dim + dim_idx] = z1_val;
        }

        sync_cube();

        if seq_idx < seq_len {
            let mut sum = F::new(0.0);
            for d in 0..head_dim {
                sum += z1_shared[seq_idx * head_dim + d];
            }
            let mean = sum / F::cast_from(head_dim);

            // Compute variance
            let mut var_sum = F::new(0.0);
            for d in 0..head_dim {
                let diff = z1_shared[seq_idx * head_dim + d] - mean;
                var_sum += diff * diff;
            }
            let variance = var_sum / F::cast_from(head_dim);
            let inv_std = F::new(1.0) / F::sqrt(variance + F::new(epsilon));

            if dim_idx == 0 {
                ln_mean_shared[seq_idx] = mean;
                ln_inv_std_shared[seq_idx] = inv_std;
            }
        }

        sync_cube();

        // grad = d/dz1 of ||LN(z1) - target||^2
        // where target = xv - xk

        if seq_idx < seq_len && dim_idx < head_dim {
            let idx = bh_seq_dim_base + seq_idx * head_dim + dim_idx;
            let mean = ln_mean_shared[seq_idx];
            let inv_std = ln_inv_std_shared[seq_idx];

            let z1_val = z1_shared[seq_idx * head_dim + dim_idx];
            let z1_norm = (z1_val - mean) * inv_std;

            // Layer norm output
            let ln_w = ln_weight[h_dim_base + dim_idx];
            let ln_b = ln_bias[h_dim_base + dim_idx];
            let ln_out = ln_w * z1_norm + ln_b;

            // Target and L2 loss gradient
            let target = xv[idx] - xk[idx];
            let dl_dout = ln_out - target;
            let dl_dnorm = dl_dout * ln_w;

            // Store dl_dnorm temporarily in grad_shared for reduction
            grad_shared[seq_idx * head_dim + dim_idx] = dl_dnorm;
        }

        sync_cube();

        // Compute sums needed for layer norm backward: sum(dl_dnorm) and sum(dl_dnorm * z1_norm)
        if seq_idx < seq_len && dim_idx < head_dim {
            let mean = ln_mean_shared[seq_idx];
            let inv_std = ln_inv_std_shared[seq_idx];

            // Compute the two sums (each thread iterates, no reduction. TODO: fix)
            let mut dl_dnorm_sum = F::new(0.0);
            let mut dl_dnorm_norm_sum = F::new(0.0);

            for d in 0..head_dim {
                let z1_d = z1_shared[seq_idx * head_dim + d];
                let z1_norm_d = (z1_d - mean) * inv_std;
                let dl_dnorm_d = grad_shared[seq_idx * head_dim + d];

                dl_dnorm_sum += dl_dnorm_d;
                dl_dnorm_norm_sum += dl_dnorm_d * z1_norm_d;
            }

            // Compute final gradient
            let z1_val = z1_shared[seq_idx * head_dim + dim_idx];
            let z1_norm = (z1_val - mean) * inv_std;
            let dl_dnorm = grad_shared[seq_idx * head_dim + dim_idx];
            let n = F::cast_from(head_dim);

            let grad_val =
                (dl_dnorm * n - dl_dnorm_sum - z1_norm * dl_dnorm_norm_sum) * inv_std / n;

            grad_shared[seq_idx * head_dim + dim_idx] = grad_val;
        }

        sync_cube();

        // b1_bar[i, d] = b[d] - sum_{j<=i}(eta[i,j] * grad[j,d])
        // eta[i,j] = token_eta[i] * ttt_lr_eta[j] for j <= i

        if seq_idx < seq_len && dim_idx < head_dim {
            let token_eta_i = token_eta[seq_idx];
            let mut eta_grad_sum = F::new(0.0);

            for j in 0..seq_len {
                if j <= seq_idx {
                    let eta_ij = token_eta_i * ttt_lr_eta[bh_seq_base + j];
                    let grad_jd = grad_shared[j * head_dim + dim_idx];
                    eta_grad_sum += eta_ij * grad_jd;
                }
            }

            let b1_bar_val = bias[bh_dim_base + dim_idx] - eta_grad_sum;
            // we'll need this for z1_bar
            z1_bar_shared[seq_idx * head_dim + dim_idx] = b1_bar_val;
        }

        sync_cube();

        // attn1[i,j] = xq[i] @ xk[j]^T for j <= i

        if seq_idx < seq_len && dim_idx < head_dim {
            let xq_base = bh_seq_dim_base + seq_idx * head_dim;
            let token_eta_i = token_eta[seq_idx];

            // xq @ W
            let mut xq_w = F::new(0.0);
            for k in 0..head_dim {
                xq_w += xq[xq_base + k] * weight[bh_dim_dim_base + k * head_dim + dim_idx];
            }

            // -(eta * attn1) @ grad
            let mut correction = F::new(0.0);
            for j in 0..seq_len {
                if j <= seq_idx {
                    // attn1[i,j] = xq[i] @ xk[j]
                    let xk_base_j = bh_seq_dim_base + j * head_dim;
                    let mut attn_ij = F::new(0.0);
                    for k in 0..head_dim {
                        attn_ij += xq[xq_base + k] * xk[xk_base_j + k];
                    }

                    let eta_ij = token_eta_i * ttt_lr_eta[bh_seq_base + j];
                    let grad_jd = grad_shared[j * head_dim + dim_idx];
                    correction += eta_ij * attn_ij * grad_jd;
                }
            }

            // b1_bar was stored in z1_bar_shared
            let b1_bar_val = z1_bar_shared[seq_idx * head_dim + dim_idx];

            let z1_bar_val = xq_w - correction + b1_bar_val;
            z1_bar_shared[seq_idx * head_dim + dim_idx] = z1_bar_val;
        }

        sync_cube();

        // W -= (last_eta_col * xk)^T @ grad
        // b -= sum_k(last_eta_col[k] * grad[k])
        // last_eta_col[k] = token_eta[seq_len-1] * ttt_lr_eta[k]

        let last_seq = seq_len - 1;
        let token_eta_last = token_eta[last_seq];

        // Weight update: W[r,c] -= sum_k(eta_k * xk[k,r] * grad[k,c])
        // Thread (dim_idx, seq_idx) handles multiple rows if needed
        // dim_idx is the column, we loop over rows
        if dim_idx < head_dim {
            // Each thread handles all rows for its column (dim_idx)
            // Only one thread per column updates to avoid race conditions
            if seq_idx == 0 {
                for row in 0..head_dim {
                    let mut update = F::new(0.0);
                    for k in 0..seq_len {
                        let eta_k = token_eta_last * ttt_lr_eta[bh_seq_base + k];
                        let xk_kr = xk[bh_seq_dim_base + k * head_dim + row];
                        let grad_kc = grad_shared[k * head_dim + dim_idx];
                        update += eta_k * xk_kr * grad_kc;
                    }

                    let w_idx = bh_dim_dim_base + row * head_dim + dim_idx;
                    weight[w_idx] = weight[w_idx] - update;
                }
            }
        }

        // Bias update: b[d] -= sum_k(eta_k * grad[k,d])
        if seq_idx == 0 && dim_idx < head_dim {
            let mut update = F::new(0.0);

            for k in 0..seq_len {
                let eta_k = token_eta_last * ttt_lr_eta[bh_seq_base + k];
                let grad_kd = grad_shared[k * head_dim + dim_idx];
                update += eta_k * grad_kd;
            }

            bias[bh_dim_base + dim_idx] = bias[bh_dim_base + dim_idx] - update;
        }

        sync_cube();

        // Compute mean and variance of z1_bar for final output

        if seq_idx < seq_len {
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

            if dim_idx == 0 {
                ln_mean_shared[seq_idx] = mean;
                ln_inv_std_shared[seq_idx] = inv_std;
            }
        }

        sync_cube();

        if seq_idx < seq_len && dim_idx < head_dim {
            let idx = bh_seq_dim_base + seq_idx * head_dim + dim_idx;
            let mean = ln_mean_shared[seq_idx];
            let inv_std = ln_inv_std_shared[seq_idx];

            let z1_bar_val = z1_bar_shared[seq_idx * head_dim + dim_idx];
            let z1_bar_norm = (z1_bar_val - mean) * inv_std;

            let ln_w = ln_weight[h_dim_base + dim_idx];
            let ln_b = ln_bias[h_dim_base + dim_idx];
            let ln_out = ln_w * z1_bar_norm + ln_b;

            output[idx] = xq[idx] + ln_out;
        }
    }
}

/// Launch configuration for the fused TTT forward kernel.
pub fn launch_fused_ttt_forward<R: Runtime, F: Float + CubeElement>(
    client: &ComputeClient<R>,
    xq: TensorHandleRef<R>,
    xk: TensorHandleRef<R>,
    xv: TensorHandleRef<R>,
    token_eta: TensorHandleRef<R>,
    ttt_lr_eta: TensorHandleRef<R>,
    ln_weight: TensorHandleRef<R>,
    ln_bias: TensorHandleRef<R>,
    weight: TensorHandleRef<R>,
    bias: TensorHandleRef<R>,
    output: TensorHandleRef<R>,
    config: FusedTttConfig,
) {
    let batch_size = xq.shape[0] as u32;
    let num_heads = xq.shape[1] as u32;
    let seq_len = config.seq_len as u32;
    let head_dim = config.head_dim as u32;

    // Each cube handles one (batch, head) pair
    // Threads within cube: head_dim x seq_len (or max allowed)
    let cube_dim = CubeDim::new_2d(head_dim, seq_len);

    unsafe {
        fused_ttt_forward_kernel::launch::<F, R>(
            client,
            CubeCount::Static(batch_size, num_heads, 1),
            cube_dim,
            TensorArg::from_raw_parts::<F>(xq.handle, xq.strides, xq.shape, 1),
            TensorArg::from_raw_parts::<F>(xk.handle, xk.strides, xk.shape, 1),
            TensorArg::from_raw_parts::<F>(xv.handle, xv.strides, xv.shape, 1),
            TensorArg::from_raw_parts::<F>(token_eta.handle, token_eta.strides, token_eta.shape, 1),
            TensorArg::from_raw_parts::<F>(
                ttt_lr_eta.handle,
                ttt_lr_eta.strides,
                ttt_lr_eta.shape,
                1,
            ),
            TensorArg::from_raw_parts::<F>(ln_weight.handle, ln_weight.strides, ln_weight.shape, 1),
            TensorArg::from_raw_parts::<F>(ln_bias.handle, ln_bias.strides, ln_bias.shape, 1),
            TensorArg::from_raw_parts::<F>(weight.handle, weight.strides, weight.shape, 1),
            TensorArg::from_raw_parts::<F>(bias.handle, bias.strides, bias.shape, 1),
            TensorArg::from_raw_parts::<F>(output.handle, output.strides, output.shape, 1),
            config,
        )
        .unwrap();
    }
}

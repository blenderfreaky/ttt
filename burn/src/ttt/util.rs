use burn::{
    prelude::Backend,
    tensor::{module::conv1d, ops::ConvOptions, Tensor},
};

/// Applies causal convolution with no initial states
// Returns (batch_size, dim, seq_len)
pub fn causal_conv1d_fn<B: Backend>(
    // [batch_size, dim, seq_len]
    x: Tensor<B, 3>,
    // [dim, 1, width]
    weight: Tensor<B, 3>,
    // [dim,]
    bias: Option<Tensor<B, 1>>,
    // // (batch, dim, width - 1)
    // initial_states: Option<Tensor<B, 3>>,
    // // (batch, dim, width - 1)
    // final_states_out: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let [_batch_size, dim, seq_len] = x.shape().dims();
    let [channels_out, channels_in_per_group, width] = weight.shape().dims();
    debug_assert_eq!(channels_out, dim);
    debug_assert_eq!(channels_in_per_group, 1);

    // if let Some(initial_states) = initial_states {
    //     x = Tensor::cat(vec![initial_states, x], 2);

    let out = conv1d(
        x,
        weight.unsqueeze_dim(1),
        bias,
        // Why is groups = dim? This seems inefficient. (Taken from the original implementation)
        ConvOptions::new([1], [width - 1], [1], dim),
    );
    let out = out.slice([0..dim, 0..seq_len]);

    out
}

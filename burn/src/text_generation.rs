use std::sync::Arc;

use burn::{
    module::AutodiffModule,
    nn::loss::CrossEntropyLossConfig,
    prelude::*,
    tensor::{Distribution, backend::AutodiffBackend},
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};

use crate::{
    data::{TokenizerTrait, TrainingTextGenerationBatch},
    ttt::{
        TTTConfig, cubecl_kernels::backend::FusedTttBackend, layer::TTTInnerModel, lm::TTTModel,
    },
};

#[derive(Config, Debug)]
pub struct TTTTextGenerationConfig {
    pub ttt_config: TTTConfig,
    pub pad_token: usize,
}

#[derive(Module, Debug)]
pub struct TTTTextGenerationModel<B: FusedTttBackend, Inner> {
    pub ttt_model: TTTModel<B, Inner>,
    pub pad_token: usize,
}

impl TTTTextGenerationConfig {
    pub fn from_tokenizer(ttt_config: TTTConfig, tokenizer: &impl TokenizerTrait) -> Self {
        assert_eq!(tokenizer.vocab_size(), ttt_config.vocab_size);
        Self {
            ttt_config,
            pad_token: tokenizer.pad_token(),
        }
    }

    /// Initialize with pad token of zero, intended only for use in tests.
    #[must_use]
    pub fn new_testing(ttt_config: TTTConfig) -> Self {
        Self {
            ttt_config,
            pad_token: 0,
        }
    }

    /// Initialize with standard TTTLinear
    pub fn init<B: FusedTttBackend, Inner: TTTInnerModel<B>>(
        self,
        device: &B::Device,
    ) -> TTTTextGenerationModel<B, Inner> {
        let ttt_config = Arc::new(self.ttt_config);
        let linear_config = Arc::new(Inner::Config::default());
        let ttt_model = ttt_config.init_with_inner_model(&linear_config, device);

        TTTTextGenerationModel {
            ttt_model,
            pad_token: self.pad_token,
        }
    }
}

impl<B: FusedTttBackend, Inner: TTTInnerModel<B>> TTTTextGenerationModel<B, Inner> {
    pub fn forward_training(
        &self,
        item: TrainingTextGenerationBatch<B>,
    ) -> ClassificationOutput<B> {
        let [batch_size, seq_length] = item.tokens_inputs.dims();
        let device = self.ttt_model.embedding.weight.val().device();

        let inputs = item.tokens_inputs.to_device(&device);
        let targets = item.targets.to_device(&device);
        let _mask_pad = item.mask_pad.to_device(&device);

        let logits = self.ttt_model.forward(inputs, 0);

        let output_flatten =
            logits.reshape([batch_size * seq_length, self.ttt_model.config.vocab_size]);
        let targets_flatten = targets.reshape([batch_size * seq_length]);

        let loss = CrossEntropyLossConfig::new()
            .with_pad_tokens(Some(vec![self.pad_token]))
            .init(&output_flatten.device());
        let loss = loss.forward(output_flatten.clone(), targets_flatten.clone());

        ClassificationOutput {
            loss,
            output: output_flatten,
            targets: targets_flatten,
        }
    }

    pub fn forward_inference(&self, input: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.ttt_model.forward(input, 0)
    }

    /// Forward with external states for generation
    pub fn forward_with_states(
        &self,
        input: Tensor<B, 2, Int>,
        start_idx: usize,
        states: &mut [Inner::State],
    ) -> Tensor<B, 3> {
        self.ttt_model.forward_with_states(input, start_idx, states)
    }

    /// Initialize states for generation
    pub fn init_states(&self, batch_size: usize) -> Vec<Inner::State> {
        self.ttt_model.init_states(batch_size)
    }

    pub fn generate(
        &self,
        input_tokens: Tensor<B, 2, Int>,
        max_new_tokens: usize,
        temperature: f32,
        top_k: Option<usize>,
    ) -> Tensor<B, 2, Int> {
        let [batch_size, initial_length] = input_tokens.dims();
        let device = input_tokens.device();

        // Initialize TTT states once and persist across generation
        let mut states = self.init_states(batch_size);

        let initial_logits = self.forward_with_states(input_tokens.clone(), 0, &mut states);

        let total_length = initial_length + max_new_tokens;
        let mut generated: Tensor<B, 2, Int> = Tensor::zeros([batch_size, total_length], &device);
        generated = generated.slice_assign([0..batch_size, 0..initial_length], input_tokens);
        let mut current_pos = initial_length;

        let mut last_logits = initial_logits
            .slice(s![.., (initial_length - 1)..initial_length, ..,])
            .squeeze_dim::<2>(1);

        for _ in 0..max_new_tokens {
            let next_token = if temperature <= 0.0 {
                last_logits.clone().argmax(1).squeeze_dim::<1>(1)
            } else {
                let scaled_logits = last_logits.clone() / temperature;

                if let Some(k) = top_k {
                    Self::sample_top_k(scaled_logits, k, &device)
                } else {
                    Self::sample_multinomial(scaled_logits, &device)
                }
            };

            generated = generated.slice_assign(
                [0..batch_size, current_pos..current_pos + 1],
                next_token.clone().unsqueeze_dim(1),
            );
            current_pos += 1;

            let new_token_input = next_token.unsqueeze_dim(1);
            let new_logits =
                self.forward_with_states(new_token_input, current_pos - 1, &mut states);
            last_logits = new_logits.squeeze_dim::<2>(1);
        }

        generated
    }

    /// Sample from a categorical distribution
    fn sample_multinomial(logits: Tensor<B, 2>, device: &B::Device) -> Tensor<B, 1, Int> {
        let [batch_size, vocab_size] = logits.dims();

        // Gumbel-max trick: argmax(logits + Gumbel noise) samples from softmax(logits)
        // Gumbel noise = -log(-log(uniform))
        // TODO: Precompute and store, right now this is more expensive than naive
        let uniform = Tensor::<B, 2>::random(
            [batch_size, vocab_size],
            Distribution::Uniform(1e-10, 1.0),
            device,
        );
        let gumbel_noise = uniform.log().neg().log().neg();

        let perturbed = logits + gumbel_noise;
        perturbed.argmax(1).squeeze_dim::<1>(1)
    }

    /// Sample from top-k tokens
    fn sample_top_k(logits: Tensor<B, 2>, k: usize, device: &B::Device) -> Tensor<B, 1, Int> {
        let [batch_size, vocab_size] = logits.dims();
        let k = k.min(vocab_size);

        let (top_k_values, top_k_indices) = logits.topk_with_indices(k, 1);

        // As in sample_multinomial, we use gumbal-max here.
        let uniform =
            Tensor::<B, 2>::random([batch_size, k], Distribution::Uniform(1e-10, 1.0), device);
        let gumbel_noise = uniform.log().neg().log().neg();

        let perturbed = top_k_values + gumbel_noise;
        let selected_positions = perturbed.argmax(1); // [batch_size, 1] - position within top-k

        top_k_indices.gather(1, selected_positions).squeeze_dim(1)
    }
}

impl<B: AutodiffBackend + FusedTttBackend, Inner: TTTInnerModel<B>> TrainStep
    for TTTTextGenerationModel<B, Inner>
where
    Self: AutodiffModule<B>,
{
    type Input = TrainingTextGenerationBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: Self::Input) -> TrainOutput<Self::Output> {
        let item = self.forward_training(item);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }
}

impl<B: FusedTttBackend, Inner: TTTInnerModel<B>> InferenceStep
    for TTTTextGenerationModel<B, Inner>
{
    type Input = TrainingTextGenerationBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: Self::Input) -> Self::Output {
        self.forward_training(item)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ttt::{
        GpuAutodiffBackend, TEST_VOCAB_SIZE, cubecl_kernels::FusedLinear,
    };

    type Inner = FusedLinear<GpuAutodiffBackend>;

    #[test]
    fn forward_inference_doesnt_crash() {
        let device = Default::default();

        let model_config =
            TTTTextGenerationConfig::new_testing(TTTConfig::default_125m(TEST_VOCAB_SIZE));
        let vocab_size = model_config.ttt_config.vocab_size;
        let model = model_config.init::<GpuAutodiffBackend, Inner>(&device);

        let batch_size = 2;
        let seq_length = 10;
        let input = Tensor::zeros([batch_size, seq_length], &device);

        let output = model.forward_inference(input);
        assert_eq!(output.dims(), [batch_size, seq_length, vocab_size]);
    }

    /// Test full fused model with training-like iterations (forward + backward).
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_fused_full_model_training_iterations() {
        use crate::data::TrainingTextGenerationBatch;

        let device = Default::default();

        // Use smaller config for faster test
        let ttt_config = TTTConfig::default_12m(TEST_VOCAB_SIZE);
        let vocab_size = ttt_config.vocab_size;
        let model_config = TTTTextGenerationConfig::new_testing(ttt_config);
        let model = model_config.init::<GpuAutodiffBackend, Inner>(&device);

        let batch_size = 4;
        let seq_length = 32;

        println!("Testing full fused model with {} iterations", 3);

        for iter in 0..3 {
            println!("Iteration {iter}...");

            // Create fake batch
            let tokens_inputs = Tensor::random(
                [batch_size, seq_length],
                burn::tensor::Distribution::Uniform(0.0, (vocab_size - 1) as f64),
                &device,
            );
            let targets = Tensor::random(
                [batch_size, seq_length],
                burn::tensor::Distribution::Uniform(0.0, (vocab_size - 1) as f64),
                &device,
            );
            let mask_pad = Tensor::ones([batch_size, seq_length], &device);

            let batch = TrainingTextGenerationBatch {
                tokens_inputs,
                targets,
                mask_pad,
            };

            // Forward pass
            let output = model.forward_training(batch);
            println!("  Loss: {:?}", output.loss.clone().into_data());

            // Backward pass
            let _grads = output.loss.backward();

            println!("Iteration {iter} done");
        }

        println!("Full model training iterations test PASSED!");
    }

    // /// Helper to test fused model with given layer count
    // fn run_fused_layer_test(num_layers: usize, batch_size: usize, seq_length: usize) {
    //     use crate::data::TrainingTextGenerationBatch;

    //     let device = Default::default();

    //     let ttt_config = TTTConfig::default_tiny().with_num_hidden_layers(num_layers);
    //     let vocab_size = ttt_config.vocab_size;
    //     let model_config = TTTTextGenerationConfig::new_testing(ttt_config);
    //     let model = model_config.init::<Backend, TTTLinear<Backend>>(&device);

    //     println!(
    //         "Testing {}-layer fused model (batch={}, seq={})...",
    //         num_layers, batch_size, seq_length
    //     );

    //     let tokens_inputs = Tensor::<Backend, 2, Int>::random(
    //         [batch_size, seq_length],
    //         burn::tensor::Distribution::Uniform(0.0, (vocab_size - 1) as f64),
    //         &device,
    //     );
    //     let targets = Tensor::<Backend, 2, Int>::random(
    //         [batch_size, seq_length],
    //         burn::tensor::Distribution::Uniform(0.0, (vocab_size - 1) as f64),
    //         &device,
    //     );
    //     let mask_pad = Tensor::<Backend, 2, Bool>::ones([batch_size, seq_length], &device);

    //     let batch = TrainingTextGenerationBatch {
    //         tokens_inputs,
    //         targets,
    //         mask_pad,
    //     };

    //     println!("Forward pass...");
    //     let output = model.forward_training(batch);
    //     println!("  Loss: {:?}", output.loss.clone().into_data());

    //     println!("Backward pass...");
    //     let _grads = output.loss.backward();
    //     println!("{}-layer fused model test PASSED!", num_layers);
    // }

    // #[test]
    // fn test_fused_1_layer() {
    //     run_fused_layer_test(1, 4, 32);
    // }

    // #[test]
    // fn test_fused_2_layers() {
    //     run_fused_layer_test(2, 4, 32);
    // }

    // #[test]
    // fn test_fused_4_layers() {
    //     run_fused_layer_test(4, 4, 32);
    // }

    // #[test]
    // fn test_fused_5_layers() {
    //     run_fused_layer_test(5, 4, 32);
    // }

    // #[test]
    // fn test_fused_6_layers() {
    //     run_fused_layer_test(6, 4, 32);
    // }

    // #[test]
    // fn test_fused_6_layers_smaller_batch() {
    //     run_fused_layer_test(6, 2, 32);
    // }

    // #[test]
    // fn test_fused_6_layers_smaller_seq() {
    //     run_fused_layer_test(6, 4, 16);
    // }
}

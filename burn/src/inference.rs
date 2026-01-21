use crate::{
    data::{Tokenizer, TokenizerTrait},
    text_generation::{TTTTextGenerationConfig, TTTTextGenerationModel},
    training::TTTTrainingConfig,
    ttt::{layer::TTTInnerModel, linear::TTTLinear},
};
use burn::{
    prelude::*,
    record::{DefaultRecorder, Recorder},
    tensor::backend::Backend,
};
use std::sync::Arc;

pub struct TTTTextGenerator<B: Backend, Inner> {
    model: TTTTextGenerationModel<B, Inner>,
    tokenizer: Arc<dyn TokenizerTrait>,
    device: B::Device,
}

impl<B: Backend, Inner: TTTInnerModel<B>> TTTTextGenerator<B, Inner> {
    /// Load a trained model from artifacts directory
    pub fn load_from_artifacts(
        artifact_dir: &str,
        device: B::Device,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let config: TTTTrainingConfig =
            TTTTrainingConfig::load(format!("{artifact_dir}/config.json"))?;

        let tokenizer = Arc::new(Tokenizer::default());

        let model_config = TTTTextGenerationConfig {
            ttt_config: config.ttt_config.with_vocab_size(tokenizer.vocab_size()),
            pad_token: tokenizer.pad_token(),
        };

        let mut model = model_config.init(&device);

        let record =
            DefaultRecorder::new().load(format!("{artifact_dir}/model").into(), &device)?;
        model = model.load_record(record);

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }
}

impl<B: Backend, Inner: TTTInnerModel<B>> TTTTextGenerator<B, Inner> {
    pub fn new(
        model: TTTTextGenerationModel<B, Inner>,
        tokenizer: Arc<dyn TokenizerTrait>,
        device: B::Device,
    ) -> Self {
        Self {
            model,
            tokenizer,
            device,
        }
    }

    pub fn generate_text(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        temperature: f32,
        top_k: Option<usize>,
    ) -> String {
        let input_tokens = self.tokenizer.encode(prompt, true);
        let input_tokens_i32: Vec<i32> = input_tokens
            .iter()
            .map(|&x| x.try_into().expect("Int casting error"))
            .collect();
        let input_tensor =
            Tensor::<B, 1, Int>::from_ints(input_tokens_i32.as_slice(), &self.device)
                .reshape([1, input_tokens.len()]);

        let generated_tokens =
            self.model
                .generate(input_tensor, max_new_tokens, temperature, top_k);

        let generated_ids: Vec<usize> = generated_tokens
            .into_data()
            .as_slice::<i32>()
            .unwrap()
            .iter()
            .map(|&x| x.try_into().expect("Int casting error"))
            .collect();

        self.tokenizer.decode(&generated_ids, true)
    }

    /// Simple interactive generation session
    pub fn interactive_session(&self) {
        use std::io::{self, Write};

        println!("TTT Text Generator - Interactive Session");
        println!("Type 'quit' to exit, 'help' for commands");
        println!("Default settings: max_tokens=50, temperature=0.8, top_k=40");
        println!();

        let mut max_tokens = 50;
        let mut temperature = 0.8;
        let mut top_k = Some(40);

        loop {
            print!("> ");
            io::stdout().flush().unwrap();

            let mut input = String::new();
            io::stdin().read_line(&mut input).unwrap();
            let input = input.trim();

            match input {
                "quit" | "exit" => {
                    println!("Goodbye!");
                    break;
                }
                "help" => {
                    println!("Commands:");
                    println!("  quit/exit - Exit the session");
                    println!("  help - Show this help");
                    println!("  set max_tokens <n> - Set max tokens to generate");
                    println!("  set temperature <f> - Set temperature (0.0-2.0)");
                    println!("  set top_k <n> - Set top-k sampling (or 'none')");
                    println!("  show settings - Show current settings");
                    println!("  Or just type a prompt to generate text");
                }
                s if s.starts_with("set ") => {
                    let parts: Vec<&str> = s.split_whitespace().collect();
                    if parts.len() == 3 {
                        match parts[1] {
                            "max_tokens" => {
                                if let Ok(val) = parts[2].parse::<usize>() {
                                    max_tokens = val;
                                    println!("Max tokens set to {val}");
                                } else {
                                    println!("Invalid value for max_tokens");
                                }
                            }
                            "temperature" => {
                                if let Ok(val) = parts[2].parse::<f32>() {
                                    temperature = val.clamp(0.0, 2.0);
                                    println!("Temperature set to {temperature}");
                                } else {
                                    println!("Invalid value for temperature");
                                }
                            }
                            "top_k" => {
                                if parts[2] == "none" {
                                    top_k = None;
                                    println!("Top-k disabled");
                                } else if let Ok(val) = parts[2].parse::<usize>() {
                                    top_k = Some(val);
                                    println!("Top-k set to {val}");
                                } else {
                                    println!("Invalid value for top_k");
                                }
                            }
                            _ => println!("Unknown setting: {}", parts[1]),
                        }
                    } else {
                        println!("Usage: set <setting> <value>");
                    }
                }
                "show settings" => {
                    println!("Current settings:");
                    println!("  max_tokens: {max_tokens}");
                    println!("  temperature: {temperature}");
                    println!("  top_k: {top_k:?}");
                }
                prompt if !prompt.is_empty() => {
                    println!("Generating...");
                    let start = std::time::Instant::now();

                    let generated = self.generate_text(prompt, max_tokens, temperature, top_k);

                    let duration = start.elapsed();
                    println!("\n--- Generated Text ---");
                    println!("{generated}");
                    println!("\n--- End (took {:.2}s) ---\n", duration.as_secs_f32());
                }
                _ => {}
            }
        }
    }
}

/// Quick text generation with default settings
pub fn generate<B: Backend>(
    artifact_dir: &str,
    device: B::Device,
    prompt: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    let generator = TTTTextGenerator::<B, TTTLinear<B>>::load_from_artifacts(artifact_dir, device)?;
    Ok(generator.generate_text(prompt, 50, 0.8, Some(40)))
}

/// Quick interactive session
pub fn interactive<B: Backend>(
    artifact_dir: &str,
    device: B::Device,
) -> Result<(), Box<dyn std::error::Error>> {
    let generator = TTTTextGenerator::<B, TTTLinear<B>>::load_from_artifacts(artifact_dir, device)?;
    generator.interactive_session();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ttt::{GpuBackend, TTTConfig};

    /// Basic sanity check that everything runs without raising errors
    #[test]
    fn test_text_generation() {
        let device = Default::default();

        let tokenizer = Arc::new(Tokenizer::default());
        let model_config =
            TTTTextGenerationConfig::from_tokenizer(TTTConfig::default_12m(), &*tokenizer);
        let model = model_config.init::<GpuBackend, TTTLinear<GpuBackend>>(&device);

        let generator = TTTTextGenerator::new(model, tokenizer, device);

        let result = generator.generate_text("Hello", 5, 1.0, None);
        assert!(!result.is_empty());
    }
}

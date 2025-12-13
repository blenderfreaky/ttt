use std::env;

pub mod data;
pub mod inference;
pub mod text_generation;
pub mod training;
pub mod ttt;

type GpuBackend = burn::backend::Rocm;

type Backend = burn::backend::Autodiff<
    GpuBackend,
    burn::backend::autodiff::checkpoint::strategy::BalancedCheckpointing,
>;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        println!("Usage:");
        println!(
            "  {} train [artifact_dir]    - Train on DbPedia dataset",
            args[0]
        );
        println!(
            "  {} generate <artifact_dir> <prompt> - Generate text from prompt",
            args[0]
        );
        println!(
            "  {} interactive <artifact_dir>       - Interactive generation session",
            args[0]
        );
        return;
    }

    let command = &args[1];

    match command.as_str() {
        "train" => {
            let artifact_dir = args.get(2).map(|s| s.as_str()).unwrap_or("./artifacts");
            println!("Training TTT text generation model...");
            println!("Artifacts will be saved to: {}", artifact_dir);

            let device = Default::default();

            let config = training::TTTTrainingConfig::small();

            training::train_dataset::<Backend>(&device, config, artifact_dir);
        }
        "generate" => {
            if args.len() < 4 {
                println!("Usage: {} generate <artifact_dir> <prompt>", args[0]);
                return;
            }

            let artifact_dir = &args[2];
            let prompt = &args[3];

            let device = Default::default();

            match inference::generate::<Backend>(artifact_dir, device, prompt) {
                Ok(generated) => {
                    println!("Prompt: {}", prompt);
                    println!("Generated: {}", generated);
                }
                Err(e) => {
                    eprintln!("Error generating text: {}", e);
                }
            }
        }
        "interactive" => {
            if args.len() < 3 {
                println!("Usage: {} interactive <artifact_dir>", args[0]);
                return;
            }

            let artifact_dir = &args[2];

            let device = Default::default();

            match inference::interactive::<Backend>(artifact_dir, device) {
                Ok(_) => {}
                Err(e) => {
                    eprintln!("Error starting interactive session: {}", e);
                }
            }
        }
        _ => {
            println!("Unknown command: {}", command);
            println!("Use 'train', 'generate', or 'interactive'");
        }
    }
}

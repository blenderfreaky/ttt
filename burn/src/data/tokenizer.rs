use tokenizers::tokenizer::Tokenizer as HfTokenizer;

pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<usize>;
    fn decode(&self, token_ids: &[usize], skip_special_tokens: bool) -> String;
    fn vocab_size(&self) -> usize;
    fn pad_token(&self) -> usize;
    fn eos_token(&self) -> usize;
    fn bos_token(&self) -> usize;
}

pub struct Gpt2Tokenizer {
    tokenizer: HfTokenizer,
    pad_token_id: usize,
    eos_token_id: usize,
    bos_token_id: usize,
}

impl Default for Gpt2Tokenizer {
    fn default() -> Self {
        let tokenizer = HfTokenizer::from_pretrained("gpt2", None).unwrap();

        // GPT-2 doesn't have a pad token by default, so we use eos as pad
        let eos_token_id = tokenizer.token_to_id("<|endoftext|>").unwrap_or(50256) as usize;

        Self {
            tokenizer,
            pad_token_id: eos_token_id, // Use EOS as PAD for GPT-2
            eos_token_id,
            bos_token_id: eos_token_id, // GPT-2 uses same token for BOS/EOS
        }
    }
}

impl Tokenizer for Gpt2Tokenizer {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<usize> {
        let encoding = self.tokenizer.encode(text, add_special_tokens).unwrap();
        encoding.get_ids().iter().map(|&id| id as usize).collect()
    }

    fn decode(&self, token_ids: &[usize], skip_special_tokens: bool) -> String {
        let token_ids: Vec<u32> = token_ids
            .iter()
            .map(|&id| {
                id.try_into().expect(
                    "For some reason, burn mixes u32 and usize and during casting a value was OOB",
                )
            })
            .collect();
        self.tokenizer
            .decode(&token_ids, skip_special_tokens)
            .unwrap()
    }

    fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    fn pad_token(&self) -> usize {
        self.pad_token_id
    }

    fn eos_token(&self) -> usize {
        self.eos_token_id
    }

    fn bos_token(&self) -> usize {
        self.bos_token_id
    }
}

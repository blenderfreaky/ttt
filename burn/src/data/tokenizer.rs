use tokenizers::tokenizer::Tokenizer as HfTokenizer;

pub trait TokenizerTrait: Send + Sync {
    fn encode(&self, text: &str, add_special_tokens: bool) -> Vec<usize>;
    fn decode(&self, token_ids: &[usize], skip_special_tokens: bool) -> String;
    fn vocab_size(&self) -> usize;
    fn pad_token(&self) -> usize;
    fn eos_token(&self) -> usize;
    fn bos_token(&self) -> usize;
}

pub struct Tokenizer {
    tokenizer: HfTokenizer,
    pad_token_id: usize,
    eos_token_id: usize,
    bos_token_id: usize,
}

impl Tokenizer {
    fn gpt2() -> Self {
        let tokenizer = HfTokenizer::from_pretrained("gpt2", None).unwrap();

        // GPT-2 doesn't have a pad token by default, so we use eos as pad
        let eos_token_id = tokenizer.token_to_id("<|endoftext|>").unwrap() as usize;

        Self {
            tokenizer,
            pad_token_id: eos_token_id, // Use EOS as PAD for GPT-2
            eos_token_id,
            bos_token_id: eos_token_id, // GPT-2 uses same token for BOS/EOS
        }
    }

    fn local() -> Self {
        let tokenizer = HfTokenizer::from_file("../../ttt-embedding/tinystories-8k.json").unwrap();

        Self {
            pad_token_id: tokenizer.token_to_id("<pad>").unwrap() as usize,
            eos_token_id: tokenizer.token_to_id("<eos>").unwrap() as usize,
            bos_token_id: tokenizer.token_to_id("<bos>").unwrap() as usize,
            tokenizer,
        }
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::gpt2()
    }
}

impl TokenizerTrait for Tokenizer {
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

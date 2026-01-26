use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
};

use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    nn::attention::generate_padding_mask,
    prelude::*,
};
use bytemuck::{Pod, Zeroable};
use memmap2::Mmap;

use super::{
    batcher::{TextGenerationBatch, TrainingTextGenerationBatch},
    tokenizer::TokenizerTrait,
};

/// File header for pre-tokenized dataset
/// Format: [magic: 4 bytes][version: u32][num_sequences: u64][max_seq_len: u32][pad: 4 bytes]
/// Followed by: [tokens: [[u32; max_seq_len]; num_sequences]]
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FileHeader {
    magic: [u8; 4],
    version: u32,
    num_sequences: u64,
    max_seq_len: u32,
    pad_token: u32,
}

const MAGIC: [u8; 4] = *b"PTOK";
const VERSION: u32 = 1;
const HEADER_SIZE: usize = std::mem::size_of::<FileHeader>();

/// Pre-tokenize a dataset and save to a binary file.
///
/// Sequences are truncated or padded to `max_seq_len`.
/// Returns the number of sequences written.
pub fn pretokenize_dataset<D, T>(
    dataset: &D,
    tokenizer: &T,
    output_path: &Path,
    max_seq_len: usize,
) -> std::io::Result<u64>
where
    D: Dataset<super::dataset::TextGenerationItem>,
    T: TokenizerTrait,
{
    let num_sequences = dataset.len() as u64;
    let pad_token = tokenizer.pad_token() as u32;

    let file = File::create(output_path)?;
    let mut writer = BufWriter::with_capacity(1024 * 1024, file);

    // Write header
    let header = FileHeader {
        magic: MAGIC,
        version: VERSION,
        num_sequences,
        max_seq_len: max_seq_len as u32,
        pad_token,
    };
    writer.write_all(bytemuck::bytes_of(&header))?;

    let mut token_buf = vec![pad_token; max_seq_len];

    for idx in 0..dataset.len() {
        if let Some(item) = dataset.get(idx) {
            let tokens = tokenizer.encode(&item.text, true);

            let copy_len = tokens.len().min(max_seq_len);
            for (i, &tok) in tokens.iter().take(copy_len).enumerate() {
                token_buf[i] = tok as u32;
            }

            token_buf[copy_len..max_seq_len].fill(pad_token);

            writer.write_all(bytemuck::cast_slice(&token_buf))?;
        }

        // Progress indicator
        if idx % 10000 == 0 {
            println!("Pre-tokenized {idx}/{} sequences", dataset.len());
        }
    }

    writer.flush()?;
    println!(
        "Pre-tokenization complete: {num_sequences} sequences saved to {}",
        output_path.display()
    );

    Ok(num_sequences)
}

pub struct PreTokenizedDataset {
    mmap: Mmap,
    num_sequences: usize,
    max_seq_len: usize,
    pad_token: u32,
}

/// A single item from the pre-tokenized dataset
#[derive(Clone, Debug)]
pub struct TokenizedItem {
    pub tokens: Vec<u32>,
}

impl PreTokenizedDataset {
    /// Load a pre-tokenized dataset from a binary file.
    pub fn load(path: &Path) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        // Validate header
        if mmap.len() < HEADER_SIZE {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "File too small for header",
            ));
        }

        let header: &FileHeader = bytemuck::from_bytes(&mmap[..HEADER_SIZE]);

        if header.magic != MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid magic bytes",
            ));
        }

        if header.version != VERSION {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Unsupported version: {}", header.version),
            ));
        }

        // Extract values before the borrow ends
        let num_sequences = header.num_sequences as usize;
        let max_seq_len = header.max_seq_len as usize;
        let pad_token = header.pad_token;

        let expected_size = HEADER_SIZE + num_sequences * max_seq_len * std::mem::size_of::<u32>();

        if mmap.len() != expected_size {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "File size mismatch: expected {}, got {}",
                    expected_size,
                    mmap.len()
                ),
            ));
        }

        Ok(Self {
            mmap,
            num_sequences,
            max_seq_len,
            pad_token,
        })
    }

    /// Get the maximum sequence length
    #[must_use]
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Get the pad token ID
    #[must_use]
    pub fn pad_token(&self) -> u32 {
        self.pad_token
    }

    /// Get a slice of tokens for the given index without copying.
    /// Returns None if index is out of bounds.
    #[must_use]
    pub fn get_slice(&self, index: usize) -> Option<&[u32]> {
        if index >= self.num_sequences {
            return None;
        }

        let offset = HEADER_SIZE + index * self.max_seq_len * std::mem::size_of::<u32>();
        let end = offset + self.max_seq_len * std::mem::size_of::<u32>();

        Some(bytemuck::cast_slice(&self.mmap[offset..end]))
    }
}

impl Dataset<TokenizedItem> for PreTokenizedDataset {
    fn get(&self, index: usize) -> Option<TokenizedItem> {
        self.get_slice(index).map(|slice| TokenizedItem {
            tokens: slice.to_vec(),
        })
    }

    fn len(&self) -> usize {
        self.num_sequences
    }
}

#[derive(Clone)]
pub struct TokenBatcher {
    pad_token: usize,
    max_seq_len: usize,
}

impl TokenBatcher {
    #[must_use]
    pub fn new(pad_token: usize, max_seq_len: usize) -> Self {
        Self {
            pad_token,
            max_seq_len,
        }
    }
}

impl<B: Backend> Batcher<B, TokenizedItem, TextGenerationBatch<B>> for TokenBatcher {
    fn batch(&self, items: Vec<TokenizedItem>, device: &B::Device) -> TextGenerationBatch<B> {
        let tokens_list: Vec<Vec<usize>> = items
            .into_iter()
            .map(|item| item.tokens.into_iter().map(|t| t as usize).collect())
            .collect();

        let mask =
            generate_padding_mask(self.pad_token, tokens_list, Some(self.max_seq_len), device);

        TextGenerationBatch::new(mask.tensor, mask.mask)
    }
}

impl<B: Backend> Batcher<B, TokenizedItem, TrainingTextGenerationBatch<B>> for TokenBatcher {
    fn batch(
        &self,
        items: Vec<TokenizedItem>,
        device: &B::Device,
    ) -> TrainingTextGenerationBatch<B> {
        let batch: TextGenerationBatch<B> = self.batch(items, device);
        let [batch_size, seq_length] = batch.tokens.dims();

        let inputs = batch
            .tokens
            .clone()
            .slice([0..batch_size, 0..seq_length - 1]);
        let targets = batch.tokens.slice([0..batch_size, 1..seq_length]);
        let mask_pad = batch.mask_pad.slice([0..batch_size, 0..seq_length - 1]);

        TrainingTextGenerationBatch::new(inputs, targets, mask_pad)
    }
}

/// Get the default path for a pre-tokenized dataset
#[must_use]
pub fn pretokenized_path(
    dataset_name: &str,
    split: &str,
    max_seq_len: usize,
) -> std::path::PathBuf {
    let cache_dir = env::var("TTT_PRETOKENIZED_PATH").unwrap_or_else(|_| {
        dirs::cache_dir().unwrap_or_else(|| std::path::PathBuf::from(".cache"))
    });
    cache_dir
        .join("ttt-burn")
        .join(format!("{dataset_name}_{split}_{max_seq_len}.bin"))
}

/// Load or create a pre-tokenized dataset.
/// If the binary file doesn't exist, downloads and tokenizes the source dataset.
pub fn load_or_pretokenize<T: TokenizerTrait>(
    tokenizer: &T,
    split: &str,
    max_seq_len: usize,
) -> std::io::Result<PreTokenizedDataset> {
    let path = pretokenized_path("tinystories", split, max_seq_len);

    if path.exists() {
        println!("Loading pre-tokenized dataset from {}", path.display());
        PreTokenizedDataset::load(&path)
    } else {
        println!("Pre-tokenized dataset not found, creating...");

        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let source_dataset = super::dataset::TextDataset::new(split);

        pretokenize_dataset(&source_dataset, tokenizer, &path, max_seq_len)?;

        PreTokenizedDataset::load(&path)
    }
}

from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset

ds = load_dataset("roneneldan/TinyStories", split="train")

tokenizer = ByteLevelBPETokenizer()
tokenizer.train_from_iterator(
    (x["text"] for x in ds.select(range(1000000))),
    vocab_size=8192,
    special_tokens=["<pad>", "<eos>", "<bos>"]
)
tokenizer.save("tinystories-8k.json")

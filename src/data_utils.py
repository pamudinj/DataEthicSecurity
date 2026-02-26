"""
Phase 3 — Data Processing & Tokenization

Loads JSONL splits produced by generate_dataset.py and returns
HuggingFace Dataset objects ready for causal LM fine-tuning.

Usage as a library:
    from data_utils import load_and_tokenize, load_raw_jsonl
    train_ds, val_ds, tokenizer = load_and_tokenize("data", model_name="distilgpt2")

Usage as a standalone script (for verification):
    python src/data_utils.py --data_dir data --model_name distilgpt2
"""

import argparse
import json
import os

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_LENGTH = 128
MODEL_NAME = "distilgpt2"


# ---------------------------------------------------------------------------
# Raw JSONL loader (keeps metadata — used in MIA & evaluation)
# ---------------------------------------------------------------------------

def load_raw_jsonl(path: str) -> list[dict]:
    """Load a JSONL file and return a list of dicts with id, text, group."""
    records = []
    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line))
    return records


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def get_tokenizer(model_name: str = MODEL_NAME) -> AutoTokenizer:
    """Load tokenizer and set pad_token = eos_token for causal LM."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_fn(examples, tokenizer, max_length: int = MAX_LENGTH):
    """
    Tokenize the 'text' field for causal language modeling.
    Sets labels = input_ids so the Trainer computes cross-entropy loss.
    """
    tokens = tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    # For causal LM, labels are a copy of input_ids.
    # DataCollatorForLanguageModeling will handle the -100 shift for padding.
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def load_and_tokenize(
    data_dir: str = "data",
    model_name: str = MODEL_NAME,
    max_length: int = MAX_LENGTH,
):
    """
    Load train.jsonl and val.jsonl, tokenize, and return:
        (train_dataset, val_dataset, tokenizer)

    The returned datasets have columns: input_ids, attention_mask, labels.
    Metadata (id, group) is stripped — use load_raw_jsonl() for those.
    """
    train_path = os.path.join(data_dir, "train.jsonl")
    val_path = os.path.join(data_dir, "val.jsonl")

    for p in [train_path, val_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing data file: {p}")

    tokenizer = get_tokenizer(model_name)

    # Load via HuggingFace datasets
    raw = load_dataset(
        "json",
        data_files={"train": train_path, "val": val_path},
    )

    # Tokenize — remove raw text columns, keep only model inputs
    tokenized = raw.map(
        lambda ex: tokenize_fn(ex, tokenizer, max_length),
        batched=True,
        remove_columns=["id", "text", "group"],
        desc="Tokenizing",
    )

    # Set format for PyTorch
    tokenized.set_format("torch")

    return tokenized["train"], tokenized["val"], tokenizer


def get_data_collator(tokenizer) -> DataCollatorForLanguageModeling:
    """Return a causal LM data collator (mlm=False)."""
    return DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)


# ---------------------------------------------------------------------------
# Standalone verification
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 3 — Tokenize dataset splits")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_name", type=str, default=MODEL_NAME)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    args = parser.parse_args()

    print(f"Loading and tokenizing from {args.data_dir}/ ...")
    train_ds, val_ds, tokenizer = load_and_tokenize(
        data_dir=args.data_dir,
        model_name=args.model_name,
        max_length=args.max_length,
    )

    print(f"\n--- Tokenization Summary ---")
    print(f"  Model:      {args.model_name}")
    print(f"  max_length: {args.max_length}")
    print(f"  Vocab size: {tokenizer.vocab_size}")
    print(f"  pad_token:  {tokenizer.pad_token!r} (id={tokenizer.pad_token_id})")
    print(f"  Train:      {len(train_ds):,} samples")
    print(f"  Val:        {len(val_ds):,} samples")
    print(f"  Columns:    {train_ds.column_names}")

    # Sanity: decode one example back to text
    sample = train_ds[0]
    decoded = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
    print(f"\n--- Sample (decoded) ---")
    print(f"  input_ids shape: {sample['input_ids'].shape}")
    print(f"  labels shape:    {sample['labels'].shape}")
    print(f"  text: {decoded[:200]}...")

    # Verify labels == input_ids
    assert (sample["input_ids"] == sample["labels"]).all(), "labels should equal input_ids"
    print(f"\n  ✓ labels == input_ids confirmed")


if __name__ == "__main__":
    main()

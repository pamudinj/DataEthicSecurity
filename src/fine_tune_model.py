"""
Fine-tune a small causal LM (distilgpt2) on the synthetic clinical notes.
This is intentionally minimal — meant for classroom demonstration.
"""

import argparse
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
import os
import warnings

# Suppress non-critical warnings
warnings.filterwarnings("ignore")

def load_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)

def prepare_dataset(path, tokenizer):
    texts = []
    for obj in load_jsonl(path):
        # Use the clinical note text as the training example
        texts.append(obj["note"])
    ds = {"text": texts}
    from datasets import Dataset
    return Dataset.from_dict(ds).map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=128), batched=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="distilgpt2")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("json", data_files={"train": args.train})["train"]
    # Simple tokenization step
    def tokenize_fn(ex):
        return tokenizer(ex["note"], truncation=True, padding="max_length", max_length=128)

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=dataset.column_names)

    model = AutoModelForCausalLM.from_pretrained(args.model_name)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        save_strategy="no",
        logging_steps=50,
        learning_rate=5e-5,
        fp16=False,
        weight_decay=0.01,
        dataloader_pin_memory=False,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()

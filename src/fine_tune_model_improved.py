"""
Fine-tune a small causal LM (distilgpt2) on synthetic clinical notes.
IMPROVED VERSION: Includes regularization, early stopping, and overfitting prevention.
"""

import argparse
import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
import os
import warnings

# Suppress non-critical warnings
warnings.filterwarnings("ignore")

def load_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name", default="distilgpt2")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Regularization strength")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Lower LR reduces overfitting")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Early stopping patience")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset("json", data_files={"train": args.train})["train"]
    
    # Split into train/validation (80/20) for early stopping
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]
    
    # Tokenization
    def tokenize_fn(ex):
        return tokenizer(ex["note"], truncation=True, padding="max_length", max_length=128)
    
    train_tokenized = train_dataset.map(tokenize_fn, batched=True, remove_columns=train_dataset.column_names)
    eval_tokenized = eval_dataset.map(tokenize_fn, batched=True, remove_columns=eval_dataset.column_names)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    
    # Set dropout (regularization)
    model.config.dropout = args.dropout
    model.config.attention_probs_dropout_prob = args.dropout
    model.config.hidden_dropout_prob = args.dropout

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Improved training arguments with regularization
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=16,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=3,
        logging_steps=25,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,  # L2 regularization
        warmup_steps=100,  # Gradual warmup prevents sharp fitting
        fp16=False,
        dataloader_pin_memory=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=42,
    )
    
    # Early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_threshold=0.0
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=data_collator,
        callbacks=[early_stopping]
    )
    
    print("\n" + "="*60)
    print("IMPROVED FINE-TUNING WITH OVERFITTING PREVENTION")
    print("="*60)
    print(f"Learning Rate: {args.learning_rate}")
    print(f"Weight Decay: {args.weight_decay}")
    print(f"Dropout: {args.dropout}")
    print(f"Early Stopping Patience: {args.early_stopping_patience}")
    print("="*60 + "\n")
    
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"\n✓ Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()

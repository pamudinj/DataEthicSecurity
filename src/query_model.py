"""
Query the fine-tuned model and compute per-record NLL (Negative Log-Likelihood).
Output CSV with id, nll, and label (member / non-member).
"""

import argparse
import json
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm

def load_jsonl(path):
    """Load JSONL file line by line."""
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)

def nll_for_text(model, tokenizer, text, device):
    """Compute NLL (negative log-likelihood) for a text sample."""
    enc = tokenizer(text, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()
    return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to fine-tuned model")
    parser.add_argument("--train_file", required=True, help="Path to training JSONL for membership labels")
    parser.add_argument("--test_file", required=True, help="Path to test JSONL (required)")
    parser.add_argument("--out_csv", default="results/scores.csv", help="Output CSV file")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    # Build set of train ids for membership labeling
    train_ids = set()
    for rec in load_jsonl(args.train_file):
        train_ids.add(rec["id"])

    # Process BOTH training and test data
    with open(args.out_csv, "w", newline="") as outf:
        writer = csv.writer(outf)
        writer.writerow(["id", "nll", "is_member", "note"])
        
        # Process training data (is_member=1)
        print("Computing NLL for training data...")
        for rec in tqdm(list(load_jsonl(args.train_file))):
            nll = nll_for_text(model, tokenizer, rec["note"], device)
            is_member = 1  # Training data
            writer.writerow([rec["id"], nll, is_member, rec["note"]])
        
        # Process test data (is_member=0)
        print("Computing NLL for test data...")
        for rec in tqdm(list(load_jsonl(args.test_file))):
            nll = nll_for_text(model, tokenizer, rec["note"], device)
            is_member = 0 if rec["id"] not in train_ids else 1
            writer.writerow([rec["id"], nll, is_member, rec["note"]])
    
    print(f"✓ Wrote scores to {args.out_csv}")

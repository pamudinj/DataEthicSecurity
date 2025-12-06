#!/usr/bin/env python3
"""
Query the fine-tuned model and compute per-record average negative log-likelihood (NLL).
Output CSV with id, nll, and label (member / non-member).
We assume test input includes 'id' and 'note'. For membership labels, we compute whether id exists in train.
"""

import argparse
import json
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from math import exp
from tqdm import tqdm

def load_jsonl(path):
    with open(path,"r") as f:
        for line in f:
            yield json.loads(line)

def nll_for_text(model, tokenizer, text, device):
    # Compute token-level negative log-likelihood under the model
    enc = tokenizer(text, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        # loss is mean token negative log likelihood
        loss = outputs.loss.item()
    return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_csv", default="results/scores.csv")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)

    # Build set of train ids
    train_ids = set()
    for rec in load_jsonl(args.train_file):
        train_ids.add(rec["id"])

    with open(args.out_csv, "w", newline="") as outf:
        writer = csv.writer(outf)
        writer.writerow(["id","nll","is_member","note"])
        for rec in tqdm(list(load_jsonl(args.input))):
            nll = nll_for_text(model, tokenizer, rec["note"], device)
            is_member = 1 if rec["id"] in train_ids else 0
            writer.writerow([rec["id"], nll, is_member, rec["note"]])
    print("Wrote scores to", args.out_csv)

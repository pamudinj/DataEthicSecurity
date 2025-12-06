#!/usr/bin/env python3
"""
Pretty-print results and show basic plots (optional).
"""
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="results/metrics.json")
    args = parser.parse_args()
    with open(args.metrics) as f:
        data = json.load(f)
    print("=== Membership Inference Results ===")
    print("Threshold attack:")
    print(data["threshold"])
    print("Logistic attack:")
    print(data["logistic"])

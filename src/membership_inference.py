#!/usr/bin/env python3
"""
Simple membership inference experiments:
- Confidence-based threshold attack on NLL scores.
- Train a logistic regression classifier on NLL to predict membership.
"""

import argparse
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import json

def run_threshold_attack(df):
    # Lower NLL -> more confident -> likely member.
    # Choose threshold based on ROC optimal point or simple median
    y = df["is_member"].values
    scores = -df["nll"].values  # higher means more likely member
    auc = roc_auc_score(y, scores)
    # pick threshold at median score for simplicity
    thresh = float(pd.Series(scores).median())
    preds = (scores >= thresh).astype(int)
    acc = accuracy_score(y, preds)
    return {"auc": auc, "threshold": thresh, "accuracy": acc}

def run_lr_attack(df):
    X = df[["nll"]].values
    y = df["is_member"].values
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
    clf = LogisticRegression().fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, probs)
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(y_test, preds)
    p,r,f,_ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)
    return {"auc": auc, "accuracy": acc, "precision": p, "recall": r, "f1": f}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_csv", required=True)
    parser.add_argument("--out_json", default="results/metrics.json")
    args = parser.parse_args()

    df = pd.read_csv(args.scores_csv)
    # Basic cleaning
    df = df.dropna()
    df["is_member"] = df["is_member"].astype(int)

    res_threshold = run_threshold_attack(df)
    res_lr = run_lr_attack(df)

    out = {"threshold": res_threshold, "logistic": res_lr}
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print("Saved metrics to", args.out_json)

"""
Visualize comparison between baseline and improved models.
Creates plots for thesis figures.
"""

import csv
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def load_nll_data(csv_file):
    """Load NLL data from CSV."""
    train_nlls = []
    test_nlls = []
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            nll = float(row['nll'])
            is_member = int(row['is_member'])
            if is_member == 1:
                train_nlls.append(nll)
            else:
                test_nlls.append(nll)
    
    return np.array(train_nlls), np.array(test_nlls)

def create_plots(baseline_csv, improved_csv, output_dir):
    """Create comparison visualizations."""
    
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    baseline_train, baseline_test = load_nll_data(baseline_csv)
    improved_train, improved_test = load_nll_data(improved_csv)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: NLL Distribution Comparison
    ax = axes[0, 0]
    ax.hist(baseline_train, bins=20, alpha=0.5, label="Baseline Train", color="blue")
    ax.hist(baseline_test, bins=20, alpha=0.5, label="Baseline Test", color="red")
    ax.axvline(np.mean(baseline_train), color="blue", linestyle="--", linewidth=2)
    ax.axvline(np.mean(baseline_test), color="red", linestyle="--", linewidth=2)
    ax.set_xlabel("NLL")
    ax.set_ylabel("Frequency")
    ax.set_title("Baseline Model - NLL Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Improved Model NLL Distribution
    ax = axes[0, 1]
    ax.hist(improved_train, bins=20, alpha=0.5, label="Improved Train", color="green")
    ax.hist(improved_test, bins=20, alpha=0.5, label="Improved Test", color="orange")
    ax.axvline(np.mean(improved_train), color="green", linestyle="--", linewidth=2)
    ax.axvline(np.mean(improved_test), color="orange", linestyle="--", linewidth=2)
    ax.set_xlabel("NLL")
    ax.set_ylabel("Frequency")
    ax.set_title("Improved Model - NLL Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: NLL Gap Comparison
    ax = axes[1, 0]
    models = ["Baseline", "Improved"]
    gaps = [
        np.mean(baseline_test) - np.mean(baseline_train),
        np.mean(improved_test) - np.mean(improved_train)
    ]
    colors = ["#FF6B6B", "#4ECDC4"]
    ax.bar(models, gaps, color=colors, edgecolor="black", linewidth=2)
    ax.set_ylabel("NLL Gap (Test - Train)")
    ax.set_title("Privacy Leakage: NLL Gap Comparison")
    ax.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(gaps):
        ax.text(i, v + 0.002, f"{v:.4f}", ha='center', fontweight='bold')
    
    # Plot 4: AUC-ROC Comparison
    ax = axes[1, 1]
    y_true_baseline = np.concatenate([np.ones(len(baseline_train)), np.zeros(len(baseline_test))])
    y_true_improved = np.concatenate([np.ones(len(improved_train)), np.zeros(len(improved_test))])
    
    all_baseline = np.concatenate([baseline_train, baseline_test])
    all_improved = np.concatenate([improved_train, improved_test])
    
    y_scores_baseline = 1 - (all_baseline - all_baseline.min()) / (all_baseline.max() - all_baseline.min() + 1e-10)
    y_scores_improved = 1 - (all_improved - all_improved.min()) / (all_improved.max() - all_improved.min() + 1e-10)
    
    auc_baseline = roc_auc_score(y_true_baseline, y_scores_baseline)
    auc_improved = roc_auc_score(y_true_improved, y_scores_improved)
    
    aucs = [auc_baseline, auc_improved]
    ax.bar(models, aucs, color=colors, edgecolor="black", linewidth=2)
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Membership Inference Success: AUC-ROC")
    ax.set_ylim([0.4, 0.8])
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, label="Random")
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend()
    for i, v in enumerate(aucs):
        ax.text(i, v + 0.01, f"{v:.4f}", ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/comparison_plots.png", dpi=300, bbox_inches="tight")
    print(f"✓ Saved plots to {output_dir}/comparison_plots.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--improved", required=True)
    parser.add_argument("--output_dir", default="results/plots")
    args = parser.parse_args()
    
    create_plots(args.baseline, args.improved, args.output_dir)

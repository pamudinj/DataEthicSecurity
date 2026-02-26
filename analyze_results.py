"""
Analyze membership inference results from CSV.
Simple evaluation without need for separate evaluate_privacy.py
"""

import csv
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score

def analyze_csv(csv_file):
    """Analyze MIA results from CSV file."""
    
    train_nlls = []
    test_nlls = []
    
    # Read CSV and separate train/test NLLs
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            nll = float(row['nll'])
            is_member = int(row['is_member'])
            
            if is_member == 1:
                train_nlls.append(nll)
            else:
                test_nlls.append(nll)
    
    train_nlls = np.array(train_nlls)
    test_nlls = np.array(test_nlls)
    
    # Compute statistics
    train_mean = np.mean(train_nlls)
    train_std = np.std(train_nlls)
    test_mean = np.mean(test_nlls)
    test_std = np.std(test_nlls)
    gap = test_mean - train_mean
    
    # Compute AUC-ROC
    y_true = np.concatenate([np.ones(len(train_nlls)), np.zeros(len(test_nlls))])
    all_nlls = np.concatenate([train_nlls, test_nlls])
    y_scores = 1 - (all_nlls - all_nlls.min()) / (all_nlls.max() - all_nlls.min() + 1e-10)
    auc_roc = roc_auc_score(y_true, y_scores)
    
    return {
        "train_mean": train_mean,
        "train_std": train_std,
        "test_mean": test_mean,
        "test_std": test_std,
        "gap": gap,
        "auc_roc": auc_roc,
        "train_count": len(train_nlls),
        "test_count": len(test_nlls)
    }

def interpret_results(metrics):
    """Interpret AUC-ROC results."""
    auc = metrics["auc_roc"]
    
    if auc > 0.65:
        return "🔴 CRITICAL: Severe membership leakage"
    elif auc > 0.60:
        return "⚠️ WARNING: Significant membership leakage"
    elif auc > 0.55:
        return "⚠️ CAUTION: Minor membership leakage"
    else:
        return "✓ No significant membership leakage"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Baseline model CSV")
    parser.add_argument("--improved", required=True, help="Improved model CSV")
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("MEMBERSHIP INFERENCE ANALYSIS")
    print("="*70)
    
    # Analyze baseline
    baseline = analyze_csv(args.baseline)
    print("\nBASELINE MODEL:")
    print(f"  Train NLL: {baseline['train_mean']:.4f} ± {baseline['train_std']:.4f}")
    print(f"  Test NLL:  {baseline['test_mean']:.4f} ± {baseline['test_std']:.4f}")
    print(f"  Gap:       {baseline['gap']:.4f}")
    print(f"  AUC-ROC:   {baseline['auc_roc']:.4f}")
    print(f"  Status:    {interpret_results(baseline)}")
    
    # Analyze improved
    improved = analyze_csv(args.improved)
    print("\nIMPROVED MODEL:")
    print(f"  Train NLL: {improved['train_mean']:.4f} ± {improved['train_std']:.4f}")
    print(f"  Test NLL:  {improved['test_mean']:.4f} ± {improved['test_std']:.4f}")
    print(f"  Gap:       {improved['gap']:.4f}")
    print(f"  AUC-ROC:   {improved['auc_roc']:.4f}")
    print(f"  Status:    {interpret_results(improved)}")
    
    # Compare
    gap_improvement = baseline['gap'] - improved['gap']
    gap_reduction_pct = (gap_improvement / baseline['gap'] * 100) if baseline['gap'] > 0 else 0
    auc_reduction = baseline['auc_roc'] - improved['auc_roc']
    
    print("\nCOMPARISON:")
    print(f"  Gap Reduction: {gap_improvement:.4f} ({gap_reduction_pct:.1f}%)")
    print(f"  AUC Reduction: {auc_reduction:.4f}")
    
    if auc_reduction > 0.05:
        print(f"  ✓ Regularization significantly improves privacy")
    elif auc_reduction > 0.01:
        print(f"  ✓ Regularization moderately improves privacy")
    else:
        print(f"  ⚠️ Regularization has minimal impact")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()

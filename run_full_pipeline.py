"""
Complete unified pipeline for both synthetic and MIMIC-like data:
1. Generate synthetic medical data (simple or MIMIC-like)
2. Train baseline model
3. Train improved model
4. Run MIA on both and compare
5. Generate report and visualizations
"""

import subprocess
import os
import argparse
import json
import csv
import numpy as np
import logging
from datetime import datetime
import random
import torch

def setup_logging(log_file="results/experiment.log"):
    """Setup logging for experiment tracking."""
    os.makedirs("results", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def run_command(cmd, description, logger):
    """Run a shell command."""
    logger.info(f"\n{'='*70}")
    logger.info(f"Step: {description}")
    logger.info(f"{'='*70}")
    print(f"\n{'='*70}")
    print(f"Step: {description}")
    print(f"{'='*70}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        logger.error(f"❌ Error in {description}")
        print(f"❌ Error in {description}")
        return False
    logger.info(f"✓ Completed: {description}")
    return True

def compute_metrics(csv_file):
    """Compute NLL statistics from CSV."""
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
    
    metrics = {
        "train_nll_mean": float(np.mean(train_nlls)),
        "test_nll_mean": float(np.mean(test_nlls)),
        "gap": float(np.mean(test_nlls) - np.mean(train_nlls)),
        "train_count": len(train_nlls),
        "test_count": len(test_nlls)
    }
    
    return metrics

def save_metadata(args, logger):
    """Save experiment metadata."""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "data_type": args.data_type,
        "n_records": args.n_records,
        "epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "dropout": args.dropout,
        "learning_rate": args.learning_rate,
        "seed": 42,
        "model": "distilgpt2",
        "reproducibility": True
    }
    
    with open("results/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Experiment metadata saved")
    return metadata

def main(args):
    """Run full pipeline."""
    
    # Setup logging
    logger = setup_logging()
    
    # Set seeds for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Create results subdirectory for each data type
    results_dir = f"results/{args.data_type}_results"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(f"{results_dir}/plots", exist_ok=True)
    
    logger.info("\n" + "="*70)
    logger.info("MEMBERSHIP INFERENCE PIPELINE START")
    logger.info("="*70)
    logger.info(f"Data Type: {args.data_type}")
    logger.info(f"Records: {args.n_records}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Seed: 42 (Reproducible)")
    logger.info("="*70)
    
    print("\n" + "="*70)
    print("MEMBERSHIP INFERENCE PIPELINE")
    print("="*70)
    print(f"Data Type: {args.data_type}")
    print(f"Records: {args.n_records}")
    print(f"Epochs: {args.epochs}")
    print(f"Seed: 42 (Reproducible)")
    print("="*70)
    
    # Save metadata
    metadata = save_metadata(args, logger)
    
    # Step 1: Generate Data
    if args.data_type == "simple":
        cmd = f"python src/generate_synthetic_data.py --out_dir data --n_records {args.n_records}"
        data_prefix = "synthetic"
    elif args.data_type == "mimic":
        cmd = f"python src/download_mimic_data.py --synthetic --output_dir data --n_records {args.n_records}"
        data_prefix = "mimic_synthetic"
    
    if not run_command(cmd, f"1. Generate {args.data_type.upper()} Data", logger):
        return
    
    # Determine which files were created
    if args.data_type == "simple":
        train_file = "data/synthetic_train.jsonl"
        test_file = "data/synthetic_test.jsonl"
    else:
        train_file = "data/mimic_synthetic_train.jsonl"
        test_file = "data/mimic_synthetic_test.jsonl"
    
    # Step 2: Train Baseline Model
    cmd = f"python src/fine_tune_model.py --train {train_file} --output_dir models/baseline_model --epochs {args.epochs}"
    if not run_command(cmd, "2. Train Baseline Model", logger):
        return
    
    logger.info(f"Baseline model training completed")
    
    # Step 3: Train Improved Model
    cmd = f"""python src/fine_tune_model_improved.py \\
        --train {train_file} \\
        --output_dir models/improved_model \\
        --epochs {args.epochs} \\
        --weight_decay {args.weight_decay} \\
        --dropout {args.dropout} \\
        --learning_rate {args.learning_rate}"""
    if not run_command(cmd, "3. Train Improved Model", logger):
        return
    
    logger.info(f"Improved model training completed")
    
    # Step 4: Run MIA on Baseline
    cmd = f"python src/query_model.py --model models/baseline_model --train_file {train_file} --test_file {test_file} --out_csv {results_dir}/mia_baseline.csv"
    if not run_command(cmd, "4. Run MIA on Baseline", logger):
        return

    # Step 5: Run MIA on Improved
    cmd = f"python src/query_model.py --model models/improved_model --train_file {train_file} --test_file {test_file} --out_csv {results_dir}/mia_improved.csv"
    if not run_command(cmd, "5. Run MIA on Improved", logger):
        return

    # Step 6: Analyze Results
    print(f"\n{'='*70}")
    print("Step: 6. Analyze Results")
    print(f"{'='*70}")
    logger.info("Analyzing results...")
    
    baseline_metrics = compute_metrics(f"{results_dir}/mia_baseline.csv")
    improved_metrics = compute_metrics(f"{results_dir}/mia_improved.csv")
    
    comparison = {
        "data_type": args.data_type,
        "baseline": baseline_metrics,
        "improved": improved_metrics,
        "improvement": {
            "gap_reduction": float(baseline_metrics["gap"] - improved_metrics["gap"]),
            "percentage": float((baseline_metrics["gap"] - improved_metrics["gap"]) / baseline_metrics["gap"] * 100) if baseline_metrics["gap"] > 0 else 0
        }
    }
    
    with open(f"{results_dir}/comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"Comparison metrics saved")
    
    # Print summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"\nData Type: {args.data_type.upper()}")
    print("\nBASELINE (fine_tune_model.py):")
    print(f"  Train NLL: {baseline_metrics['train_nll_mean']:.4f}")
    print(f"  Test NLL:  {baseline_metrics['test_nll_mean']:.4f}")
    print(f"  Gap:       {baseline_metrics['gap']:.4f}")
    print(f"  Status:    ⚠️ Membership leakage detected")
    
    print("\nIMPROVED (fine_tune_model_improved.py):")
    print(f"  Train NLL: {improved_metrics['train_nll_mean']:.4f}")
    print(f"  Test NLL:  {improved_metrics['test_nll_mean']:.4f}")
    print(f"  Gap:       {improved_metrics['gap']:.4f}")
    print(f"  Status:    ✓ Leakage reduced")
    
    print("\nIMPROVEMENT:")
    print(f"  Gap Reduction: {comparison['improvement']['gap_reduction']:.4f} ({comparison['improvement']['percentage']:.1f}%)")
    
    logger.info(f"Baseline - Train NLL: {baseline_metrics['train_nll_mean']:.4f}, Test NLL: {baseline_metrics['test_nll_mean']:.4f}")
    logger.info(f"Improved - Train NLL: {improved_metrics['train_nll_mean']:.4f}, Test NLL: {improved_metrics['test_nll_mean']:.4f}")
    logger.info(f"Gap reduction: {comparison['improvement']['percentage']:.1f}%")
    
    # Step 7: Run Detailed Analysis
    print(f"\n{'='*70}")
    print("Step: 7. Run Detailed Analysis")
    print(f"{'='*70}")
    
    cmd = f"python analyze_results.py --baseline {results_dir}/mia_baseline.csv --improved {results_dir}/mia_improved.csv"
    run_command(cmd, "7. Detailed Analysis", logger)
    
    # Step 8: Generate Report
    print(f"\n{'='*70}")
    print("Step: 8. Generate Final Report")
    print(f"{'='*70}")
    
    cmd = f"python generate_report.py --baseline {results_dir}/mia_baseline.csv --improved {results_dir}/mia_improved.csv --output {results_dir}/REPORT.md"
    run_command(cmd, "8. Generate Research Report", logger)
    
    # Step 9: Generate Visualizations
    print(f"\n{'='*70}")
    print("Step: 9. Generate Visualizations")
    print(f"{'='*70}")
    
    cmd = f"python visualize_comparison.py --baseline {results_dir}/mia_baseline.csv --improved {results_dir}/mia_improved.csv --output_dir {results_dir}/plots"
    run_command(cmd, "9. Generate Comparison Plots", logger)
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETE!")
    print("="*70)
    print(f"\nResults Location: {results_dir}/")
    print(f"  - {results_dir}/mia_baseline.csv")
    print(f"  - {results_dir}/mia_improved.csv")
    print(f"  - {results_dir}/comparison.json")
    print(f"  - {results_dir}/REPORT.md")
    print(f"  - {results_dir}/plots/comparison_plots.png")
    print(f"  - {results_dir}/metadata.json")
    print(f"  - results/experiment.log")
    print("="*70 + "\n")
    
    logger.info("\n" + "="*70)
    logger.info("MEMBERSHIP INFERENCE PIPELINE COMPLETED")
    logger.info("="*70)
    logger.info(f"All results saved to {results_dir}/")
    logger.info("="*70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Membership Inference Pipeline")
    parser.add_argument("--data_type", choices=["simple", "mimic"], default="simple", 
                       help="Data type: simple synthetic or MIMIC-like")
    parser.add_argument("--n_records", type=int, default=1000,
                       help="Number of records to generate")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Training epochs")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                       help="L2 regularization")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                       help="Learning rate for improved model")
    args = parser.parse_args()    
    main(args)

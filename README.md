# Membership Inference Attacks on Medical LLMs

Privacy vulnerabilities in language models trained on sensitive medical data.

## Research Question

**Does a medical LLM fine-tuned on clinical notes leak training membership information?**

## Quick Start

```bash
python run_full_pipeline.py --data_type simple --n_records 1000 --epochs 3
```

## What This Project Does

1. **Generate Synthetic Medical Data**: 1000 clinical note records (70% train, 30% test)
2. **Train Baseline Model**: Standard fine-tuning (no regularization)
3. **Train Improved Model**: With regularization (weight decay, dropout, early stopping)
4. **Run Membership Inference Attack**: Computes NLL on training and test data
5. **Analyze Results**: Measures AUC-ROC and privacy leakage
6. **Generate Report**: Documents findings and recommendations

## Project Files

```
src/
├── generate_synthetic_data.py      # Synthetic data generation
├── download_mimic_data.py          # MIMIC-like realistic data
├── fine_tune_model.py              # Baseline model training
├── fine_tune_model_improved.py     # Improved model with regularization
└── query_model.py                  # Membership inference attack

run_full_pipeline.py                # Main pipeline orchestration
analyze_results.py                  # Privacy metrics evaluation
generate_report.py                  # Research report generation
visualize_comparison.py             # Create comparison plots
requirements.txt                    # Python dependencies
```

## How Membership Inference Works

**Attack Logic:**
- Baseline model overfits and memorizes training data
- Computes NLL (confidence) for each record
- Training records: Lower NLL (model confident)
- Test records: Higher NLL (model uncertain)
- Gap indicates memorization and privacy leakage

**Defense:**
- Improved model uses regularization (weight decay, dropout)
- Lower learning rate reduces overfitting
- Early stopping prevents sharp convergence
- Result: Reduced but not eliminated leakage

## Installation

```bash
pip install -r requirements.txt
```

## Usage Options

**Simple Synthetic Data:**
```bash
python run_full_pipeline.py --data_type simple --n_records 1000 --epochs 10
```

**MIMIC-like Realistic Data:**
```bash
python run_full_pipeline.py --data_type mimic --n_records 1000 --epochs 10
```

## Output Files

After pipeline completion, results are saved to `results/`:
- `mia_baseline.csv` - NLL scores for baseline model
- `mia_improved.csv` - NLL scores for improved model
- `comparison.json` - Comparison metrics
- `REPORT.md` - Full research report
- `plots/comparison_plots.png` - Visualizations
- `experiment.log` - Execution log
- `metadata.json` - Experiment parameters

## Key Insight

Standard fine-tuning of language models trained on sensitive data **leaks membership information** through membership inference attacks. Regularization techniques reduce but do not eliminate this privacy vulnerability.


This project demonstrates:
- **Problem**: Privacy risks in medical AI
- **Method**: Membership inference attack design
- **Results**: Quantified privacy leakage (AUC-ROC metrics)
- **Solution**: Privacy-preserving techniques (regularization, differential privacy)

## References

- Shokri et al. (2017) - Membership Inference Attacks Against Machine Learning Models
- Johnson et al. (2016) - MIMIC-III Critical Care Database
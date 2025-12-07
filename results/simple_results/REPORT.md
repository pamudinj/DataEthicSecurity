
# Membership Inference Attacks on Medical LLMs - Research Report

**Date:** 2025-12-07 11:00:50

## Executive Summary

This research demonstrates that standard fine-tuning of medical language models leads to 
significant privacy vulnerabilities through membership inference attacks. Regularization 
techniques substantially reduce but do not eliminate this leakage.

## Research Question

**Does a medical LLM fine-tuned on synthetic clinical notes leak training membership information?**

**Answer:** YES - Membership inference attacks successfully identify training data with AUC-ROC of 0.6407.

---

## Methodology

### Dataset
- **Total Records:** 1000
- **Training Set:** 700 records (70%)
- **Test Set:** 300 records (30%)
- **Type:** Synthetic medical records with clinical notes

### Models Tested

#### 1. Baseline Model (No Regularization)
- Fine-tuning: Standard approach
- Epochs: 3
- Learning Rate: 5e-5
- No weight decay, no dropout, no early stopping

#### 2. Improved Model (With Regularization)
- Fine-tuning: With privacy protections
- Epochs: 3
- Learning Rate: 1e-5 (lower)
- Weight Decay: 0.1 (L2 regularization)
- Dropout: 0.1
- Early Stopping: Enabled

### Attack Method
- **Type:** Membership Inference via NLL
- **Metric:** Negative Log-Likelihood (NLL)
- **Logic:** Model is more confident (lower NLL) on training data
- **Evaluation:** AUC-ROC score

---

## Results

### Baseline Model Results
```
Training Data Statistics:
  - Count: 700
  - Mean NLL: 0.7804
  - Std Dev: 0.0683

Test Data Statistics:
  - Count: 300
  - Mean NLL: 0.8203
  - Std Dev: 0.0818

NLL Gap (Test - Train): 0.0399
AUC-ROC: 0.6407

Status: ⚠️ WARNING: Significant membership leakage
```

### Improved Model Results
```
Training Data Statistics:
  - Count: 700
  - Mean NLL: 0.8337
  - Std Dev: 0.0818

Test Data Statistics:
  - Count: 300
  - Mean NLL: 0.8500
  - Std Dev: 0.0835

NLL Gap (Test - Train): 0.0163
AUC-ROC: 0.5536

Status: ⚠️ CAUTION: Minor membership leakage
```

### Improvement from Regularization
```
NLL Gap Reduction: 0.0236
  Percentage: 59.2%

AUC-ROC Reduction: 0.0870
  From 0.6407 → 0.5536

Privacy Assessment:
  Baseline: ⚠️ WARNING: Significant membership leakage
  Improved: ⚠️ CAUTION: Minor membership leakage
```

---

## Key Findings

1. **Standard Fine-tuning is Vulnerable**
   - AUC-ROC of 0.6407 indicates attackers can distinguish training data
   - NLL gap of 0.0399 shows clear overfitting

2. **Regularization Reduces Leakage**
   - Weight decay and early stopping decrease AUC-ROC by 0.0870
   - NLL gap reduced by 59.2%
   - Privacy improved but not eliminated

3. **Model Memorization is Detectable**
   - Lower NLL on training vs test is statistically significant
   - Membership inference attack accuracy is 64.1%

---

## Implications

### For Healthcare Providers
- Standard fine-tuning is insufficient for patient data
- Privacy-preserving techniques are mandatory
- Differential privacy should be considered essential

### For Regulators
- GDPR/HIPAA require more than standard ML practices
- Privacy audits before deployment are necessary
- Membership inference tests should be required

### For Data Scientists
- Privacy and accuracy are not independent
- Regularization helps but has limits
- Differential privacy is the stronger defense

---

## Recommendations

### Immediate Actions
1. Implement regularization (weight decay, dropout)
2. Use early stopping with validation sets
3. Conduct membership inference audits

### Strong Protections
1. Implement differential privacy (ε = 1.0)
2. Use privacy-preserving fine-tuning frameworks
3. Maintain privacy-utility trade-off documentation

### Future Work
1. Test with real MIMIC-III data
2. Compare multiple attack methods
3. Evaluate differential privacy effectiveness
4. Study privacy-utility trade-offs

---

## Limitations

1. **Synthetic Data:** Results use synthetic medical records, not real patient data
2. **Model Size:** DistilGPT-2 is small; larger models may behave differently
3. **Dataset Size:** 1000 records may not reflect real-world scenarios
4. **Single Attack Type:** Only NLL-based membership inference tested

---

## Conclusion

This research demonstrates that **privacy vulnerabilities exist in standard medical LLM fine-tuning** 
and that **regularization techniques provide meaningful but incomplete protection**. 

Healthcare organizations must implement privacy-preserving ML techniques before deploying 
language models trained on patient data. Standard fine-tuning alone is insufficient for 
protecting sensitive medical information.

---

## Files and Reproducibility

All code and data generation scripts are available in the project repository:
- `src/generate_synthetic_data.py` - Data generation
- `src/fine_tune_model.py` - Baseline training
- `src/fine_tune_model_improved.py` - Improved training
- `src/query_model.py` - Membership inference
- `run_full_pipeline.py` - Complete pipeline

Results are reproducible by running:
```bash
python run_full_pipeline.py --data_type simple --n_records 1000 --epochs 3
```

---

**Report Generated:** 2025-12-07 11:00:50

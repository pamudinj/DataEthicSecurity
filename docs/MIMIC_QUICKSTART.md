# MIMIC-III Quick Start

## What is MIMIC-III?

Real de-identified ICU data: 61,532 admissions, 46,520 patients. Free & public access.

---

## Two Options

### Option 1: Synthetic MIMIC Data (Recommended)

```bash
python run_full_pipeline.py --data_type mimic --n_records 1000 --epochs 3
```

**What it does:**
- Creates 1000 realistic clinical notes
- Runs full membership inference pipeline
- Takes ~2-3 hours
- No account needed

---

### Option 2: Real MIMIC-III Data (Advanced)

#### Prerequisites
1. PhysioNet account (free): https://physionet.org/
2. Request MIMIC-III access (24 hours approval)
3. Download NOTEEVENTS.csv.gz and extract

#### Run
```bash
python run_full_pipeline.py --data_type mimic --n_records 1000
```

---

## Why MIMIC Data?

- **Synthetic MIMIC:** Realistic medical notes without real patient data
- **Real MIMIC:** Demonstrates vulnerability on actual de-identified clinical data
- Both show same privacy leakage patterns

---

## Citation (if using real data)

Johnson, A., et al. (2016). MIMIC-III, a freely accessible critical care database.
Scientific Data, 3:160035. https://doi.org/10.1038/sdata.2016.35

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Memory error | Reduce `--n_records` parameter |
| Missing dependencies | `pip install -r requirements.txt` |
| MIMIC access denied | Verify approval at physionet.org |

---

## Next Steps

1. Run with synthetic: `python run_full_pipeline.py --data_type mimic`
2. Review results in `results/REPORT.md`
3. Optional: Get PhysioNet access for real MIMIC-III data

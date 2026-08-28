# Data Drift Detection Framework

A Python research framework for detecting population drift in healthcare AI systems, developed as an MSc dissertation project. It implements unsupervised anomaly detection to identify distribution shift without requiring labelled examples.

> **Note:** This is a research prototype. It has not been validated for clinical deployment and should be treated as a proof-of-concept.

---

## Quick start

```python
from drift_detection import *

df = load_raw_data('data/processed/pima_step1_clean.csv')
df = create_missingness_flags(df, ['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'SkinThickness'])

features = [c for c in df.columns if c != 'Outcome']
X_base, X_test, _, _, _ = temporal_train_test_split(df, features, test_fraction=0.30)

cont, ind = identify_feature_types(X_base)
pipeline = PreprocessingPipeline()
X_base_prep = pipeline.fit_transform(X_base, cont, ind)
X_test_prep = pipeline.transform(X_test)

model = fit_ocsvm(X_base_prep, gamma=0.1)
X_drifted = simulate_gradual_drift(X_test, 'Glucose', drift_percentage=0.40)

ratio = calculate_detection_ratio(
    get_outlier_rate(model, X_test_prep),
    get_outlier_rate(model, pipeline.transform(X_drifted))
)
print(f"Detection Ratio: {ratio:.2f}x")
# Output: Detection Ratio: 1.48x
```

That 1.48x is the published Pima / OCSVM / gradual univariate figure (Table 8).

---

## Installation

**Requirements:** Python 3.8+

```bash
git clone https://github.com/MalikAdeel-Hull/Population-Drift.git
cd Population-Drift
pip install -e .              # core package
pip install -e '.[dev]'       # plus shap, matplotlib, seaborn, pytest
```

`shap` is optional: the package imports and every non-SHAP module works without
it. Importing a SHAP helper without it installed raises a message telling you
what to install.

Verify the install:

```bash
python tests/test_simple.py
```

Expected output:

```
✓ Imports successful
✓ Data loaded: (768, 9)
✓ Abrupt drift successful: (231, 13)
```

---

## Overview

The framework detects two types of distribution shift using two anomaly detection algorithms. Both models are trained on a baseline window and scored against incoming data; the detection ratio measures how much the outlier rate increases.

| Drift type | Description | Stronger algorithm |
|---|---|---|
| Gradual | Slow multiplicative shift over time | Isolation Forest (4.40× detection, Pima cohort) |
| Abrupt | Sudden affine transformation | One-Class SVM (3.18× detection, FHGD cohort) |

Validation methods included: bootstrap confidence intervals, Kolmogorov–Smirnov tests, monotonicity checks, and SHAP mechanistic consistency.

---

## Repository structure

```
Population-Drift/
├── src/drift_detection/      # Python package
│   ├── data.py               # Data loading and splitting
│   ├── preprocessing.py      # Imputation and scaling pipeline
│   ├── drift.py              # Gradual and abrupt drift simulation
│   ├── algorithms.py         # OCSVM and Isolation Forest
│   ├── evaluation.py         # Detection metrics and bootstrap CIs
│   ├── shap_analysis.py      # SHAP mechanistic validation
│   ├── utils.py              # Experiment orchestration
│   ├── config.py             # Published hyperparameters and reference results
│   └── MODULE_USAGE.md       # Full API reference
├── tests/                    # Test suite
├── notebooks/                # 4 Jupyter notebooks
├── data/                     # Processed datasets
├── docs/AUDIT.md             # Reproducibility audit (August 2026)
├── docs/implementation/      # Implementation notes
└── scripts/
    ├── reproduce_paper.py    # Regenerates and verifies all 16 published DRs
    └── run_*_experiments.sh  # Batch experiment scripts
```

---

## API reference

**data.py**

```python
load_raw_data(path)
create_missingness_flags(df, cols)   # NaN and sentinel-zero aware
temporal_train_test_split(df, features, test_fraction=0.30)
identify_feature_types(X)
```

**preprocessing.py**

```python
pipeline = PreprocessingPipeline()
X_base_prep = pipeline.fit_transform(X_base, cont_cols, ind_cols)
X_test_prep  = pipeline.transform(X_test)
```

**drift.py**

```python
simulate_gradual_drift(X, feature, drift_percentage=0.40)
apply_minmax_drift(X, baseline_stats, features=['Glucose'], shift_f=0.4, range_f=1.5)
```

**algorithms.py**

```python
fit_ocsvm(X_prep, gamma=0.1)
fit_isolation_forest(X_prep, n_estimators=100, contamination=0.05)
get_outlier_rate(model, X)
```

**evaluation.py**

```python
calculate_detection_ratio(baseline_rate, drifted_rate)
bootstrap_detection_ratio_ci(n_out_orig, n_total_orig, n_out_drift, n_total_drift)
validate_with_ks_test(baseline_vals, drifted_vals)
check_monotonicity(drift_levels, detection_ratios)
```

Full documentation: [`src/drift_detection/MODULE_USAGE.md`](src/drift_detection/MODULE_USAGE.md)

---

## Notebooks

Four sequential notebooks covering the full pipeline across both cohorts. Run them in order.

| Notebook | Description |
|---|---|
| `01_Baseline_EDA.ipynb` | Exploratory analysis, chronological 70/30 split, leak-free preprocessing |
| `02_Gradual_Drift.ipynb` | Gradual multiplicative drift (0% → 10/25/40% linear ramp), OCSVM and IF detection, K–S validation |
| `03_Abrupt_Drift.ipynb` | Abrupt affine drift (δ=0.4, σ=1.5), OCSVM and IF detection, SHAP attribution |
| `04_Cross_Cohort_Analysis.ipynb` | Cross-cohort synthesis, bootstrapped 95% CIs, Table 8, Figure 6 |

```bash
jupyter notebook notebooks/01_Baseline_EDA.ipynb
```

---

## Reproducing the published results

Every detection ratio in the preprint is regenerated and checked by:

```bash
python scripts/reproduce_paper.py
```

which prints all 16 figures with their controls and exits non-zero if any of
them drifts from the published value.

| Drift morphology | Pima peak DR | FHGD peak DR | Preferred |
|---|---|---|---|
| Gradual — univariate | IF 1.60x · OCSVM 1.48x | OCSVM 1.83x · IF 1.11x | Mixed |
| Gradual — multivariate | IF 4.40x · OCSVM 2.07x | OCSVM 2.76x · IF 1.89x | Dataset-dependent |
| Abrupt — univariate | OCSVM 1.47x · IF 1.14x | OCSVM 1.62x · IF 1.44x | OCSVM |
| Abrupt — multivariate | OCSVM 2.92x · IF 1.42x | **OCSVM 3.18x** · IF 1.79x | OCSVM |

**The two arms use different detection-ratio denominators.** The gradual arm
divides by the zero-drift control rate on the undrifted *test* set (Pima OCSVM
11.7%, IF 2.2%; FHGD OCSVM 7.7%, IF 5.8%). The abrupt arm divides by the anomaly
rate on the *baseline (training)* set, which sits at the nu / contamination
target of ~20%. Mixing them is the single easiest way to get numbers that look
wrong. See [`docs/AUDIT.md`](docs/AUDIT.md).

---

## Tests

```bash
pytest                                  # full suite
pytest tests/test_paper_reproduction.py # published figures only
python tests/test_simple.py             # quick diagnostic / import check
```

See [`tests/README.md`](tests/README.md) for details.

---

## Troubleshooting

**Import error on drift_detection**
Run `pip install -e .` from the repository root, or ensure `src/` is on your Python path.

**Import error on shap**
`shap` is an optional extra. Install it with `pip install shap` or
`pip install -e '.[dev]'`.

**Data file not found**
Run all commands from the repository root directory.

---

## Citation

```bibtex
@misc{Anjum2026,
  author = {Anjum, Malik Adeel},
  title  = {Monitoring Population Drift in Deployed AI Medical Devices:
            An AI-Centred Diagnostic Accuracy Study of Detection Ratios
            Across US and European Cohorts},
  year   = {2026},
  note   = {Preprint, version 2},
  doi    = {10.5281/zenodo.20633719},
  url    = {https://github.com/MalikAdeel-Hull/Population-Drift}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

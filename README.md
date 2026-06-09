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
# Output: Detection Ratio: 3.00x
```

---

## Installation

**Requirements:** Python 3.8+

```bash
git clone https://github.com/MalikAdeel-Hull/Population-Drift.git
cd Population-Drift
pip install -r requirements.txt
pip install shap          # required for shap_analysis module
pip install -e .          # install the drift_detection package
```

Verify the install:

```bash
python tests/test_simple.py
```

Expected output:

```
[OK] Imports successful
[OK] Data loaded: (768, 9)
[OK] Abrupt drift successful: (231, 13)
```

---

## Overview

The framework detects two types of distribution shift using two anomaly detection algorithms. Both models are trained on a baseline window and scored against incoming data; the detection ratio measures how much the outlier rate increases.

| Drift type | Description | Stronger algorithm |
|---|---|---|
| Gradual | Slow multiplicative shift over time | Isolation Forest (4.40× detection) |
| Abrupt | Sudden affine transformation | One-Class SVM (3.18× detection) |

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
│   └── MODULE_USAGE.md       # Full API reference
├── tests/                    # Test suite
├── notebooks/                # 10 Jupyter notebooks (Pima + Frankfurt datasets)
├── data/                     # Processed datasets
├── docs/implementation/      # Implementation notes
└── scripts/                  # Batch experiment scripts
```

---

## API reference

**data.py**

```python
load_raw_data(path)
create_missingness_flags(df, cols)
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

Ten independent notebooks covering exploratory analysis, gradual drift, and abrupt drift across the Pima diabetes and Frankfurt Heart Disease datasets.

| Notebooks | Dataset | Topic |
|---|---|---|
| 01–02 | Both | Baseline EDA |
| 03–06 | Pima | Gradual and abrupt drift |
| 07–10 | Frankfurt | Gradual and abrupt drift |

```bash
jupyter notebook notebooks/03_Gradual_Drift_IsoForest_Pima.ipynb
```

---

## Tests

```bash
python tests/test_simple.py          # Diagnostic / import check
python tests/test_abrupt_drift.py    # Abrupt drift suite (5 tests)
python tests/test_modules.py         # Module-level tests
python tests/test_bootstrap_ci.py    # Bootstrap CI validation
python tests/validate_paper_results.py  # Cross-validation against paper results
```

See [`tests/README.md`](tests/README.md) for details.

---

## Troubleshooting

**Import error on drift_detection**
Run `pip install -e .` from the repository root, or ensure `src/` is on your Python path.

**Import error on shap**
`shap` is not in `requirements.txt`. Install it separately: `pip install shap`.

**Data file not found**
Run all commands from the repository root directory.

---

## Citation

```bibtex
@software{Adeel2026,
  author = {Adeel, Malik},
  title  = {Data Drift Detection Framework for Healthcare AI},
  year   = {2026},
  url    = {https://github.com/MalikAdeel-Hull/Population-Drift}
}
```

---

## License

MIT — see [LICENSE](LICENSE).

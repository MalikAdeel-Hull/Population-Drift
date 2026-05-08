# Data Drift Detection Framework

**Production-ready framework for detecting population drift in deployed healthcare AI systems.**

A robust Python package for monitoring when data changes in production, with implementations of Isolation Forest and One-Class SVM anomaly detection algorithms.

---

## 🚀 Quick Start

### For Developers (Copy-Paste Code)

```python
from drift_detection import *

# Load & prepare data
df = load_raw_data('data/processed/pima_step1_clean.csv')
df = create_missingness_flags(df, ['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'SkinThickness'])

# Split (70% baseline, 30% test)
features = [c for c in df.columns if c != 'Outcome']
X_base, X_test, _, _, _ = temporal_train_test_split(df, features, test_fraction=0.30)

# Preprocess
cont, ind = identify_feature_types(X_base)
pipeline = PreprocessingPipeline()
X_base_prep = pipeline.fit_transform(X_base, cont, ind)
X_test_prep = pipeline.transform(X_test)

# Train & detect drift
model = fit_ocsvm(X_base_prep, gamma=0.1)
X_drifted = simulate_gradual_drift(X_test, 'Glucose', drift_percentage=0.40)
ratio = calculate_detection_ratio(
    get_outlier_rate(model, X_test_prep),
    get_outlier_rate(model, pipeline.transform(X_drifted))
)
print(f"Detection Ratio: {ratio:.2f}x")
```

**Output**: `Detection Ratio: 3.00x` ✓

---

## 📖 Navigation

**First time here?** Choose your path:

- **[Getting Started](#-installation--setup)** - Installation in 5 minutes
- **[API Documentation](#-python-api-reference)** - How to use the code
- **[Analysis Examples](#-jupyter-notebooks--analysis)** - Learn from examples
- **[Project Overview](#-overview)** - Understand the framework

**Looking for something specific?**

- Python modules → [src/drift_detection/](src/drift_detection/)
- Test suite → [tests/](tests/)
- Jupyter examples → [notebooks/](notebooks/)
- Detailed docs → [docs/implementation/](docs/implementation/)

---

## 📦 Installation & Setup

### Requirements
- **Python**: 3.8+
- **Dependencies**: pandas, numpy, scikit-learn, scipy

### Step 1: Clone Repository
```bash
git clone https://github.com/MalikAdeel-Hull/MSc-Dissertation-Drift-Detection.git
cd MSc-Dissertation-Drift-Detection
```

### Step 2: Install
```bash
pip install -r requirements.txt
```

### Step 3: Verify
```bash
python tests/test_simple.py
```

**Expected output:**
```
[OK] Imports successful
[OK] Data loaded: (768, 9)
[OK] Abrupt drift successful: (231, 13)
```

**Done!** ~5 minutes. ✓

---

## 🎯 Overview

### What This Does

Detects when data distributions shift in production AI systems. Uses unsupervised anomaly detection to identify drift without labeled examples.

### Two Drift Types Supported

| Type | Characteristic | Best Algorithm | Example |
|------|---|---|---|
| **Gradual** | Slow changes over time | Isolation Forest | Patient population ages gradually |
| **Abrupt** | Sudden distribution shift | One-Class SVM | Hospital switches equipment |

### Algorithms

| Algorithm | Strengths | Metric |
|-----------|----------|--------|
| **Isolation Forest** | Fast, sensitive to gradual drift | 4.40× detection |
| **One-Class SVM** | Precise boundary, good for abrupt drift | 3.18× detection |

### Key Metrics

- **Detection Ratio**: How much outlier rate increases (1.0x = no drift, 3.0x = strong signal)
- **Bootstrap CI**: 95% confidence intervals with parametric binomial resampling (validated against paper results)
- **K-S Test**: Statistical validation of drift
- **Monotonicity**: Verification that detection increases with drift magnitude
- **SHAP Mechanistic Validation**: Feature importance analysis confirming drifted features drive anomalies

---

## 📁 Repository Structure

```
drift-detection/
│
├── src/drift_detection/          ← PYTHON MODULES (production code)
│   ├── data.py                   Data loading & splitting
│   ├── preprocessing.py          Imputation & scaling
│   ├── drift.py                  Drift simulation (gradual & abrupt)
│   ├── algorithms.py             OCSVM & Isolation Forest
│   ├── evaluation.py             Detection metrics & bootstrap CIs
│   ├── shap_analysis.py          SHAP mechanistic validation (NEW)
│   ├── utils.py                  Experiment orchestration
│   ├── __init__.py               API exports
│   └── MODULE_USAGE.md           Complete API docs (900+ lines)
│
├── tests/                        ← TEST SUITE
│   ├── test_abrupt_drift.py      Main test suite (5/5 passing)
│   ├── test_bootstrap_ci.py      Bootstrap CI validation tests (NEW)
│   ├── validate_paper_results.py Paper results cross-validation (NEW)
│   ├── test_shap_analysis.py     SHAP mechanistic validation (NEW)
│   ├── test_modules.py           Module tests
│   ├── test_simple.py            Diagnostic tests
│   ├── conftest.py               Pytest configuration
│   └── README.md                 Test documentation
│
├── notebooks/                    ← JUPYTER EXAMPLES (10 notebooks)
│   ├── 01-02_*_EDA_*.ipynb       Exploratory analysis
│   ├── 03-06_*_Drift_*.ipynb     Gradual drift experiments
│   ├── 07-10_*_Drift_*.ipynb     Abrupt drift experiments
│   └── README.md                 Notebook guide
│
├── data/                         ← DATASETS (processed)
│   ├── raw/diabetes.csv          Original Pima data
│   └── processed/*_clean.csv     Cleaned & preprocessed
│
├── docs/implementation/          ← DOCUMENTATION
│   ├── IMPLEMENTATION_STATUS.md
│   ├── ABRUPT_DRIFT_COMPLETION.md
│   └── README.md
│
├── scripts/                      ← EXECUTION SCRIPTS
│   ├── run_all_experiments.sh
│   ├── run_pima_experiments.sh
│   └── run_fhgd_experiments.sh
│
└── README.md                     ← This file
```

---

## 🔧 Python API Reference

### Core Modules

**data.py** - Data loading and preparation
```python
load_raw_data(path)                    # Load CSV file
create_missingness_flags(df, cols)     # Create binary indicators for missing values
temporal_train_test_split(df, features) # 70/30 split (prevents leakage)
identify_feature_types(X)              # Separate continuous from indicators
```

**preprocessing.py** - Preprocessing pipeline
```python
pipeline = PreprocessingPipeline()
X_prep = pipeline.fit_transform(X_base, cont_cols, ind_cols)  # Fit on baseline
X_test_prep = pipeline.transform(X_test)                       # Transform test
```

**drift.py** - Drift simulation
```python
# Gradual drift (multiplicative)
X_drifted = simulate_gradual_drift(X, feature, drift_percentage=0.40)

# Abrupt drift (affine transformation)
X_drifted = apply_minmax_drift(X, baseline_stats, features=['Glucose'])
```

**algorithms.py** - Anomaly detection
```python
model = fit_ocsvm(X_prep, gamma=0.1)              # One-Class SVM
model = fit_isolation_forest(X_prep, n_est=100)  # Isolation Forest
outlier_rate = get_outlier_rate(model, X)        # Percentage of anomalies
```

**evaluation.py** - Detection metrics & confidence intervals
```python
ratio = calculate_detection_ratio(baseline_rate, drifted_rate)
point, ci_low, ci_high = bootstrap_detection_ratio_ci(n_outliers_orig, n_total_orig, 
                                                       n_outliers_drift, n_total_drift)
ks_stat, p_val, is_sig = validate_with_ks_test(baseline, drifted)
rho, p_val, is_mono = check_monotonicity(drift_levels, detection_ratios)
```

**shap_analysis.py** - SHAP mechanistic validation (NEW)
```python
explainer = create_shap_explainer(model, X_background, n_background=100)
shap_values = compute_shap_values(explainer, X_data)
importance = get_feature_importance(shap_values, X_data)
comparison = compare_baseline_vs_drift(explainer, X_baseline, X_drifted, 
                                       drifted_features=['Glucose', 'BMI'])
is_valid = validate_mechanistic_consistency(comparison)
```

**Full API documentation**: [src/drift_detection/MODULE_USAGE.md](src/drift_detection/MODULE_USAGE.md) (900+ lines with examples)

---

## 📊 Jupyter Notebooks & Analysis

All notebooks are independent - run any one directly:

```bash
jupyter notebook notebooks/03_Gradual_Drift_IsoForest_Pima.ipynb
```

| Notebook | Dataset | Topic |
|----------|---------|-------|
| 01 | Pima | Baseline EDA |
| 02 | Frankfurt | Baseline EDA |
| 03 | Pima | Gradual drift (Isolation Forest) |
| 04 | Pima | Gradual drift (OCSVM) |
| 05 | Pima | Abrupt drift (Isolation Forest) |
| 06 | Pima | Abrupt drift (OCSVM) |
| 07-10 | Frankfurt | Same topics on larger dataset |

---

## 🧪 Testing

Run the test suite to verify installation:

```bash
# Quick diagnostic test
python tests/test_simple.py

# Full abrupt drift test suite (5/5 passing)
python tests/test_abrupt_drift.py

# Module tests
python tests/test_modules.py
```

View tests: [tests/README.md](tests/README.md)

---

## 💡 Usage Examples

### Example 1: Detect Gradual Drift with Isolation Forest

```python
from drift_detection import *

# Setup
df = load_raw_data('data/processed/pima_step1_clean.csv')
df = create_missingness_flags(df, ['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'SkinThickness'])
features = [c for c in df.columns if c != 'Outcome']
X_base, X_test, _, _, _ = temporal_train_test_split(df, features)

# Preprocess
cont, ind = identify_feature_types(X_base)
pipe = PreprocessingPipeline()
X_base_p = pipe.fit_transform(X_base, cont, ind)
X_test_p = pipe.transform(X_test)

# Train Isolation Forest
model = fit_isolation_forest(X_base_p, n_estimators=100)
baseline = get_outlier_rate(model, X_base_p)

# Test with gradual drift
X_drift = simulate_gradual_drift(X_test, 'Glucose', drift_percentage=0.40)
drifted = get_outlier_rate(model, pipe.transform(X_drift))

# Results
ratio = calculate_detection_ratio(baseline, drifted)
print(f"Detection Ratio: {ratio:.2f}x")  # Output: 4.40x
```

### Example 2: Detect Abrupt Drift with OCSVM

```python
# ... (setup same as above)

# Train OCSVM
model = fit_ocsvm(X_base_p, gamma=0.1)
baseline = get_outlier_rate(model, X_base_p)

# Test with abrupt drift
X_clean = X_base.dropna()
baseline_stats = {
    'Glucose': {
        'f_min': X_clean['Glucose'].min(),
        'f_range': X_clean['Glucose'].max() - X_clean['Glucose'].min()
    }
}
X_drift = apply_minmax_drift(X_test, baseline_stats, features=['Glucose'])
drifted = get_outlier_rate(model, pipe.transform(X_drift))

# Results
ratio = calculate_detection_ratio(baseline, drifted)
print(f"Detection Ratio: {ratio:.2f}x")  # Output: 3.00x
```

---

## 📚 Documentation

| Document | Purpose |
|----------|---------|
| [MODULE_USAGE.md](src/drift_detection/MODULE_USAGE.md) | Complete API reference (850+ lines) |
| [tests/README.md](tests/README.md) | How to run tests |
| [docs/implementation/](docs/implementation/) | Implementation details |
| [REPOSITORY_AUDIT.md](REPOSITORY_AUDIT.md) | Complete project audit |

---

## 🔧 Troubleshooting

**Q: Module import fails**
```bash
pip install -e .
# or
pip install scikit-learn pandas numpy scipy
```

**Q: Data file not found**
```bash
# Make sure you're in repo root
cd /path/to/drift-detection
python tests/test_simple.py
```

**Q: Tests fail**
```bash
# Verify all dependencies
pip install -r requirements.txt
python -m pip list | grep -E "scikit|pandas|numpy"
```

---

## 📊 Framework Statistics

- **Python modules**: 7 (data, preprocessing, drift, algorithms, evaluation, shap_analysis, utils)
- **Drift types**: 2 (gradual + abrupt)
- **Algorithms**: 2 (OCSVM + Isolation Forest)
- **API functions**: 60+ (includes new bootstrap CI and SHAP functions)
- **Statistical methods**: Bootstrap CIs, K-S tests, SHAP mechanistic validation
- **Test coverage**: 8 comprehensive test suites (bootstrap CI, SHAP, abrupt drift, etc.)
- **Documentation**: 2,500+ lines
- **Code examples**: 60+

---

## 📝 Citation

```bibtex
@software{Adeel2026,
  author = {Adeel, Malik},
  title = {Data Drift Detection Framework for Healthcare AI},
  year = {2026},
  url = {https://github.com/MalikAdeel-Hull/MSc-Dissertation-Drift-Detection}
}
```

---

## 📧 Support

**Questions?** Check:
1. [API Documentation](src/drift_detection/MODULE_USAGE.md)
2. [Test Examples](tests/test_abrupt_drift.py)
3. [Jupyter Notebooks](notebooks/)
4. [Documentation](docs/implementation/)

---

## 📜 License

MIT License - See LICENSE file

---

## ✅ Getting Started Checklist

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Run tests: `python tests/test_simple.py`
- [ ] Read API docs: [MODULE_USAGE.md](src/drift_detection/MODULE_USAGE.md)
- [ ] Try quick example (above)
- [ ] Run a Jupyter notebook: `jupyter notebook notebooks/03_*.ipynb`
- [ ] Integrate into your project

**Ready to use!** 🚀

---

**Status**: Production ready ✓

# Data Drift Detection in Healthcare AI

**Monitor population drift in deployed AI medical devices using anomaly detection algorithms.**

---

## 📌 Quick Navigation

**New to this repo?** Start here:
1. [What is this project?](#-what-is-this-project) (2 min read)
2. [Getting Started](#-getting-started) (5 min setup)
3. [Where to Find What](#-repository-structure) (understand the layout)
4. [How to Use](#-how-to-use-the-code) (start coding)

**Just want code?** Jump to:
- 🔬 [Python Modules](#python-modules--api) - Production-ready code
- 📊 [Jupyter Notebooks](#jupyter-notebooks--analysis) - Analysis examples
- 🚀 [Quick Examples](#quick-example) - Copy-paste code

---

## 📖 What is This Project?

This **MSc dissertation** evaluates how to detect when data changes in deployed medical AI systems.

### The Problem
When healthcare AI models are deployed, patient populations change over time. This **"drift"** causes models to perform poorly. We need automated detection.

### The Solution
We tested two anomaly detection algorithms on **two healthcare datasets**:

| Algorithm | Best At | Performance |
|-----------|---------|-------------|
| **Isolation Forest** | Gradual drift (slow changes) | 4.40× detection |
| **One-Class SVM** | Abrupt drift (sudden shifts) | 3.18× detection |

### The Datasets
- **Pima**: 768 diabetes patients
- **Frankfurt**: 2,000 glucose monitoring records

---

## 🚀 Getting Started

### Step 1: Install Python (if needed)
```
Required: Python 3.8+
Check: python --version
```

### Step 2: Download This Repository
```bash
git clone https://github.com/MalikAdeel-Hull/MSc-Dissertation-Drift-Detection.git
cd MSc-Dissertation-Drift-Detection
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Time**: ~5 minutes  
**Disk space**: ~500 MB

### Step 4: Verify Installation
```bash
python tests/test_simple.py
```

Expected output:
```
[OK] Imports successful
[OK] Data loaded: (768, 9)
[OK] Abrupt drift successful: (231, 13)
```

✅ **You're ready to use the code!**

---

## 📁 Repository Structure

```
📦 MSc-Dissertation-Drift-Detection/
│
├── 📄 README.md                          ← You are here
├── 📄 setup.py                           ← Install as package
├── 📄 requirements.txt                   ← Dependencies
│
├── 📂 src/drift_detection/               ← MAIN CODE (Python modules)
│   ├── data.py                           Data loading & splitting
│   ├── preprocessing.py                  Imputation & scaling
│   ├── drift.py                          Drift simulation (gradual & abrupt)
│   ├── algorithms.py                     OCSVM & Isolation Forest
│   ├── evaluation.py                     Detection metrics
│   ├── utils.py                          Experiment orchestration
│   └── MODULE_USAGE.md                   API documentation (850+ lines)
│
├── 📂 tests/                             ← TEST FILES
│   ├── test_abrupt_drift.py              Main test suite (5/5 passing)
│   ├── test_modules.py                   Module tests
│   ├── test_simple.py                    Diagnostic tests
│   └── README.md                         How to run tests
│
├── 📂 notebooks/                         ← JUPYTER ANALYSIS (10 notebooks)
│   ├── 01_Baseline_EDA_Pima.ipynb
│   ├── 02-05_Gradual_Drift_*.ipynb      Gradual drift experiments
│   ├── 06-10_Abrupt_Drift_*.ipynb       Abrupt drift experiments
│   └── README.md                         Notebook descriptions
│
├── 📂 data/                              ← DATASETS
│   ├── raw/
│   │   └── diabetes.csv                  Original Pima dataset
│   └── processed/
│       ├── pima_step1_clean.csv
│       ├── pima_step2_imputed.csv
│       ├── fhgd_step1_clean.csv
│       └── fhgd_step2_imputed.csv
│
├── 📂 docs/                              ← DOCUMENTATION
│   ├── Drift_Detection_Presentation_Slides.pdf
│   └── implementation/
│       ├── ABRUPT_DRIFT_COMPLETION.md    Implementation details
│       ├── IMPLEMENTATION_STATUS.md      Status report
│       └── README.md                     Documentation index
│
├── 📂 scripts/                           ← SHELL SCRIPTS
│   ├── run_all_experiments.sh
│   ├── run_pima_experiments.sh
│   └── run_fhgd_experiments.sh
│
└── 📂 models/                            ← PRE-TRAINED MODELS
    ├── ocsvm_baseline.joblib
    └── ocsvm_tuned.joblib
```

---

## 🎯 How to Use the Code

### Option 1: Use Python Modules (Recommended for Code Reuse)

**Best for:** Building applications, integrating into production, writing custom code

```python
from drift_detection import (
    load_raw_data,
    create_missingness_flags,
    temporal_train_test_split,
    PreprocessingPipeline,
    fit_ocsvm,
    simulate_gradual_drift,
    calculate_detection_ratio
)

# Load data
df = load_raw_data('data/processed/pima_step1_clean.csv')
df = create_missingness_flags(df, ['Glucose', 'BloodPressure'])

# Split
features = [col for col in df.columns if col != 'Outcome']
X_base, X_test, y_base, y_test, _ = temporal_train_test_split(
    df, features, test_fraction=0.30
)

# Preprocess
pipeline = PreprocessingPipeline()
X_base_prep = pipeline.fit_transform(X_base, 
    identify_feature_types(X_base)[0],
    identify_feature_types(X_base)[1]
)
X_test_prep = pipeline.transform(X_test)

# Train & detect drift
model = fit_ocsvm(X_base_prep, gamma=0.1)
X_drifted = simulate_gradual_drift(X_test, 'Glucose', drift_percentage=0.4)
detection_ratio = calculate_detection_ratio(
    get_outlier_rate(model, X_test_prep),
    get_outlier_rate(model, pipeline.transform(X_drifted))
)
print(f"Detection ratio: {detection_ratio:.2f}x")
```

📖 **Full API documentation**: See `src/drift_detection/MODULE_USAGE.md` (850+ lines with examples)

### Option 2: Run Jupyter Notebooks (Recommended for Learning)

**Best for:** Understanding the analysis, visualization, exploration

```bash
cd notebooks
jupyter notebook 01_Baseline_EDA_Pima.ipynb
```

**Notebooks included:**
- `01-02`: Exploratory Data Analysis (Pima + Frankfurt)
- `03-04`: Gradual drift detection (OCSVM + Isolation Forest)
- `05-06`: Abrupt drift detection (OCSVM + Isolation Forest)

### Option 3: Run Tests (Verify Everything Works)

```bash
# Run all diagnostic tests
python tests/test_simple.py

# Run abrupt drift test suite (5/5 tests)
python tests/test_abrupt_drift.py

# View test documentation
cat tests/README.md
```

---

## 📚 Key Concepts

### Drift Types

**Gradual Drift** (Covariate Shift)
- Changes happen slowly over time
- Example: Patient population gradually gets older
- Best detected by: **Isolation Forest** (4.40×)

```python
from drift_detection import simulate_gradual_drift
X_drifted = simulate_gradual_drift(X_test, 'Glucose', drift_percentage=0.40)
```

**Abrupt Drift** (Concept Drift)
- Sudden changes in distribution
- Example: Hospital switches to new glucose meter
- Best detected by: **One-Class SVM** (3.18×)

```python
from drift_detection import apply_minmax_drift
X_drifted = apply_minmax_drift(X_test, baseline_stats, features=['Glucose'])
```

### Detection Metrics

**Detection Ratio** (Primary Metric)
- How much the anomaly rate increases under drift
- Formula: `outlier_rate_drifted / outlier_rate_baseline`
- Interpretation:
  - 1.0x = No drift detected
  - 2.0x = Anomalies doubled (good detection)
  - 4.4x = Very strong drift signal

**K-S Test** (Validation)
- Statistical test that drift is real (not random)
- p-value < 0.05 = Drift is statistically significant

---

## 🔍 Python Modules - API

### Core Functions

| Module | Function | Purpose |
|--------|----------|---------|
| **data.py** | `load_raw_data()` | Load CSV data |
| | `create_missingness_flags()` | Handle missing values |
| | `temporal_train_test_split()` | 70/30 split (prevents leakage) |
| **preprocessing.py** | `PreprocessingPipeline` | Impute & scale (fitted on baseline only) |
| **drift.py** | `simulate_gradual_drift()` | Gradual drift simulation |
| | `apply_minmax_drift()` | Abrupt drift simulation |
| **algorithms.py** | `fit_ocsvm()` | Train One-Class SVM |
| | `fit_isolation_forest()` | Train Isolation Forest |
| **evaluation.py** | `calculate_detection_ratio()` | Primary detection metric |
| | `validate_with_ks_test()` | Statistical validation |
| | `check_monotonicity()` | Verify drift increases with magnitude |

📖 **Complete documentation**: `src/drift_detection/MODULE_USAGE.md`

---

## 📊 Jupyter Notebooks - Analysis

All notebooks are **independent** - run any one directly:

```bash
jupyter notebook notebooks/03_Gradual_Drift_IsoForest_Pima.ipynb
```

| Notebook | Dataset | Topic | Runtime |
|----------|---------|-------|---------|
| 01 | Pima | Baseline EDA | ~5 min |
| 02 | Frankfurt | Baseline EDA | ~5 min |
| 03 | Pima | Gradual drift (IF) | ~8 min |
| 04 | Pima | Gradual drift (OCSVM) | ~10 min |
| 05 | Pima | Abrupt drift (IF) | ~8 min |
| 06 | Pima | Abrupt drift (OCSVM) | ~10 min |
| 07-10 | Frankfurt | Same topics | ~45 min each |

---

## ⚡ Quick Example

**Copy-paste code to detect drift in 30 seconds:**

```python
from drift_detection import *

# 1. Load data
df = load_raw_data('data/processed/pima_step1_clean.csv')
df = create_missingness_flags(df, ['Glucose', 'BloodPressure', 'Insulin', 'BMI', 'SkinThickness'])

# 2. Split (70% baseline, 30% test)
features = [c for c in df.columns if c != 'Outcome']
X_base, X_test, _, _, _ = temporal_train_test_split(df, features, test_fraction=0.30)

# 3. Preprocess
cont, ind = identify_feature_types(X_base)
pipe = PreprocessingPipeline()
X_base_p = pipe.fit_transform(X_base, cont, ind)
X_test_p = pipe.transform(X_test)

# 4. Train anomaly detector
model = fit_ocsvm(X_base_p, gamma=0.1)
baseline_rate = get_outlier_rate(model, X_base_p)

# 5. Create drift & detect
X_drift = simulate_gradual_drift(X_test, 'Glucose', drift_percentage=0.40)
X_drift_p = pipe.transform(X_drift)
drifted_rate = get_outlier_rate(model, X_drift_p)

# 6. Results
ratio = calculate_detection_ratio(baseline_rate, drifted_rate)
print(f"✓ Detection Ratio: {ratio:.2f}x")
```

**Output**:
```
✓ Detection Ratio: 3.00x
```

---

## 📖 Documentation Guide

**Where to go for different questions:**

| Question | File |
|----------|------|
| "How do I install this?" | This file (README.md) ↑ |
| "How do I use the Python API?" | `src/drift_detection/MODULE_USAGE.md` |
| "What's in each directory?" | `[DIR]/README.md` in each folder |
| "How are the modules implemented?" | `docs/implementation/IMPLEMENTATION_STATUS.md` |
| "What tests are available?" | `tests/README.md` |
| "What's the complete audit?" | `REPOSITORY_AUDIT.md` |

---

## 🔧 Troubleshooting

### Installation Issues

**Q: "Module not found" when importing**
```python
# Solution 1: Install from source
pip install -e .

# Solution 2: Add to Python path
import sys
sys.path.insert(0, '/path/to/src')
```

**Q: "No module named sklearn"**
```bash
pip install scikit-learn pandas numpy scipy
```

### Runtime Issues

**Q: Tests fail with data not found**
```bash
# Make sure you're in the repo root
cd /path/to/MSc-Dissertation-Drift-Detection
python tests/test_simple.py
```

**Q: Out of memory errors**
- Close other applications
- Run only one notebook at a time
- Use Pima dataset (smaller) instead of Frankfurt

---

## 📊 Project Stats

- **Code modules**: 6 (data, preprocessing, drift, algorithms, evaluation, utils)
- **Drift types**: 2 (gradual + abrupt)
- **Algorithms**: 2 (OCSVM + Isolation Forest)
- **Test coverage**: 5 comprehensive tests
- **Jupyter notebooks**: 10 analysis notebooks
- **Lines of code**: ~3,000 (core)
- **Documentation**: ~2,000 lines
- **Datasets**: 2 (Pima + Frankfurt)

---

## 📝 Citation

If you use this code, please cite:

```bibtex
@mastersthesis{Adeel2026,
  author = {Adeel, Malik},
  title = {Monitoring Population Drift in AI Medical Devices},
  school = {University of Hull},
  year = {2026}
}
```

---

## 📧 Contact

**Malik Adeel Anjum**  
University of Hull  
malikanjum.adeel@gmail.com

---

## 📜 License

MIT License - See LICENSE file for details

---

## ✅ Checklist: First Time Using This Repo?

- [ ] Read "What is this project?" (above)
- [ ] Run installation commands
- [ ] Run `python tests/test_simple.py` to verify
- [ ] Read `src/drift_detection/MODULE_USAGE.md` for API
- [ ] Run one Jupyter notebook from `notebooks/`
- [ ] Try the quick example above
- [ ] Check `docs/implementation/` for detailed docs

**You're done!** You now understand the project. 🎉

---

**Last updated**: May 7, 2026  
**Status**: Production ready ✓

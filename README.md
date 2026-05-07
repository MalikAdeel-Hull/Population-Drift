# Data Drift Detection in Healthcare

Monitoring population drift in deployed AI medical devices using unsupervised anomaly detection.

## Overview

This MSc dissertation evaluates drift detection methods on two independent healthcare cohorts:
- **Pima Indians Diabetes Dataset** (n=768)
- **Frankfurt Hospital Glucose Dataset** (n=2,000)

## Key Finding

**No single algorithm detects all drift types equally:**
- **Isolation Forest**: 4.40× better at gradual drift
- **One-Class SVM**: 3.18× better at abrupt drift

## System Requirements

### Hardware
- **Minimum:** 4 GB RAM, 2 GHz CPU
- **Recommended:** 8+ GB RAM, modern CPU
- **Disk Space:** ~500 MB for dependencies + datasets

### Software
- Python 3.8+
- pip package manager
- Jupyter Notebook (included in requirements.txt)
- Bash or equivalent shell (for running scripts)

### Installation Time
- Fresh environment setup: **5-10 minutes**
- Dependency installation: **3-5 minutes**
- **Total initial setup: ~15 minutes**

---

## Running Experiments

### Expected Runtime

| Experiment | Dataset | Runtime | Output |
|-----------|---------|---------|--------|
| Baseline EDA | Pima | ~5 min | Exploratory visualizations |
| Baseline EDA | FHGD | ~5 min | Exploratory visualizations |
| Gradual Drift (OCSVM) | Pima | ~10 min | Drift detection metrics |
| Gradual Drift (IsoForest) | Pima | ~8 min | Drift detection metrics |
| Abrupt Drift (OCSVM) | Pima | ~10 min | Drift detection metrics |
| Abrupt Drift (IsoForest) | Pima | ~8 min | Drift detection metrics |
| **Pima Total** | — | **~45 minutes** | — |
| **FHGD Total** | — | **~45 minutes** | — |
| **All Experiments** | Both | **~90 minutes** | — |

### Running All Experiments

```bash
bash scripts/run_all_experiments.sh
```

- **Time:** ~90 minutes
- **Output:** Executed notebooks saved to `reports/`
- **Results:** Metrics and visualizations in each notebook

### Running Dataset-Specific Experiments

```bash
# Pima dataset only (~45 minutes)
bash scripts/run_pima_experiments.sh

# FHGD dataset only (~45 minutes)
bash scripts/run_fhgd_experiments.sh
```

### Running Individual Notebooks

```bash
cd notebooks
jupyter notebook 01_Baseline_EDA_Pima.ipynb
```

Each notebook is independent and can be run individually.

### Expected Output

After running experiments, you will find:
- **Executed notebooks:** `reports/` directory
- **Generated models:** `models/` directory (joblib files)
- **Analysis results:** Within each notebook (metrics, plots, statistics)

### Key Metrics Generated

- **Kolmogorov-Smirnov (KS) test statistics** for drift detection
- **ROC-AUC scores** for algorithm performance
- **Confusion matrices** for classification accuracy
- **Drift detection visualizations** (time-series plots, heatmaps)

---

## Troubleshooting

### Issue: "Module not found" error
**Solution:** Ensure you've installed requirements: `pip install -r requirements.txt`

### Issue: Jupyter command not found
**Solution:** Install jupyter: `pip install jupyter`

### Issue: Long runtime with no progress
**Solution:** This is normal for large datasets. Experiments can take 1-2 hours.

### Issue: Out of memory errors
**Solution:** Close other applications and try running dataset-specific scripts instead of all at once.
```

## Project Structure
notebooks/          10 analysis notebooks (01-10)
data/              Raw & processed datasets
scripts/           Execution scripts
models/            Trained models (.joblib files)
docs/              Manuscript & slides

## Datasets

Both datasets are **publicly available** on Kaggle and can be easily downloaded.

### Pima Indians Diabetes Dataset

| Property | Value |
|----------|-------|
| **Source** | [Kaggle - UCI ML Repository Mirror](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) |
| **Samples** | 768 |
| **Features** | 8 numeric + 1 binary target |
| **License** | Public Domain |
| **Download** | `kaggle datasets download -d uciml/pima-indians-diabetes-database` |

### Frankfurt Hospital Glucose Dataset (FHGD)

| Property | Value |
|----------|-------|
| **Source** | [Kaggle - Public Domain](https://www.kaggle.com/datasets/johndasilva/diabetes) |
| **Samples** | 2,000 |
| **Features** | Clinical glucose monitoring (8+ features) |
| **License** | CC0 1.0 Universal (Public Domain) |
| **Download** | `kaggle datasets download -d johndasilva/diabetes` |

**Note:** Both datasets are processed identically through cleaning and imputation pipelines. See `data/raw/README.md` for detailed sourcing information.
## Methodology

**Algorithms:**
- One-Class SVM (OCSVM): kernel='rbf', nu=0.05
- Isolation Forest: n_estimators=100, contamination=0.05

**Drift Types:**
- Gradual: Multivariate erosion (slow changes)
- Abrupt: Systemic shocks (sudden shifts)

**Validation:** Kolmogorov-Smirnov tests, 70/30 train-test split

## Results

| Scenario | Winner | Performance |
|----------|--------|-------------|
| Gradual Drift | Isolation Forest | 4.40× |
| Abrupt Drift | OCSVM | 3.18× |

**Conclusion:** Hybrid monitoring framework needed for clinical safety.

## Citation

```bibtex
@mastersthesis{Adeel2026,
  author = {Adeel, Malik},
  title = {Monitoring Population Drift in AI Medical Devices},
  school = {University of Hull},
  year = {2026}
}
```

## License

MIT License - See LICENSE file

## Author

Malik Adeel Anjum  
University of Hull  
malikanjum.adeel@gmail.com
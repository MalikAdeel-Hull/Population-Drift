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

## Quick Start

```bash
git clone https://github.com/MalikAdeel-Hull/MSc-Dissertation-Drift-Detection.git
cd MSc-Dissertation-Drift-Detection

# Setup
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt

# Run experiments
bash scripts/run_pima_experiments.sh    # Pima only
bash scripts/run_fhgd_experiments.sh    # FHGD only
bash scripts/run_all_experiments.sh     # Both datasets
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
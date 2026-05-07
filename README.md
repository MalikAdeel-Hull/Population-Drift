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

| Dataset | Samples | Features | Source |
|---------|---------|----------|--------|
| Pima | 768 | 8 | UCI ML Repository |
| FHGD | 2,000 | 8+ | Frankfurt Hospital |

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
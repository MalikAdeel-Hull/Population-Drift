# Raw Data Directory

## Pima Indians Diabetes Dataset

- **Source:** Kaggle (UCI ML Repository Mirror)
- **URL:** https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- **File:** `diabetes.csv`
- **Samples:** 768
- **Features:** 8 numeric features + 1 binary target (diabetes onset within 5 years)
- **License:** Public Domain
- **Citation:** Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. (1988). Using the ADAP learning algorithm to forecast the onset of diabetes mellitus. *Proceedings of the Annual Symposium on Computer Application in Medical Care*, 261–265.
- **Accessed:** 2025

### How to obtain:
```bash
# Option 1: Download from Kaggle (requires kaggle CLI)
kaggle datasets download -d uciml/pima-indians-diabetes-database
unzip pima-indians-diabetes-database.zip
mv diabetes.csv data/raw/

# Option 2: Manual download from https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
# Download and place diabetes.csv in this directory
```

---

## Frankfurt Hospital Glucose Dataset (FHGD)

- **Source:** Kaggle
- **URL:** https://www.kaggle.com/datasets/johndasilva/diabetes
- **Samples:** 2,000
- **Features:** Clinical glucose monitoring data (8+ features)
- **License:** CC0 1.0 Universal Public Domain Dedication
- **Access:** Freely available - no restrictions
- **Accessed:** 2025

### How to obtain:
```bash
# Option 1: Download from Kaggle (requires kaggle CLI)
kaggle datasets download -d johndasilva/diabetes
unzip diabetes.zip
# Extract FHGD-specific files to data/raw/

# Option 2: Manual download from https://www.kaggle.com/datasets/johndasilva/diabetes
# Download and place files in this directory
```

---

## Processing Pipeline

Both datasets undergo identical preprocessing steps:

1. **Step 1 (Cleaning):** 
   - Missing value identification and handling
   - Outlier detection and treatment
   - Output: `{pima,fhgd}_step1_clean.csv` in `/data/processed/`

2. **Step 2 (Imputation):** 
   - Statistical imputation for missing values
   - Feature normalization
   - Output: `{pima,fhgd}_step2_imputed.csv` in `/data/processed/`

Processed datasets are ready for analysis and are used in all notebooks.

---

## File Manifest

### Raw Data Files
- `diabetes.csv` - Pima Indians Diabetes Dataset (original, unmodified)

### Processed Data Files (in `/data/processed/`)
- `pima_step1_clean.csv` - Pima data after cleaning
- `pima_step2_imputed.csv` - Pima data after imputation (analysis-ready)
- `fhgd_step1_clean.csv` - FHGD data after cleaning
- `fhgd_step2_imputed.csv` - FHGD data after imputation (analysis-ready)

All notebooks use the `step2_imputed` versions for drift detection analysis.
# Monitoring Population Drift in Deployed Medical AI
**Author:** Malik Adeel Anjum  
**Project:** MSc AI for Healthcare Dissertation (University of Hull)

---

## 📂 Repository Structure

### 1. Pima Indians Diabetes Dataset (Core Comparison)
*Located in the `/Pima/` folder:*
* **`01_Baseline_EDA.ipynb`**: Data cleaning, missingness handling, and temporal splitting.
* **`02_Gradual_Drift_OCSVM.ipynb`**: Baseline drift detection using One-Class SVM.
* **`03_Gradual_Drift_IF.ipynb`**: Evaluating Isolation Forest on multiplicative shifts.
* **`04_Abrupt_Drift_OCSVM.ipynb`**: Testing distance-based sensitivity to systemic shocks.
* **`05_Abrupt_Drift_IF.ipynb`**: Testing ensemble-based sensitivity to systemic shocks.

### 2. FHGD (Federated Healthcare Group) Dataset (Validation)
*Located in the `/FHGD/` folder:*
* **`FHGD_IF_Gradual.ipynb`**: Validation of Isolation Forest sensitivity on high-dimensional clinical data.
* **`abrupt_IF_FHGD.ipynb`**: Mechanistic study of Isolation Forest response to sudden affine remapping.

---

## 🔑 Key Findings (The "Detection Trade-off")
* **Gradual Drift:** **Isolation Forest** outperformed OCSVM by **4.40x** in sensitivity, making it ideal for detecting slow biological shifts.
* **Abrupt Drift:** **One-Class SVM** was **2.92x** more reactive, making it the better choice for sudden systemic "shocks" (e.g., equipment changes).

---

## 🚀 How to Run
1. **Environment Setup**: 
   Install the required libraries using:
   ```bash
   pip install -r requirements.txt
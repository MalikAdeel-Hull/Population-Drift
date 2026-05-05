# Monitoring Population Drift in Deployed Medical AI
**Unsupervised Anomaly Detection for the Mitigation of "Silent Failures"**

**Author:** Malik Adeel Anjum  
**Project:** MSc AI for Healthcare Dissertation (University of Hull)  
**Academic Year:** 2024-2025  

---

## 📊 Research Overview
In clinical AI deployment, **Population Drift** occurs when patient demographics or clinical environments shift away from training data. This leads to "Silent Failures"—where models provide inaccurate predictions without warning. 

This repository explores the efficacy of **Isolation Forest (IF)** and **One-Class Support Vector Machines (OCSVM)** as unsupervised monitors to detect these shifts in real-time, focusing on two distinct drift archetypes:
* **Gradual Multiplicative Drift:** Simulates slow biological or demographic changes (e.g., rising BMI/Age trends).
* **Abrupt Affine Drift:** Simulates sudden "systemic shocks" (e.g., changes in hospital equipment calibration).

---

## 📁 Repository Structure

### 1. Pima Indians Diabetes Dataset (Core Comparison)
*Located in the `/Pima/` folder. These notebooks establish fundamental performance trade-offs.*
* **`01_Baseline_EDA.ipynb`**: Data cleaning, missingness analysis, and creation of the leak-free 70/30 temporal split.
* **`02_Gradual_Drift_OCSVM.ipynb`**: Performance of distance-based detection on slow clinical shifts.
* **`03_Gradual_Drift_IF.ipynb`**: Evaluating ensemble-based sensitivity to gradual covariate drift.
* **`04_Abrupt_Drift_OCSVM.ipynb`**: Benchmarking OCSVM reactivity to sudden systemic shocks.
* **`05_Abrupt_Drift_IF.ipynb`**: Testing Isolation Forest's response to abrupt distributional displacement.

### 2. Frankfurt General Hospital Dataset (FHGD) (Sensitivity Validation)
*Located in the `/FHGD/` folder. Advanced validation on high-dimensional clinical data.*
* **`FHGD_IF_Gradual.ipynb`**: Validation of IF sensitivity to slow covariate deterioration across multiple clinical features.
* **`abrupt_IF_FHGD.ipynb`**: Mechanistic study of Isolation Forest response to sudden affine remapping.
* *(Includes additional data processing and validation notebooks within the FHGD directory)*.

---

## 🔑 Key Research Findings: The "Detection Trade-off"
The study reveals that no single algorithm is universally superior. A strategic choice is required based on the expected drift:

| Drift Archetype | Sensitivity Winner | Key Result |
| :--- | :--- | :--- |
| **Gradual (Multi-factor)** | **Isolation Forest** | **4.40x** higher sensitivity than OCSVM |
| **Abrupt (Systemic Shock)** | **One-Class SVM** | **2.92x** higher reactivity than IF |

---

## 🔬 Methodology Highlights
* **Leak-Free Pipeline:** Preprocessing parameters (Imputation/Scaling) are fit *exclusively* on the baseline to ensure rigorous validation.
* **Statistical Validation:** Detection signals are verified using **Two-sample Kolmogorov-Smirnov (K-S) tests** ($p < 0.05$).
* **Explainability:** Integrated **SHAP (SHapley Additive exPlanations)** to identify which clinical features drive the drift signal.

---

## 🚀 Getting Started
1. **Clone the Repository**:
   ```bash
   git clone [https://github.com/MalikAdeel-Hull/medical-ai-drift.git](https://github.com/MalikAdeel-Hull/medical-ai-drift.git)
   cd medical-ai-drift
   
## 🛠 Reproduction Guide
To reproduce the findings presented in the paper:
1. **Environment Setup**: 
   `pip install -r requirements.txt`
2. **Execute Pipeline**:
   - Open `/Pima/01_Baseline_EDA.ipynb` to generate the 70/30 chronological split.
   - Run the corresponding `Drift` notebooks to generate Detection Ratios.
   - Run `SHAP_Validation.ipynb` to verify feature importance.
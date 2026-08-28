"""
Published Experiment Configuration
==================================
Single source of truth for the parameters and reference results of:

    Anjum, M. A. (2026). Monitoring Population Drift in Deployed AI Medical
    Devices: An AI-Centred Diagnostic Accuracy Study of Detection Ratios Across
    US and European Cohorts (v2). Zenodo. doi:10.5281/zenodo.20633719

Every constant here is taken from the preprint (Tables 1, 2 and 8, and the
Methods section) or from the notebooks that generated it. `scripts/reproduce_
paper.py` uses these values to regenerate and verify all 16 reported detection
ratios.

Two conventions in here are easy to get wrong; both are load-bearing:

1. Detection-ratio denominators differ by arm.
   - Gradual: zero-drift control rate on the undrifted TEST set.
   - Abrupt:  anomaly rate on the BASELINE (training) set.
   See DR_DENOMINATOR below and docs/AUDIT.md.

2. Isolation Forest `max_samples` is cohort-specific for the gradual arm
   (Pima 256, FHGD 128 — both selected by the notebooks' 80/20 grid search)
   and left at scikit-learn's default for the abrupt arm.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Cohorts (preprint Table 1)
# ---------------------------------------------------------------------------

COHORTS = {
    "Pima": {
        "label": "Pima Indians (n=768)",
        "step1_path": "data/processed/pima_step1_clean.csv",
        "n_total": 768,
        "n_baseline": 537,
        "n_test": 231,
    },
    "FHGD": {
        "label": "Frankfurt Hospital (n=2,000)",
        "step1_path": "data/processed/fhgd_step1_clean.csv",
        "n_total": 2000,
        "n_baseline": 1400,
        "n_test": 600,
    },
}

DRIFT_FEATURES_UNI = ["Glucose"]
DRIFT_FEATURES_MULTI = ["Glucose", "BMI", "Age"]
DRIFT_LEVELS = [0.10, 0.25, 0.40]

# ---------------------------------------------------------------------------
# Physiological clipping bounds (preprint Methods, "Mathematical Simulation of
# Data Drift"). Applied to the GRADUAL arm only; the abrupt arm is unclipped.
# ---------------------------------------------------------------------------

PAPER_FEATURE_RANGES = {
    "Pregnancies": (0, 17),
    "Glucose": (70, 200),
    "BloodPressure": (40, 120),
    "SkinThickness": (7, 99),
    "Insulin": (14, 846),
    "BMI": (15, 50),
    "DiabetesPedigreeFunction": (0.078, 2.42),
    "Age": (21, 81),
}

# ---------------------------------------------------------------------------
# Hyperparameters (preprint Table 2)
# ---------------------------------------------------------------------------

GRADUAL_CONFIG = {
    "ocsvm": {"kernel": "rbf", "gamma": 0.1, "nu": 0.05},
    "isolation_forest": {
        "Pima": {
            "n_estimators": 100,
            "max_samples": 256,
            "contamination": 0.05,
            "random_state": 42,
        },
        "FHGD": {
            "n_estimators": 100,
            "max_samples": 128,
            "contamination": 0.05,
            "random_state": 42,
        },
    },
}

ABRUPT_CONFIG = {
    "ocsvm": {"kernel": "rbf", "gamma": "scale", "nu": 0.20},
    # max_samples deliberately left at the scikit-learn default ('auto') here:
    # the abrupt notebook does not set it, and setting it changes the result.
    "isolation_forest": {
        "n_estimators": 100,
        "contamination": 0.20,
        "random_state": 42,
    },
    "shift_factor": 0.4,   # delta - 60% downward location shift
    "range_factor": 1.5,   # sigma - 50% variance expansion
}

DR_DENOMINATOR = {
    "gradual": "zero-drift control rate on the undrifted test set",
    "abrupt": "anomaly rate on the baseline (training) set",
}

# ---------------------------------------------------------------------------
# Published results (preprint Table 8). Keyed (cohort, algorithm, scenario).
# ---------------------------------------------------------------------------

PAPER_GRADUAL_DR = {
    ("Pima", "OCSVM", "Univariate"): 1.48,
    ("Pima", "OCSVM", "Multivariate"): 2.07,
    ("Pima", "IF", "Univariate"): 1.60,
    ("Pima", "IF", "Multivariate"): 4.40,
    ("FHGD", "OCSVM", "Univariate"): 1.83,
    ("FHGD", "OCSVM", "Multivariate"): 2.76,
    ("FHGD", "IF", "Univariate"): 1.11,
    ("FHGD", "IF", "Multivariate"): 1.89,
}

PAPER_ABRUPT_DR = {
    ("Pima", "OCSVM", "Univariate"): 1.47,
    ("Pima", "OCSVM", "Multivariate"): 2.92,
    ("Pima", "IF", "Univariate"): 1.14,
    ("Pima", "IF", "Multivariate"): 1.42,
    ("FHGD", "OCSVM", "Univariate"): 1.62,
    ("FHGD", "OCSVM", "Multivariate"): 3.18,
    ("FHGD", "IF", "Univariate"): 1.44,
    ("FHGD", "IF", "Multivariate"): 1.79,
}

# Zero-drift control rates quoted in the preprint's Results section
# (gradual arm, nu = contamination = 0.05).
PAPER_ZERO_DRIFT_CONTROL = {
    ("Pima", "OCSVM"): 0.117,
    ("Pima", "IF"): 0.022,
    ("FHGD", "OCSVM"): 0.077,
    ("FHGD", "IF"): 0.058,
}

# Headline result: FHGD abrupt multivariate OCSVM.
#
# The point estimate comes from the true baseline counts (284 outliers of the
# 1,400-sample training set = 20.286%) against 387 of 600 drifted samples:
#     0.6450 / 0.20286 = 3.1796 -> 3.18x
#
# The published 95% CI (2.69-3.78) was, however, bootstrapped as though the
# baseline denominator carried only 600 samples. Resampling with the real
# n=1,400 baseline gives a tighter [2.83, 3.60]. Both are recorded here: the
# published pair is what tests/test_paper_reproduction.py checks, the corrected
# pair is what a future revision should quote. See docs/AUDIT.md, finding 6.
PAPER_HEADLINE = {
    "cohort": "FHGD",
    "algorithm": "OCSVM",
    "scenario": "Multivariate abrupt",
    "n_outliers_baseline": 284,
    "n_total_baseline": 1400,
    "n_outliers_drifted": 387,
    "n_total_drifted": 600,
    "detection_ratio": 3.18,
    # As published - bootstrap run with a 600-sample baseline denominator.
    "ci": (2.69, 3.78),
    "ci_bootstrap_inputs_as_published": (122, 600, 387, 600),
    # Corrected - bootstrap run with the real 1,400-sample baseline.
    "ci_corrected": (2.83, 3.60),
}

__all__ = [
    "COHORTS",
    "DRIFT_FEATURES_UNI",
    "DRIFT_FEATURES_MULTI",
    "DRIFT_LEVELS",
    "PAPER_FEATURE_RANGES",
    "GRADUAL_CONFIG",
    "ABRUPT_CONFIG",
    "DR_DENOMINATOR",
    "PAPER_GRADUAL_DR",
    "PAPER_ABRUPT_DR",
    "PAPER_ZERO_DRIFT_CONTROL",
    "PAPER_HEADLINE",
]

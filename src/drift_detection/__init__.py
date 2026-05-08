"""
Drift Detection Code Modules
=============================
Extracted and refactored code from Jupyter notebooks for reusability.

This package provides modular, production-ready code for data drift detection in medical AI systems.
Implements One-Class SVM and Isolation Forest algorithms with comprehensive evaluation metrics.

Modules:
  - data.py: Data loading, preprocessing setup, and 70/30 temporal splitting
  - preprocessing.py: Imputation and standardization pipeline
  - drift.py: Drift simulation (gradual and multivariate)
  - algorithms.py: OCSVM and Isolation Forest training and prediction
  - evaluation.py: Drift detection metrics (detection ratio, K-S test, monotonicity)
  - utils.py: High-level experiment orchestration and result management

Quick Start:
  >>> from drift_detection import (
  ...     load_raw_data, create_missingness_flags,
  ...     temporal_train_test_split, PreprocessingPipeline,
  ...     fit_ocsvm, simulate_gradual_drift, evaluate_drift_detection
  ... )

For complete usage examples, see MODULE_USAGE.md
"""

__version__ = '1.0.0'
__author__ = 'Malik Adeel'

# Convenience imports
from .data import (
    load_raw_data,
    create_missingness_flags,
    temporal_train_test_split,
    identify_feature_types,
    PIMA_FEATURES,
    PIMA_TARGET,
    PIMA_COLS_WITH_MISSING
)

from .preprocessing import (
    PreprocessingPipeline,
    preprocess_baseline_and_test
)

from .drift import (
    simulate_gradual_drift,
    simulate_multivariate_drift,
    apply_minmax_drift,
    DEFAULT_CLINICAL_RANGES
)

from .algorithms import (
    fit_ocsvm,
    fit_isolation_forest,
    tune_ocsvm_gamma,
    tune_isolation_forest,
    predict_anomalies,
    get_outlier_rate
)

from .evaluation import (
    calculate_outlier_rate,
    calculate_detection_ratio,
    bootstrap_detection_ratio_ci,
    validate_with_ks_test,
    validate_multiple_features,
    check_monotonicity,
    evaluate_drift_detection,
    create_results_dataframe
)

from .utils import (
    run_drift_detection_experiment,
    run_univariate_drift_sweep,
    save_results,
    load_results,
    generate_summary_statistics,
    DEFAULT_CONFIG
)

from .shap_analysis import (
    create_shap_explainer,
    compute_shap_values,
    get_feature_importance,
    compare_baseline_vs_drift,
    plot_shap_summary,
    plot_shap_waterfall,
    validate_mechanistic_consistency
)

__all__ = [
    # data.py
    'load_raw_data',
    'create_missingness_flags',
    'temporal_train_test_split',
    'identify_feature_types',
    'PIMA_FEATURES',
    'PIMA_TARGET',
    'PIMA_COLS_WITH_MISSING',
    # preprocessing.py
    'PreprocessingPipeline',
    'preprocess_baseline_and_test',
    # drift.py
    'simulate_gradual_drift',
    'simulate_multivariate_drift',
    'apply_minmax_drift',
    'DEFAULT_CLINICAL_RANGES',
    # algorithms.py
    'fit_ocsvm',
    'fit_isolation_forest',
    'tune_ocsvm_gamma',
    'tune_isolation_forest',
    'predict_anomalies',
    'get_outlier_rate',
    # evaluation.py
    'calculate_outlier_rate',
    'calculate_detection_ratio',
    'bootstrap_detection_ratio_ci',
    'validate_with_ks_test',
    'validate_multiple_features',
    'check_monotonicity',
    'evaluate_drift_detection',
    'create_results_dataframe',
    # utils.py
    'run_drift_detection_experiment',
    'run_univariate_drift_sweep',
    'save_results',
    'load_results',
    'generate_summary_statistics',
    'DEFAULT_CONFIG',
    # shap_analysis.py
    'create_shap_explainer',
    'compute_shap_values',
    'get_feature_importance',
    'compare_baseline_vs_drift',
    'plot_shap_summary',
    'plot_shap_waterfall',
    'validate_mechanistic_consistency'
]

# Drift Detection Modules Usage Guide

Complete reference for using the drift detection modules with practical examples.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Data Module](#data-module)
3. [Preprocessing Module](#preprocessing-module)
4. [Drift Simulation](#drift-simulation)
   - [Gradual Drift](#gradual-drift)
   - [Abrupt Drift](#abrupt-drift)
5. [Anomaly Detection Algorithms](#anomaly-detection-algorithms)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Experiment Orchestration](#experiment-orchestration)
8. [Complete Workflows](#complete-workflows)

---

## Quick Start

```python
from drift_detection import (
    load_raw_data,
    create_missingness_flags,
    temporal_train_test_split,
    PreprocessingPipeline,
    fit_ocsvm,
    fit_isolation_forest,
    simulate_gradual_drift,
    evaluate_drift_detection
)

# 1. Load and split data
X_raw, y = load_raw_data('data/processed/pima_step1_clean.csv')
X_raw_flags = create_missingness_flags(X_raw)
X_base_raw, X_test_raw, _, _ = temporal_train_test_split(
    X_raw_flags, y, test_fraction=0.30
)

# 2. Preprocess
pipeline = PreprocessingPipeline(
    imputation_strategy='median',
    scaling_method='standard'
)
X_base_prep = pipeline.fit_transform(X_base_raw)
X_test_prep = pipeline.transform(X_test_raw)

# 3. Train anomaly detector
model = fit_ocsvm(X_base_prep, gamma=0.1)

# 4. Apply drift and evaluate
X_test_drifted = simulate_gradual_drift(
    X_test_raw, 'Glucose', drift_percentage=0.4
)
X_test_drifted_prep = pipeline.transform(X_test_drifted)

results = evaluate_drift_detection(
    model, X_test_prep, X_test_drifted_prep,
    feature_for_ks='Glucose'
)
print(f"Detection Ratio: {results['detection_ratio']:.2f}x")
```

---

## Data Module

### Loading Data

```python
from drift_detection import load_raw_data, PIMA_FEATURES, PIMA_TARGET

# Load Pima Indians Diabetes Dataset
X, y = load_raw_data('data/processed/pima_step1_clean.csv')

print(f"Shape: {X.shape}")  # (768, 8)
print(f"Features: {X.columns.tolist()}")  # 8 medical features
print(f"Missing data: {X.isnull().sum().sum()} cells")
```

### Creating Missingness Flags

The Pima dataset encodes missing values as 0 in certain features. Create binary indicator columns:

```python
from drift_detection import create_missingness_flags, PIMA_COLS_WITH_MISSING

X_raw = load_raw_data('data/processed/pima_step1_clean.csv')[0]
X_with_flags = create_missingness_flags(X_raw)

# New columns added: f"{feature}_missing" for each feature in PIMA_COLS_WITH_MISSING
print(X_with_flags.shape)  # (768, 15) - 8 original + 7 missing flags
print(X_with_flags.columns.tolist())
# ['Pregnancies', 'Glucose', ..., 'Glucose_missing', 'BloodPressure_missing', ...]
```

### Temporal Train/Test Split

**Critical**: Split BEFORE preprocessing to prevent data leakage.

```python
from drift_detection import temporal_train_test_split

X_raw, y = load_raw_data('data/processed/pima_step1_clean.csv')
X_raw = create_missingness_flags(X_raw)

# 70/30 temporal split (preserves temporal ordering)
X_base, X_test, y_base, y_test = temporal_train_test_split(
    X_raw, y, test_fraction=0.30
)

print(f"Baseline: {X_base.shape[0]} samples (70%)")
print(f"Test: {X_test.shape[0]} samples (30%)")
```

---

## Preprocessing Module

### Basic Pipeline Usage

```python
from drift_detection import PreprocessingPipeline

# Initialize pipeline
pipeline = PreprocessingPipeline(
    imputation_strategy='median',  # Options: 'median', 'mean', 'most_frequent'
    scaling_method='standard'       # Options: 'standard', 'minmax'
)

# Fit on baseline only
X_base_preprocessed = pipeline.fit_transform(X_base_raw)

# Transform test data using baseline statistics
X_test_preprocessed = pipeline.transform(X_test_raw)

# No data leakage: scaling params come from baseline only
print(f"Baseline mean: {X_base_preprocessed.mean():.3f}")
print(f"Baseline std: {X_base_preprocessed.std():.3f}")
```

### Accessing Fitted Transformers

```python
# Get the fitted imputer and scaler
imputer = pipeline.imputer
scaler = pipeline.scaler

# Check what was learned from baseline
print(f"Median values for imputation: {imputer.statistics_}")
print(f"Mean for scaling: {scaler.mean_}")
print(f"Std for scaling: {scaler.scale_}")
```

### Preprocessing Multiple Datasets

```python
# Wrapper function for convenience
from drift_detection import preprocess_baseline_and_test

X_base_prep, X_test_prep, pipeline = preprocess_baseline_and_test(
    X_base_raw, X_test_raw,
    imputation_strategy='median',
    scaling_method='standard'
)
```

---

## Drift Simulation

### Gradual Drift

Gradual drift (covariate shift) increases linearly over time.

#### Single Feature Drift

```python
from drift_detection import simulate_gradual_drift, DEFAULT_CLINICAL_RANGES

# Apply 40% gradual drift to Glucose over 100 samples
X_drifted = simulate_gradual_drift(
    X_test_raw,
    feature='Glucose',
    drift_percentage=0.40,
    duration=100,
    feature_ranges=DEFAULT_CLINICAL_RANGES
)

# Drift schedule increases linearly:
# Sample 0:   0% drift → value * 1.00
# Sample 25: 10% drift → value * 1.10
# Sample 50: 20% drift → value * 1.20
# Sample 75: 30% drift → value * 1.30
# Sample 100: 40% drift → value * 1.40
```

#### Multivariate Drift

Apply the same drift schedule to multiple features simultaneously:

```python
from drift_detection import simulate_multivariate_drift

X_drifted = simulate_multivariate_drift(
    X_test_raw,
    features=['Glucose', 'BMI', 'BloodPressure'],
    drift_percentage=0.30,
    duration=len(X_test_raw),
    feature_ranges=DEFAULT_CLINICAL_RANGES
)

# All three features drift together by the same schedule
```

#### Drift Schedule Control

```python
# Short duration: rapid drift
X_rapid = simulate_gradual_drift(
    X_test_raw, 'Glucose',
    drift_percentage=0.40,
    duration=20  # Reach 40% in just 20 samples
)

# Long duration: slow drift
X_slow = simulate_gradual_drift(
    X_test_raw, 'Glucose',
    drift_percentage=0.40,
    duration=500  # Reach 40% over 500 samples
)

# Custom ranges
custom_ranges = {
    'Glucose': (60, 250),
    'BMI': (10, 70)
}
X_drifted = simulate_gradual_drift(
    X_test_raw, 'Glucose',
    drift_percentage=0.40,
    feature_ranges=custom_ranges
)
```

#### Verification

```python
from drift_detection import verify_drift_application

stats = verify_drift_application(
    X_test_raw, X_drifted,
    feature='Glucose',
    expected_direction='increase'
)

print(f"Mean change: {stats['mean_change_percent']:.2f}%")
print(f"NaNs preserved: {stats['nans_preserved']}")
print(f"Number of NaNs: {stats['n_nans']}")
```

---

### Abrupt Drift

Abrupt drift (concept drift) applies an affine transformation across the entire dataset at once.

#### Basic Abrupt Drift

```python
from drift_detection import apply_minmax_drift

# First, compute baseline statistics
X_base_clean = X_base_raw.dropna()
X_base_stats = {
    'Glucose': {
        'f_min': X_base_clean['Glucose'].min(),
        'f_range': X_base_clean['Glucose'].max() - X_base_clean['Glucose'].min()
    },
    'BMI': {
        'f_min': X_base_clean['BMI'].min(),
        'f_range': X_base_clean['BMI'].max() - X_base_clean['BMI'].min()
    }
}

# Apply abrupt drift
X_drifted = apply_minmax_drift(
    X_test_raw,
    X_base_stats,
    features=['Glucose', 'BMI'],
    shift_f=0.4,      # 40% downward shift
    range_f=1.5,      # 1.5x range expansion
    verbose=True
)

# Output:
# ======================================================================
# Applying Abrupt Min/Max Drift
# Shift Factor: 0.40 | Range Factor: 1.50
# ======================================================================
# 
#   Glucose: mean 122.05 → 165.23 (+35.3%)
#   BMI: mean 31.54 → 42.87 (+35.8%)
```

#### Helper Function for Baseline Statistics

```python
def compute_baseline_stats(X_baseline, features):
    """Compute min and range for each feature."""
    stats = {}
    X_clean = X_baseline.dropna()
    for feature in features:
        f_min = X_clean[feature].min()
        f_max = X_clean[feature].max()
        f_range = f_max - f_min
        stats[feature] = {'f_min': f_min, 'f_range': f_range}
    return stats

X_base_stats = compute_baseline_stats(
    X_base_raw,
    features=['Glucose', 'BMI', 'BloodPressure', 'Insulin']
)
```

#### Multivariate Abrupt Drift

```python
# Apply to 4 features simultaneously
X_drifted = apply_minmax_drift(
    X_test_raw,
    X_base_stats,
    features=['Glucose', 'BMI', 'BloodPressure', 'Insulin'],
    shift_f=0.4,
    range_f=1.5
)
```

#### Parameter Tuning

```python
# Mild shift, moderate expansion
X_drifted_mild = apply_minmax_drift(
    X_test_raw, X_base_stats,
    features=['Glucose'],
    shift_f=0.2,   # 20% shift
    range_f=1.2    # 1.2x expansion
)

# Aggressive shift, large expansion
X_drifted_severe = apply_minmax_drift(
    X_test_raw, X_base_stats,
    features=['Glucose'],
    shift_f=0.6,   # 60% shift
    range_f=2.0    # 2.0x expansion
)
```

---

## Anomaly Detection Algorithms

### One-Class SVM

```python
from drift_detection import fit_ocsvm

# Train on baseline
model = fit_ocsvm(
    X_base_preprocessed,
    kernel='rbf',
    nu=0.05,          # 5% expected anomalies
    gamma=0.1         # Kernel parameter
)

# Predict
y_pred = model.predict(X_test_preprocessed)
# Returns: +1 (inlier) or -1 (outlier)

n_anomalies = (y_pred == -1).sum()
anomaly_rate = n_anomalies / len(y_pred)
print(f"Anomalies: {n_anomalies}/{len(y_pred)} ({anomaly_rate:.1%})")
```

### Gamma Tuning for OCSVM

```python
from drift_detection import tune_ocsvm_gamma

best_model, best_gamma, results = tune_ocsvm_gamma(
    X_base_preprocessed,
    gamma_values=[0.001, 0.01, 0.1, 0.5],
    nu=0.05,
    kernel='rbf'
)

print(f"Best gamma: {best_gamma}")
for gamma_str, metrics in results.items():
    print(f"  {gamma_str}: outlier_rate={metrics['outlier_rate']:.3f}")
```

### Isolation Forest

```python
from drift_detection import fit_isolation_forest

model = fit_isolation_forest(
    X_base_preprocessed,
    n_estimators=100,
    max_samples=256,
    contamination=0.05,
    random_state=42
)

y_pred = model.predict(X_test_preprocessed)
anomaly_rate = (y_pred == -1).sum() / len(y_pred)
```

### Hyperparameter Tuning for Isolation Forest

```python
from drift_detection import tune_isolation_forest

# Split baseline into train/validation
split_idx = int(0.8 * len(X_base_preprocessed))
X_train = X_base_preprocessed[:split_idx]
X_val = X_base_preprocessed[split_idx:]

best_model, best_params, results = tune_isolation_forest(
    X_train, X_val,
    n_estimators_list=[50, 100, 200],
    max_samples_list=[128, 256, 512],
    contamination=0.05
)

print(f"Best params: {best_params}")
```

---

## Evaluation Metrics

### Outlier Rate

```python
from drift_detection import calculate_outlier_rate

baseline_rate = calculate_outlier_rate(model, X_base_preprocessed)
drifted_rate = calculate_outlier_rate(model, X_drifted_preprocessed)

print(f"Baseline: {baseline_rate:.1%}")
print(f"Drifted: {drifted_rate:.1%}")
```

### Detection Ratio (Primary Metric)

```python
from drift_detection import calculate_detection_ratio

detection_ratio = calculate_detection_ratio(baseline_rate, drifted_rate)

# Interpretation:
# 1.0x = No change (drift not detected)
# 1.5x = 50% increase in anomalies
# 2.0x = Doubling of anomalies (strong detection)
# 4.4x = Very strong drift signal

if detection_ratio > 2.0:
    print("✓ Strong drift signal detected")
elif detection_ratio > 1.5:
    print("⚠ Moderate drift detected")
else:
    print("✗ Weak or no drift detected")
```

### K-S Test (Statistical Validation)

```python
from drift_detection import validate_with_ks_test

ks_stat, p_value, is_significant = validate_with_ks_test(
    X_base_raw['Glucose'].dropna().values,
    X_drifted_raw['Glucose'].dropna().values,
    alpha=0.05
)

print(f"K-S Statistic: {ks_stat:.4f}")
print(f"P-value: {p_value:.6f}")
print(f"Significant: {is_significant}")
```

### Multiple Feature K-S Tests

```python
from drift_detection import validate_multiple_features

results = validate_multiple_features(
    X_base_preprocessed,
    X_drifted_preprocessed,
    features=['Glucose', 'BMI', 'BloodPressure'],
    alpha=0.05
)

for feature, res in results.items():
    sig = "✓" if res['is_significant'] else "✗"
    print(f"{sig} {feature}: K-S={res['ks_statistic']:.4f}, p={res['p_value']:.6f}")
```

### Monotonicity Check

```python
from drift_detection import check_monotonicity

drift_percentages = [0.0, 0.1, 0.2, 0.3, 0.4]
detection_ratios = [1.0, 1.2, 1.5, 2.1, 2.8]

rho, p_val, is_monotonic = check_monotonicity(
    drift_percentages, detection_ratios,
    min_correlation=0.90
)

print(f"Spearman ρ: {rho:.3f}")
print(f"P-value: {p_val:.6f}")
print(f"Monotonic: {is_monotonic}")
```

### Comprehensive Evaluation

```python
from drift_detection import evaluate_drift_detection

results = evaluate_drift_detection(
    model,
    X_base_preprocessed,
    X_drifted_preprocessed,
    feature_for_ks='Glucose'
)

print(f"Baseline rate: {results['baseline_outlier_rate']:.1%}")
print(f"Drifted rate: {results['drifted_outlier_rate']:.1%}")
print(f"Detection ratio: {results['detection_ratio']:.2f}x")
print(f"K-S p-value: {results['ks_p_value']:.6f}")
print(f"Interpretation: {results['detection_ratio_interpretation']}")
```

---

## Experiment Orchestration

### Single Drift Experiment

```python
from drift_detection import run_drift_detection_experiment

results = run_drift_detection_experiment(
    model=model,
    X_test_raw=X_test_raw,
    X_test_preprocessed=X_test_preprocessed,
    pipeline=pipeline,
    feature='Glucose',
    drift_percentage=0.40,
    baseline_outlier_rate=0.048,
    baseline_feature_values=X_base_raw['Glucose'].dropna().values,
    algorithm_name='OCSVM'
)

print(results)
# {'algorithm': 'OCSVM',
#  'feature': 'Glucose',
#  'drift_percentage': 0.4,
#  'outlier_rate_original': 0.048,
#  'outlier_rate_drifted': 0.22,
#  'detection_ratio': 4.58,
#  'ks_statistic': 0.234,
#  'ks_p_value': 3.2e-8,
#  'ks_is_significant': True}
```

### Univariate Drift Sweep

```python
from drift_detection import run_univariate_drift_sweep

results_df = run_univariate_drift_sweep(
    model=ocsvm,
    X_baseline_preprocessed=X_base_prep,
    X_test_raw=X_test_raw,
    X_test_preprocessed=X_test_prep,
    pipeline=pipeline,
    feature='Glucose',
    drift_percentages=[0.0, 0.1, 0.2, 0.3, 0.4],
    algorithm_name='OCSVM'
)

print(results_df)
#    algorithm feature drift_percentage  outlier_rate_original  outlier_rate_drifted  detection_ratio  ks_statistic  ks_p_value  ks_is_significant
# 0  OCSVM     Glucose            0.0  0.048                0.050                  1.04            NaN            NaN            False
# 1  OCSVM     Glucose            0.1  0.048                0.095                  1.98            0.067          0.021            True
# 2  OCSVM     Glucose            0.2  0.048                0.131                  2.73            0.102          0.0001           True
# 3  OCSVM     Glucose            0.3  0.048                0.182                  3.79            0.151          1.2e-6           True
# 4  OCSVM     Glucose            0.4  0.048                0.223                  4.64            0.198          3.4e-8           True
```

---

## Complete Workflows

### Workflow 1: Gradual Drift Detection (OCSVM)

```python
from drift_detection import *

# Step 1: Load and prepare data
X_raw, y = load_raw_data('data/processed/pima_step1_clean.csv')
X_raw = create_missingness_flags(X_raw)

X_base_raw, X_test_raw, y_base, y_test = temporal_train_test_split(
    X_raw, y, test_fraction=0.30
)

# Step 2: Preprocess
pipeline = PreprocessingPipeline(imputation_strategy='median')
X_base_prep = pipeline.fit_transform(X_base_raw)
X_test_prep = pipeline.transform(X_test_raw)

# Step 3: Train OCSVM
ocsvm = fit_ocsvm(X_base_prep, gamma=0.1)
baseline_rate = get_outlier_rate(ocsvm, X_base_prep)

# Step 4: Test gradual drift
X_test_drifted_raw = simulate_gradual_drift(
    X_test_raw, 'Glucose',
    drift_percentage=0.40,
    duration=len(X_test_raw)
)
X_test_drifted_prep = pipeline.transform(X_test_drifted_raw)
drifted_rate = get_outlier_rate(ocsvm, X_test_drifted_prep)

# Step 5: Evaluate
detection_ratio = calculate_detection_ratio(baseline_rate, drifted_rate)
ks_stat, p_val, is_sig = validate_with_ks_test(
    X_base_raw['Glucose'].dropna().values,
    X_test_drifted_raw['Glucose'].dropna().values
)

print(f"Detection Ratio: {detection_ratio:.2f}x")
print(f"K-S Test: {ks_stat:.4f} (p={p_val:.6f}, significant={is_sig})")
```

### Workflow 2: Abrupt Drift Detection (Isolation Forest)

```python
from drift_detection import *

# Step 1: Load and prepare data (same as above)
X_raw, y = load_raw_data('data/processed/pima_step1_clean.csv')
X_raw = create_missingness_flags(X_raw)

X_base_raw, X_test_raw, y_base, y_test = temporal_train_test_split(
    X_raw, y, test_fraction=0.30
)

# Step 2: Preprocess
pipeline = PreprocessingPipeline(imputation_strategy='median')
X_base_prep = pipeline.fit_transform(X_base_raw)
X_test_prep = pipeline.transform(X_test_raw)

# Step 3: Train Isolation Forest
iforest = fit_isolation_forest(
    X_base_prep,
    n_estimators=100,
    max_samples=256,
    contamination=0.05
)
baseline_rate = get_outlier_rate(iforest, X_base_prep)

# Step 4: Compute baseline statistics and apply abrupt drift
X_base_clean = X_base_raw.dropna()
X_base_stats = {
    'Glucose': {
        'f_min': X_base_clean['Glucose'].min(),
        'f_range': X_base_clean['Glucose'].max() - X_base_clean['Glucose'].min()
    },
    'BMI': {
        'f_min': X_base_clean['BMI'].min(),
        'f_range': X_base_clean['BMI'].max() - X_base_clean['BMI'].min()
    }
}

X_test_drifted_raw = apply_minmax_drift(
    X_test_raw, X_base_stats,
    features=['Glucose', 'BMI'],
    shift_f=0.4, range_f=1.5
)
X_test_drifted_prep = pipeline.transform(X_test_drifted_raw)
drifted_rate = get_outlier_rate(iforest, X_test_drifted_prep)

# Step 5: Evaluate
detection_ratio = calculate_detection_ratio(baseline_rate, drifted_rate)

print(f"Detection Ratio: {detection_ratio:.2f}x")
```

### Workflow 3: Multi-Feature Evaluation

```python
from drift_detection import *

# Setup (load, split, preprocess as above)
# Train models
ocsvm = fit_ocsvm(X_base_prep, gamma=0.1)
iforest = fit_isolation_forest(X_base_prep)

features_to_test = ['Glucose', 'BMI', 'BloodPressure']
drift_levels = [0.1, 0.2, 0.3, 0.4]

results = []
for feature in features_to_test:
    for drift_pct in drift_levels:
        # Gradual drift
        X_drifted = simulate_gradual_drift(
            X_test_raw, feature, drift_percentage=drift_pct
        )
        X_drifted_prep = pipeline.transform(X_drifted)
        
        # Evaluate both models
        for model, name in [(ocsvm, 'OCSVM'), (iforest, 'IForest')]:
            baseline_rate = get_outlier_rate(model, X_base_prep)
            drifted_rate = get_outlier_rate(model, X_drifted_prep)
            detection_ratio = calculate_detection_ratio(baseline_rate, drifted_rate)
            
            results.append({
                'model': name,
                'feature': feature,
                'drift': drift_pct,
                'detection_ratio': detection_ratio
            })

results_df = pd.DataFrame(results)
print(results_df.pivot_table(
    values='detection_ratio',
    index=['feature', 'drift'],
    columns='model'
))
```

---

## Notes

- **Data Leakage Prevention**: Always fit preprocessing pipeline on baseline only
- **Clinical Ranges**: Features are clipped to realistic bounds to prevent unrealistic values
- **NaN Handling**: Drift functions preserve NaN values automatically
- **Drift Types**:
  - **Gradual**: Linear increase over time; detection ratio typically 1.5-4.5x
  - **Abrupt**: Immediate distribution shift; detection ratio typically 1.2-3.0x
- **Contamination/Nu Parameter**:
  - Gradual: 0.05 (5% baseline anomalies)
  - Abrupt: 0.20 (20% baseline anomalies)

"""
Data Loading and Preparation Module
====================================
Handles raw data loading, missingness flag creation, and temporal train/test splitting.
Prevents data leakage by splitting BEFORE preprocessing.

Key Design: Temporal 70/30 split on original raw data (no scaling/imputation applied yet)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List


def load_raw_data(data_path: str) -> pd.DataFrame:
    """
    Load raw CSV data from disk.

    Args:
        data_path: Path to CSV file (e.g., "data/processed/pima_step1_clean.csv")

    Returns:
        DataFrame with raw, unimputed data

    Raises:
        FileNotFoundError: If data_path doesn't exist
    """
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(path)
    return df


def create_missingness_flags(
    df: pd.DataFrame,
    cols_with_missing: List[str],
    zero_is_missing: bool = True,
) -> pd.DataFrame:
    """
    Create binary indicator columns for each feature with missing values.

    Missingness in this project is encoded two different ways depending on which
    file you are holding:

    * ``data/raw/*.csv``            - missing values are sentinel **zeros**.
    * ``data/processed/*_step1_clean.csv`` - the cleaning step has already
      converted those sentinels to **NaN**.

    Both are treated as missing here. This matters: an earlier version of this
    function tested ``df[col] == 0`` only, which silently produced all-zero
    indicator columns when run against the ``step1_clean`` files that every
    notebook, test and README example uses - quietly discarding five of the
    thirteen features and moving the Isolation Forest zero-drift control rate
    from the published 2.2% (5/231) to 4.8% (11/231). See docs/AUDIT.md.

    Args:
        df: Raw DataFrame
        cols_with_missing: Columns known to carry missing values,
                          e.g. ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        zero_is_missing: Also treat exact zeros as missing (default True). Set
                        False for a feature where zero is a legitimate value.

    Returns:
        DataFrame with new binary columns added (e.g., "Glucose_is_missing")

    Example:
        >>> df = load_raw_data("data/processed/pima_step1_clean.csv")
        >>> df = create_missingness_flags(df, ["Glucose", "BloodPressure", "Insulin"])
        >>> df["Glucose_is_missing"].sum()
        5
    """
    df = df.copy()

    for col in cols_with_missing:
        flag = df[col].isna()
        if zero_is_missing:
            flag = flag | (df[col] == 0)
        df[f"{col}_is_missing"] = flag.astype(int)

    return df


def temporal_train_test_split(
    df: pd.DataFrame,
    features_cols: List[str],
    target_col: str = "Outcome",
    test_fraction: float = 0.3
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, int]:
    """
    Perform temporal train/test split on raw data.

    CRITICAL: This split happens BEFORE preprocessing to prevent data leakage.
    The preprocessing (imputation, scaling) is fit on baseline data only.

    Args:
        df: Raw DataFrame with all features
        features_cols: List of feature column names (e.g., all columns except target)
        target_col: Name of target variable column (default "Outcome")
        test_fraction: Fraction for test set (default 0.3 → 70/30 split)

    Returns:
        Tuple of (X_base_raw, X_test_raw, y_base, y_test, split_point)
        - X_base_raw: 70% of data, features only, unimputed
        - X_test_raw: 30% of data, features only, unimputed
        - y_base: Baseline targets
        - y_test: Test targets
        - split_point: Index where split occurred

    Raises:
        AssertionError: If baseline and test sets overlap

    Example:
        >>> df = load_raw_data("data/processed/pima_step1_clean.csv")
        >>> df = create_missingness_flags(df, ["Glucose", "BloodPressure", ...])
        >>> features = [col for col in df.columns if col != "Outcome"]
        >>> X_base, X_test, y_base, y_test, split = temporal_train_test_split(
        ...     df, features, target_col="Outcome", test_fraction=0.3
        ... )
        >>> print(f"Baseline: {len(X_base)} samples, Test: {len(X_test)} samples")
    """
    X = df[features_cols].copy()
    y = df[target_col].copy()

    # Calculate split point
    split_point = int((1 - test_fraction) * len(X))

    # Non-overlapping temporal split
    X_base_raw = X.iloc[:split_point].copy()
    X_test_raw = X.iloc[split_point:].copy()
    y_base = y.iloc[:split_point].copy()
    y_test = y.iloc[split_point:].copy()

    # Validate no overlap
    assert X_base_raw.index.max() < X_test_raw.index.min(), \
        "Baseline and test sets overlap - temporal split failed!"

    return X_base_raw, X_test_raw, y_base, y_test, split_point


def identify_feature_types(
    X: pd.DataFrame,
    indicator_suffix: str = "_is_missing"
) -> Tuple[List[str], List[str]]:
    """
    Separate features into continuous and binary indicator columns.

    Continuous columns: Actual features that need imputation and scaling
    Indicator columns: Binary flags (0/1) that should remain unscaled

    Args:
        X: Feature DataFrame
        indicator_suffix: Suffix used for indicator columns (default "_is_missing")

    Returns:
        Tuple of (continuous_cols, indicator_cols)

    Example:
        >>> X = df[feature_names]
        >>> continuous, indicators = identify_feature_types(X)
        >>> print(f"Continuous: {continuous}")
        >>> print(f"Indicators: {indicators}")
    """
    indicator_cols = [col for col in X.columns if col.endswith(indicator_suffix)]
    continuous_cols = [col for col in X.columns if col not in indicator_cols]

    return continuous_cols, indicator_cols


# ============================================================================
# METADATA FOR PIMA DATASET
# ============================================================================

PIMA_FEATURES = [
    'Pregnancies',          # Number of times pregnant
    'Glucose',              # Plasma glucose concentration (2 hours after glucose tolerance test)
    'BloodPressure',        # Diastolic blood pressure (mm Hg)
    'SkinThickness',        # Triceps skin fold thickness (mm)
    'Insulin',              # 2-Hour serum insulin (mu U/ml)
    'BMI',                  # Body mass index (weight in kg/(height in m)^2)
    'DiabetesPedigreeFunction',  # Diabetes pedigree function
    'Age'                   # Age (years)
]

PIMA_TARGET = 'Outcome'  # 1 = positive diabetes test, 0 = negative

PIMA_COLS_WITH_MISSING = [
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI'
]

PIMA_CONTINUOUS_COLS = PIMA_FEATURES  # All 8 features are continuous before adding indicators

# After adding missingness flags, these will be added:
# ['Glucose_is_missing', 'BloodPressure_is_missing', 'SkinThickness_is_missing',
#  'Insulin_is_missing', 'BMI_is_missing']

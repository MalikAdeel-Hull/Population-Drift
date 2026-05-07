"""
Data Preprocessing Pipeline
============================
Handles missing value imputation and feature scaling.

Key Design Principle: ALL transformers (imputer, scaler) are fit ONLY on the baseline
(70% training) data. Then applied to test data using those fitted transformers.
This prevents data leakage and ensures realistic preprocessing.

Pipeline Order:
1. Temporal split (in data.py) - BEFORE preprocessing
2. Create missingness flags (in data.py)
3. Impute missing values (median on baseline only)
4. Scale continuous features (z-score on baseline only)
5. Concatenate scaled continuous + unscaled indicators

Binary indicators are NOT scaled because they must remain as 0/1.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List


class PreprocessingPipeline:
    """
    Manages imputation and scaling for drift detection experiments.

    Ensures that transformers are fit on baseline data only, preventing leakage.
    """

    def __init__(self):
        """Initialize the preprocessing pipeline."""
        self.imputer = None
        self.scaler = None
        self.continuous_cols = None
        self.indicator_cols = None
        self.is_fitted = False

    def fit(
        self,
        X_base_raw: pd.DataFrame,
        continuous_cols: List[str],
        indicator_cols: List[str]
    ) -> 'PreprocessingPipeline':
        """
        Fit imputer and scaler on baseline data ONLY.

        Args:
            X_base_raw: Baseline (70%) raw, unimputed data
            continuous_cols: Names of continuous feature columns
            indicator_cols: Names of binary indicator columns (will not be scaled)

        Returns:
            self (for method chaining)

        Example:
            >>> pipeline = PreprocessingPipeline()
            >>> pipeline.fit(X_base_raw, continuous_cols, indicator_cols)
        """
        self.continuous_cols = continuous_cols
        self.indicator_cols = indicator_cols

        # Step 1: Fit imputer on baseline continuous features
        self.imputer = SimpleImputer(strategy="median")
        self.imputer.set_output(transform="pandas")
        self.imputer.fit(X_base_raw[continuous_cols])

        # Step 2: Impute baseline and fit scaler on imputed baseline
        X_base_imputed_cont = self.imputer.transform(X_base_raw[continuous_cols])

        self.scaler = StandardScaler()
        self.scaler.set_output(transform="pandas")
        self.scaler.fit(X_base_imputed_cont)

        self.is_fitted = True
        return self

    def transform(self, X_raw: pd.DataFrame) -> pd.DataFrame:
        """
        Apply preprocessing to any dataset using fitted transformers.

        Steps:
        1. Impute continuous features using fitted imputer
        2. Scale imputed continuous features using fitted scaler
        3. Concatenate scaled continuous + unscaled indicators

        Args:
            X_raw: Raw, unimputed data (same structure as training data)

        Returns:
            Preprocessed DataFrame with imputed + scaled continuous features,
            and unscaled binary indicators

        Raises:
            RuntimeError: If pipeline not yet fitted

        Example:
            >>> X_test_preprocessed = pipeline.transform(X_test_raw)
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform. Call .fit() first.")

        # Impute continuous features
        X_imputed_cont = self.imputer.transform(X_raw[self.continuous_cols])

        # Scale imputed continuous features
        X_scaled_cont = self.scaler.transform(X_imputed_cont)

        # Concatenate: scaled continuous + unscaled indicators
        X_preprocessed = pd.concat(
            [X_scaled_cont, X_raw[self.indicator_cols]],
            axis=1
        )

        return X_preprocessed

    def fit_transform(
        self,
        X_base_raw: pd.DataFrame,
        continuous_cols: List[str],
        indicator_cols: List[str]
    ) -> pd.DataFrame:
        """
        Fit on baseline data and return preprocessed baseline.

        Args:
            X_base_raw: Baseline raw data
            continuous_cols: Continuous feature names
            indicator_cols: Indicator feature names

        Returns:
            Preprocessed baseline DataFrame

        Example:
            >>> X_base_preprocessed = pipeline.fit_transform(X_base_raw, cont, ind)
        """
        self.fit(X_base_raw, continuous_cols, indicator_cols)
        return self.transform(X_base_raw)


def preprocess_baseline_and_test(
    X_base_raw: pd.DataFrame,
    X_test_raw: pd.DataFrame,
    continuous_cols: List[str],
    indicator_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame, PreprocessingPipeline]:
    """
    Convenience function: Create pipeline, fit on baseline, apply to both baseline and test.

    Args:
        X_base_raw: Baseline (70%) raw data
        X_test_raw: Test (30%) raw data
        continuous_cols: Continuous feature names
        indicator_cols: Indicator feature names

    Returns:
        Tuple of (X_base_preprocessed, X_test_preprocessed, pipeline)

    Example:
        >>> X_base_prep, X_test_prep, pipeline = preprocess_baseline_and_test(
        ...     X_base_raw, X_test_raw, continuous_cols, indicator_cols
        ... )
    """
    pipeline = PreprocessingPipeline()
    pipeline.fit(X_base_raw, continuous_cols, indicator_cols)

    X_base_preprocessed = pipeline.transform(X_base_raw)
    X_test_preprocessed = pipeline.transform(X_test_raw)

    return X_base_preprocessed, X_test_preprocessed, pipeline


# ============================================================================
# QUICK REFERENCE: CONTINUOUS VS INDICATOR COLUMNS
# ============================================================================

# For PIMA dataset after missingness flags are added:
#
# Continuous (need imputation + scaling):
#   ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
#    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
#
# Indicators (remain unscaled):
#   ['Glucose_is_missing', 'BloodPressure_is_missing', 'SkinThickness_is_missing',
#    'Insulin_is_missing', 'BMI_is_missing']
#
# Final feature count after preprocessing: 8 continuous + 5 indicators = 13 features

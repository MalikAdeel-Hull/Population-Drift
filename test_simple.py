#!/usr/bin/env python
"""Simple diagnostic test for drift detection modules"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

print(f"Python path: {sys.path[0]}")
print(f"Current dir: {Path.cwd()}")

# Test imports
try:
    from drift_detection import load_raw_data, PIMA_COLS_WITH_MISSING
    print("✓ Imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test data loading
try:
    df = load_raw_data('data/processed/pima_step1_clean.csv')
    print(f"✓ Data loaded: {df.shape}")
except Exception as e:
    print(f"✗ Load failed: {e}")
    sys.exit(1)

# Test missingness flags
try:
    from drift_detection import create_missingness_flags
    df = create_missingness_flags(df, PIMA_COLS_WITH_MISSING)
    print(f"✓ Missingness flags created: {df.shape}")
except Exception as e:
    print(f"✗ Flags failed: {e}")
    sys.exit(1)

# Test split
try:
    from drift_detection import temporal_train_test_split
    features = [col for col in df.columns if col != 'Outcome']
    X_base, X_test, y_base, y_test, split = temporal_train_test_split(
        df, features, target_col='Outcome', test_fraction=0.30
    )
    print(f"✓ Split successful: base={X_base.shape}, test={X_test.shape}")
except Exception as e:
    print(f"✗ Split failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test preprocessing
try:
    from drift_detection import PreprocessingPipeline, identify_feature_types
    continuous_cols, indicator_cols = identify_feature_types(X_base)
    pipeline = PreprocessingPipeline()
    X_base_prep = pipeline.fit_transform(X_base, continuous_cols, indicator_cols)
    X_test_prep = pipeline.transform(X_test)
    print(f"✓ Preprocessing successful: base={X_base_prep.shape}, test={X_test_prep.shape}")
except Exception as e:
    print(f"✗ Preprocessing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test OCSVM
try:
    from drift_detection import fit_ocsvm, get_outlier_rate
    ocsvm = fit_ocsvm(X_base_prep, gamma=0.1)
    baseline_rate = get_outlier_rate(ocsvm, X_base_prep)
    print(f"✓ OCSVM trained: baseline_rate={baseline_rate:.4f}")
except Exception as e:
    print(f"✗ OCSVM failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test gradual drift
try:
    from drift_detection import simulate_gradual_drift
    X_drifted = simulate_gradual_drift(X_test, 'Glucose', drift_percentage=0.40)
    print(f"✓ Gradual drift successful: {X_drifted.shape}")
except Exception as e:
    print(f"✗ Gradual drift failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test abrupt drift
try:
    from drift_detection import apply_minmax_drift

    # Compute baseline stats
    X_clean = X_base.dropna()
    X_base_stats = {
        'Glucose': {
            'f_min': X_clean['Glucose'].min(),
            'f_range': X_clean['Glucose'].max() - X_clean['Glucose'].min()
        }
    }

    X_drifted_abrupt = apply_minmax_drift(
        X_test,
        X_base_stats,
        features=['Glucose'],
        shift_f=0.4,
        range_f=1.5,
        verbose=False
    )
    print(f"✓ Abrupt drift successful: {X_drifted_abrupt.shape}")
except Exception as e:
    print(f"✗ Abrupt drift failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*70)
print("ALL DIAGNOSTICS PASSED ✓")
print("="*70)

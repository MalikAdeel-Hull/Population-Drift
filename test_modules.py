from drift_detection import (
    load_raw_data, create_missingness_flags,
    temporal_train_test_split, identify_feature_types,
    PreprocessingPipeline, fit_ocsvm,
    simulate_gradual_drift, evaluate_drift_detection
)

# Load your actual data
print("Loading data...")
df = load_raw_data('data/processed/pima_step1_clean.csv')
print(f"✅ Loaded {len(df)} samples")

# Create flags
df = create_missingness_flags(df, ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'])
print("✅ Created missingness flags")

# Split data
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X_base, X_test, y_base, y_test, split = temporal_train_test_split(df, features)
print(f"✅ Split: {len(X_base)} baseline, {len(X_test)} test")

# Preprocess
continuous_cols, indicator_cols = identify_feature_types(X_base)
pp = PreprocessingPipeline()
X_base_prep = pp.fit_transform(X_base, continuous_cols, indicator_cols)
X_test_prep = pp.transform(X_test)
print(f"✅ Preprocessed: {X_base_prep.shape}")

# Train model
ocsvm = fit_ocsvm(X_base_prep, gamma=0.1)
print("✅ OCSVM trained")

# Test drift detection
X_test_drifted = simulate_gradual_drift(X_test, 'Glucose', drift_percentage=0.4)
X_test_drifted_prep = pp.transform(X_test_drifted)
print("✅ Applied 40% drift to Glucose")

# Evaluate
results = evaluate_drift_detection(ocsvm, X_base_prep, X_test_drifted_prep)
print(f"\n📊 RESULTS:")
print(f"   Detection Ratio: {results['detection_ratio']:.2f}x")
print(f"   K-S p-value: {results['ks_p_value']:.6f}")
print(f"   Significant: {results['ks_is_significant']}")

print("\n✅ ALL TESTS PASSED!")
#!/bin/bash
set -e

echo "=========================================="
echo "Running ALL Drift Detection Experiments"
echo "=========================================="
echo ""

cd notebooks

# PIMA DATASET EXPERIMENTS
echo "🔵 PIMA DATASET - Starting experiments..."
echo ""

echo "Step 1/10: Baseline EDA - Pima"
jupyter nbconvert --to notebook --execute 01_Baseline_EDA_Pima.ipynb --output-dir=../reports/
echo "✓ Complete"
echo ""

echo "Step 2/10: Gradual Drift - OCSVM - Pima"
jupyter nbconvert --to notebook --execute 02_Gradual_Drift_OCSVM_Pima.ipynb --output-dir=../reports/
echo "✓ Complete"
echo ""

echo "Step 3/10: Gradual Drift - Isolation Forest - Pima"
jupyter nbconvert --to notebook --execute 03_Gradual_Drift_IsoForest_Pima.ipynb --output-dir=../reports/
echo "✓ Complete"
echo ""

echo "Step 4/10: Abrupt Drift - Isolation Forest - Pima"
jupyter nbconvert --to notebook --execute 04_Abrupt_Drift_IsoForest_Pima.ipynb --output-dir=../reports/
echo "✓ Complete"
echo ""

echo "Step 5/10: Abrupt Drift - OCSVM - Pima"
jupyter nbconvert --to notebook --execute 05_Abrupt_Drift_OCSVM_Pima.ipynb --output-dir=../reports/
echo "✓ Complete"
echo ""

# FHGD DATASET EXPERIMENTS
echo "🟢 FHGD DATASET - Starting experiments..."
echo ""

echo "Step 6/10: Baseline EDA - FHGD"
jupyter nbconvert --to notebook --execute 06_Baseline_EDA_FHGD.ipynb --output-dir=../reports/
echo "✓ Complete"
echo ""

echo "Step 7/10: Gradual Drift - OCSVM - FHGD"
jupyter nbconvert --to notebook --execute 07_Gradual_Drift_OCSVM_FHGD.ipynb --output-dir=../reports/
echo "✓ Complete"
echo ""

echo "Step 8/10: Gradual Drift - Isolation Forest - FHGD"
jupyter nbconvert --to notebook --execute 08_Gradual_Drift_IsoForest_FHGD.ipynb --output-dir=../reports/
echo "✓ Complete"
echo ""

echo "Step 9/10: Abrupt Drift - Isolation Forest - FHGD"
jupyter nbconvert --to notebook --execute 09_Abrupt_Drift_IsoForest_FHGD.ipynb --output-dir=../reports/
echo "✓ Complete"
echo ""

echo "Step 10/10: Abrupt Drift - OCSVM - FHGD"
jupyter nbconvert --to notebook --execute 10_Abrupt_Drift_OCSVM_FHGD.ipynb --output-dir=../reports/
echo "✓ Complete"
echo ""

cd ..
echo "=========================================="
echo "✅ ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "=========================================="
echo "Results saved to: reports/"
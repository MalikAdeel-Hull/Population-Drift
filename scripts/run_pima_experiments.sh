#!/bin/bash
set -e

echo "=========================================="
echo "Running PIMA Dataset Experiments"
echo "=========================================="
echo ""

cd notebooks

echo "Step 1/5: Baseline EDA - Pima"
jupyter nbconvert --to notebook --execute 01_Baseline_EDA_Pima.ipynb --output-dir=../reports/
echo "✓ Complete"
echo ""

echo "Step 2/5: Gradual Drift - OCSVM - Pima"
jupyter nbconvert --to notebook --execute 02_Gradual_Drift_OCSVM_Pima.ipynb --output-dir=../reports/
echo "✓ Complete"
echo ""

echo "Step 3/5: Gradual Drift - Isolation Forest - Pima"
jupyter nbconvert --to notebook --execute 03_Gradual_Drift_IsoForest_Pima.ipynb --output-dir=../reports/
echo "✓ Complete"
echo ""

echo "Step 4/5: Abrupt Drift - Isolation Forest - Pima"
jupyter nbconvert --to notebook --execute 04_Abrupt_Drift_IsoForest_Pima.ipynb --output-dir=../reports/
echo "✓ Complete"
echo ""

echo "Step 5/5: Abrupt Drift - OCSVM - Pima"
jupyter nbconvert --to notebook --execute 05_Abrupt_Drift_OCSVM_Pima.ipynb --output-dir=../reports/
echo "✓ Complete"
echo ""

cd ..
echo "=========================================="
echo "✅ PIMA EXPERIMENTS COMPLETED!"
echo "=========================================="
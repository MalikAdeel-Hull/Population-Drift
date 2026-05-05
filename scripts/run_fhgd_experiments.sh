#!/bin/bash
set -e

echo "=========================================="
echo "Running FHGD Dataset Experiments"
echo "=========================================="
echo ""

cd notebooks

echo "Step 1/5: Baseline EDA - FHGD"
jupyter nbconvert --to notebook --execute 06_Baseline_EDA_FHGD.ipynb --output-dir=../reports/
echo "✓ Complete"
echo ""

echo "Step 2/5: Gradual Drift - OCSVM - FHGD"
jupyter nbconvert --to notebook --execute 07_Gradual_Drift_OCSVM_FHGD.ipynb --output-dir=../reports/
echo "✓ Complete"
echo ""

echo "Step 3/5: Gradual Drift - Isolation Forest - FHGD"
jupyter nbconvert --to notebook --execute 08_Gradual_Drift_IsoForest_FHGD.ipynb --output-dir=../reports/
echo "✓ Complete"
echo ""

echo "Step 4/5: Abrupt Drift - Isolation Forest - FHGD"
jupyter nbconvert --to notebook --execute 09_Abrupt_Drift_IsoForest_FHGD.ipynb --output-dir=../reports/
echo "✓ Complete"
echo ""

echo "Step 5/5: Abrupt Drift - OCSVM - FHGD"
jupyter nbconvert --to notebook --execute 10_Abrupt_Drift_OCSVM_FHGD.ipynb --output-dir=../reports/
echo "✓ Complete"
echo ""

cd ..
echo "=========================================="
echo "✅ FHGD EXPERIMENTS COMPLETED!"
echo "=========================================="
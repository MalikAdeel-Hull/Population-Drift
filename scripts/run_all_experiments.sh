#!/bin/bash
# Execute the full analysis pipeline end to end.
#
# The four notebooks each cover BOTH cohorts (Pima and FHGD), so there is no
# longer a per-dataset split. Run from the repository root.
set -e

cd "$(dirname "$0")/.."

OUT=reports/executed
mkdir -p "$OUT"

echo "=========================================="
echo "Drift Detection - full pipeline"
echo "=========================================="

for nb in 01_Baseline_EDA 02_Gradual_Drift 03_Abrupt_Drift 04_Cross_Cohort_Analysis; do
    echo ""
    echo ">>> $nb"
    jupyter nbconvert --to notebook --execute \
        --ExecutePreprocessor.timeout=1800 \
        "notebooks/$nb.ipynb" \
        --output-dir="../$OUT"
done

echo ""
echo ">>> Verifying published detection ratios"
python scripts/reproduce_paper.py

echo ""
echo "=========================================="
echo "Done. Executed notebooks in $OUT/"
echo "=========================================="

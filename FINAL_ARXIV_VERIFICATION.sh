#!/bin/bash

echo "════════════════════════════════════════════════════════"
echo "FINAL ARXIV READINESS CHECK"
echo "════════════════════════════════════════════════════════"
echo ""

PASS=0
FAIL=0

echo "1. CHECKING FILE NAMES"
[ -f "README.md" ] && echo "✅ README.md" || (echo "❌ README.md MISSING"; ((FAIL++)))
[ -f "LICENSE" ] && echo "✅ LICENSE" || (echo "❌ LICENSE MISSING"; ((FAIL++)))
[ -f "CITATION.cff" ] && echo "✅ CITATION.cff" || (echo "❌ CITATION.cff MISSING"; ((FAIL++)))
[ -f "setup.py" ] && echo "✅ setup.py" || (echo "❌ setup.py MISSING"; ((FAIL++)))
[ ! -f "ReadMe.md" ] && echo "✅ ReadMe.md removed" || (echo "❌ ReadMe.md still exists"; ((FAIL++)))
[ ! -f "Setup.py" ] && echo "✅ Setup.py removed" || (echo "❌ Setup.py still exists"; ((FAIL++)))

echo ""
echo "2. CHECKING NOTEBOOKS (need 10)"
notebook_count=$(ls notebooks/[0-9][0-9]_*.ipynb 2>/dev/null | wc -l)
echo "Found $notebook_count notebooks"
[ "$notebook_count" -eq 10 ] && echo "✅ All 10 notebooks present" || (echo "❌ Only $notebook_count found"; ((FAIL++)))

echo ""
echo "3. CHECKING DATA FILES"
[ -f "data/raw/diabetes.csv" ] && echo "✅ Pima raw data" || (echo "❌ Pima missing"; ((FAIL++)))
[ -f "data/processed/pima_step1_clean.csv" ] && echo "✅ Pima processed" || (echo "❌ Pima processed missing"; ((FAIL++)))

echo ""
echo "4. CHECKING SCRIPTS"
[ -f "scripts/run_all_experiments.sh" ] && echo "✅ run_all_experiments.sh" || (echo "❌ Missing"; ((FAIL++)))
[ -f "scripts/run_pima_experiments.sh" ] && echo "✅ run_pima_experiments.sh" || (echo "❌ Missing"; ((FAIL++)))

echo ""
echo "5. CHECKING requirements.txt"
pkg_count=$(wc -l < requirements.txt)
echo "Packages: $pkg_count"
if ! grep -q "tensorflow\|keras\|torch" requirements.txt; then
    echo "✅ No unnecessary packages"
else
    echo "❌ Found tensorflow/keras/torch"
    ((FAIL++))
fi

echo ""
echo "6. GIT STATUS"
uncommitted=$(git status --porcelain 2>/dev/null | wc -l)
[ "$uncommitted" -eq 0 ] && echo "✅ All committed" || (echo "❌ $uncommitted uncommitted changes"; ((FAIL++)))

echo ""
echo "════════════════════════════════════════════════════════"
if [ $FAIL -eq 0 ]; then
    echo "✅ ALL CHECKS PASSED - READY FOR ARXIV!"
else
    echo "❌ $FAIL ISSUES FOUND - FIX BEFORE SUBMISSION"
fi
echo "════════════════════════════════════════════════════════"
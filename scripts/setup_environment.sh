#!/bin/bash
# Create a virtualenv and install the package with its development extras.
set -e

cd "$(dirname "$0")/.."

python -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

pip install --upgrade pip setuptools wheel
pip install -e '.[dev]'

echo ""
echo "Environment ready. Activate with: source .venv/bin/activate"
echo "Verify with:                      python scripts/reproduce_paper.py"

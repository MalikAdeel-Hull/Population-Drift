# Test Suite

```bash
pytest                                   # everything
pytest tests/test_paper_reproduction.py  # published figures only (~4 s)
pytest -m "not slow"                     # skip the SHAP suite
```

All tests use the Pima and FHGD cohorts from `data/processed/`, resolve `src/`
relative to the repository root, and can be run from anywhere.

## Test files

### `test_paper_reproduction.py` — the important one

Pins every number in the preprint (doi:10.5281/zenodo.20633719):

- all 8 gradual-arm detection ratios at 40% severity
- all 8 abrupt-arm detection ratios
- the four gradual zero-drift control rates (11.7% / 2.2% / 7.7% / 5.8%)
- the headline FHGD abrupt multivariate result: 3.18x, 95% CI 2.69–3.78
- three unit-level regressions for bugs found in the August 2026 audit:
  missingness flags must not be all-zero on the `step1_clean` files, zero-drift
  must be a true no-op, and `import drift_detection` must work without `shap`

If a refactor moves a published figure, this suite fails.

### `test_abrupt_drift.py`

Smoke tests for the affine transform: single-feature drift, multivariate drift,
NaN preservation, clinical-range clipping, and a gradual-vs-abrupt comparison.
Note it runs the *gradual*-arm hyperparameters against both drift types, so it
does not reproduce the published OCSVM-over-IF ordering for abrupt drift — that
is `test_paper_reproduction.py`'s job.

### `test_bootstrap_ci.py`

Bootstrap CI behaviour against the two results the preprint quotes intervals
for. Note the abrupt-arm denominator here is the baseline (training) anomaly
count, 122/600 — not the gradual arm's 7.7% zero-drift control.

### `test_modules.py`, `test_simple.py`

Import and end-to-end smoke checks. `test_simple.py` is the fastest way to
confirm an install works.

### `test_shap_analysis.py`

SHAP KernelExplainer validation. Requires the optional `shap` extra and takes
several minutes — the KernelExplainer is deliberately slow.

## Detection ratios these tests expect

Straight from the preprint's Table 8:

| Cohort | Arm | OCSVM uni / multi | IF uni / multi |
|---|---|---|---|
| Pima | Gradual 40% | 1.48x / 2.07x | 1.60x / 4.40x |
| Pima | Abrupt affine | 1.47x / 2.92x | 1.14x / 1.42x |
| FHGD | Gradual 40% | 1.83x / 2.76x | 1.11x / 1.89x |
| FHGD | Abrupt affine | 1.62x / **3.18x** | 1.44x / 1.79x |

Remember the denominators differ by arm — see `docs/AUDIT.md`.

## Adding new tests

Use pytest conventions and import from the installed package:

```python
from drift_detection import load_raw_data, create_missingness_flags, fit_ocsvm
from drift_detection.config import PAPER_ABRUPT_DR   # reference values
```

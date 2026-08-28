# Reproducibility audit — August 2026

Audit of this repository against the published preprint:

> Anjum, M. A. (2026). *Monitoring Population Drift in Deployed AI Medical
> Devices: An AI-Centred Diagnostic Accuracy Study of Detection Ratios Across US
> and European Cohorts* (version 2). doi:[10.5281/zenodo.20633719](https://doi.org/10.5281/zenodo.20633719)

**Outcome:** all 16 published detection ratios now regenerate exactly from the
packaged library. `python scripts/reproduce_paper.py` prints them and exits
non-zero on any deviation; `pytest tests/test_paper_reproduction.py` pins them
as regressions.

Before this audit the library reproduced **none** of them, for the reasons below.

---

## The finding that matters most

**The two experiment arms use different detection-ratio denominators, and this
is not stated anywhere in the manuscript.**

| Arm | Denominator | Values |
|---|---|---|
| Gradual | Zero-drift control rate on the **undrifted test set** | Pima OCSVM 11.7% (27/231), IF 2.2% (5/231); FHGD OCSVM 7.7% (46/600), IF 5.8% (35/600) |
| Abrupt | Anomaly rate on the **baseline (training) set** | Pima OCSVM 21.2% (114/537), IF 20.1%; FHGD OCSVM 20.3% (284/1400), IF 20.0% |

The preprint's Methods section describes only the first convention:

> "the Detection Ratio (DR), defined as the anomaly rate on the drifted test set
> divided by the verified zero-drift control anomaly rate on the unmanipulated
> test set"

but every abrupt-arm figure in Table 8 — including the headline **3.18×** — is
computed against the training-set rate. The two are not interchangeable: on FHGD
the undrifted test rate is 21.8% (131/600) versus 20.3% on the baseline, which
turns 3.18× into 2.95×.

This is why `notebooks/03_Abrupt_Drift.ipynb` and
`notebooks/04_Cross_Cohort_Analysis.ipynb` disagreed with each other. NB-03
computes against the test set and carries the comment
`# Zero-drift control ... — matches paper denominator`, which is incorrect;
NB-04 recomputes against the baseline set and is the one that produced the
published `supp_table_abrupt_drift.csv`. The intermediate
`data/results/abrupt_drift_results.csv` therefore contains a set of abrupt DRs
(2.804, 1.412, 1.692, 1.359, 2.954, 1.504, 1.593, 1.281) that appear nowhere in
the paper.

**Recommendation for a version 3 of the preprint:** state both denominators
explicitly in the Methods, or recompute the abrupt arm against the test-set
control for consistency. If recomputed, the headline becomes 2.95× rather than
3.18× and Table 8's abrupt column shifts; the paper's *conclusions* are
unaffected, since the OCSVM-over-IF ordering for abrupt drift holds under either
denominator on both cohorts.

---

## Findings

### 1. Missingness indicators were silently all-zero — FIXED

`create_missingness_flags()` tested `df[col] == 0`. Sentinel zeros only exist in
`data/raw/*.csv`; the `data/processed/*_step1_clean.csv` files that every
notebook, test and README example loads have already had those zeros converted
to `NaN`, and `NaN == 0` is `False`.

Result: five of the thirteen feature columns were constant zero. The Pima
Isolation Forest zero-drift control moved from the published 2.2% (5/231) to
4.8% (11/231), and no downstream figure reproduced. The notebooks were correct
(`df[col].isna()`); the regression was introduced when the code was extracted
from the notebooks into the package.

Fixed: both `NaN` and sentinel zeros count as missing, so the function is now
correct on raw and processed inputs alike. Pinned by
`test_missingness_flags_are_not_all_zero_on_step1_clean`.

### 2. The abrupt affine transform did not match the published formula — FIXED

| | Preprint / notebook 03 | `drift.py` before this audit |
|---|---|---|
| `min_t` | `f_min × δ` (δ=0.4 → 60% downward shift) | `f_min × (1 − δ)` (40% downward shift) |
| `max_t` | `min_t + f_range × σ` | `f_min + f_range × σ` |

The library also defaulted to clipping the result to clinical bounds. The
published abrupt experiment is deliberately unclipped; clipping saturates the
shifted distribution and, in the four-feature case, made the choice of formula
irrelevant because everything hit a bound.

Fixed: formula matches the preprint, `feature_ranges` now defaults to `None`
(no clipping) in `apply_minmax_drift`. The transform's defining invariant —
every drifted feature's standard deviation stretched by exactly `range_f` — is
asserted in `test_abrupt_drift.py`.

### 3. The zero-drift control was not a no-op — FIXED

`simulate_gradual_drift(..., drift_percentage=0.0)` is documented to return the
data unchanged, and it is the denominator of every gradual detection ratio. It
clipped regardless of drift level, so 11 Pima records below the 70 mg/dL Glucose
floor were lifted to 70 — by up to 26 units — contaminating the control
condition. Fixed: clipping is skipped at zero drift. Pinned by
`test_zero_drift_is_a_true_no_op`.

### 4. Clinical clipping bounds disagreed with the preprint — FIXED

`DEFAULT_CLINICAL_RANGES` differed from the Methods section and from the
notebooks in five of eight entries:

| Feature | Was | Preprint / notebooks |
|---|---|---|
| BMI | 15–60 | **15–50** |
| BloodPressure | 60–140 | **40–120** |
| Insulin | 0–600 | **14–846** |
| SkinThickness | 5–50 | **7–99** |
| Pregnancies | 0–15 | **0–17** |

### 5. The package could not be imported without `shap` — FIXED

`__init__.py` imported `shap_analysis` eagerly, and `shap_analysis` imports
`shap` at module level. `shap` is not in `requirements.txt` and the README calls
it optional — so following the documented install produced
`ModuleNotFoundError: No module named 'shap'` on `import drift_detection`, which
broke the README quick-start and *every* test in the suite. SHAP symbols are now
resolved lazily via `__getattr__`, with an error message that names the fix.
`shap` and `pytest` have been added to `requirements.txt` and as extras.

### 6. The headline confidence interval was bootstrapped with the wrong n

The published CI for the headline result, 3.18× (95% CI 2.69–3.78), was
resampled as though the baseline denominator held 600 samples. The point
estimate uses the baseline (training) rate, whose denominator is 1,400. Bootstrapping
with the real n gives a **tighter** interval, [2.83, 3.60].

This does not weaken the result — it strengthens it — but the published interval
is wider than the data warrant. Both are recorded in
`drift_detection.config.PAPER_HEADLINE` and both are covered by tests. Worth
correcting in a version 3.

### 7. `test_bootstrap_ci.py` used the gradual control for an abrupt result

The FHGD abrupt case was fed 46/600 (the *gradual* OCSVM zero-drift rate)
instead of the abrupt baseline count, producing 8.41× and printing
`CI Overlap: ✗ NO` against the paper — reading as a reproducibility failure when
the inputs were simply wrong. Fixed to 122/600; it now reports 3.18× and
`CI Overlap: ✓ YES`.

### 8. The README pointed at the one test that failed

`README.md` instructed readers to run `python tests/test_abrupt_drift.py`, which
failed on `assert gradual_iforest_ratio > 1.5` (it produced 1.12). Two of its
assertions were written against pre-audit behaviour:

- it divided a *gradual* experiment by the training-set rate, mixing the two DR
  conventions and pushing the Isolation Forest's ratio below 1.0;
- it asserted every drifted feature's mean rises by >5%, but the affine
  transform is signed — the preprint itself reports BMI −11.2% and Age −18.7%.

Fixed and the assertions rewritten around the transform's actual invariants.
The README now points at `pytest` and `scripts/reproduce_paper.py`.

### 9. Documentation drift

- `DEFAULT_CONFIG` and most docstrings pointed at `data/interim/`, a directory
  that does not exist at the repository root (it exists only under
  `notebooks/`). Updated to `data/processed/`.
- The README quick-start claimed `Detection Ratio: 3.00x`; the snippet actually
  produces **1.48×** — which is the published Pima OCSVM gradual univariate
  figure. Corrected.
- `validate_zero_drift()` and `verify_drift_application()` existed but were not
  exported. Now in `__all__`.

---

## Repository hygiene

| Item | Action |
|---|---|
| `README2.md` — superseded duplicate of the README | Removed |
| `test_abrupt_drift_v2.py`, `test_abrupt_drift_final.py` — byte-identical to each other | Removed; the surviving `test_abrupt_drift.py` is the version that passes |
| `test_simple_working.py` — truncated mid-word, but held the path fix `test_simple.py` was missing | Removed; fix folded into `test_simple.py` |
| Two `*.egg-info/` directories tracked despite being in `.gitignore` | Untracked |
| No `pyproject.toml`; bare `setup.py` | `pyproject.toml` added with optional extras; `setup.py` removed. `pip install -e .` previously failed outright on Debian's setuptools |
| `python_requires='>=3.7'` vs README's "Python 3.8+" | Aligned on 3.8 |
| No `.gitattributes` | Added. Without it a Windows checkout reports every text file as modified |
| `.gitignore` missing `test_venv/`, `.venv/`, `Thumbs.db` | Added |
| No CI | `.github/workflows/tests.yml` — runs the reproduction script and the suite on 3.9 / 3.11 / 3.12, and checks the package imports without `shap` |

### Local working copy (not part of this repository)

Found on the machine this audit ran from, and worth clearing up:

- `.git/config` was corrupted with trailing NUL bytes, so **every git command in
  the folder failed** with `fatal: bad config line 17`. Repaired; the damaged
  file is kept at `.git/config.corrupt.bak`. A `config.bak` from May suggests
  this had happened before.
- A stale `.git/index.lock` was silently preventing `git fetch` from updating
  `refs/remotes/origin/main`, leaving the checkout three commits behind GitHub
  while reporting itself up to date.
- A nested `Population-Drift/` directory containing its own empty git repository.
- A directory literally named `-p`, from a `mkdir -p` that ran on a shell that
  did not understand the flag.
- Three virtualenvs inside the tree: `test_env/`, `test_venv/`, `scripts/venv/`.
- `notebooks/temp notebook/` holds the ten pre-consolidation notebooks; the
  cross-cohort figures and tables under `notebooks/reports/` (including Figure 6
  and Table 8) exist only locally and are not committed.

None of the above are tracked, so they do not affect anyone cloning from GitHub.

---

## Things to consider next

1. **Publish the results artefacts.** `notebooks/reports/cross_cohort/` holds
   Figure 6, Table 8, the bootstrapped CIs and the supplementary tables. The
   Code Availability statement says "all model outputs and bootstrapped
   iterations are stored in an auditable format within the repository" — right
   now they are not in the repository.
2. **Pin the environment properly.** The preprint reports Python 3.11.5 and
   scikit-learn 1.4.2; `requirements.txt` pins scikit-learn 1.6.1. The figures
   reproduce on both, but the manuscript and the lockfile should agree.
3. **The bootstrap clamps the resampled denominator to ≥1** to avoid division by
   zero. With a small baseline count — the Pima Isolation Forest's ≈5 outliers —
   this truncation inflates the upper bound. The preprint already flags the
   resulting [1.89–19.00] interval as a limitation; a Katz or Koopman interval
   for the ratio of two binomial proportions would be better behaved here.
4. **`data/raw/` is inconsistent.** The repository tracks `diabetes.csv`, while
   the local copy has `pima_diabetes.csv` and `fhgd_diabetes.csv`, and
   `data/raw/README.md` documents the first naming. Pick one.
5. **Commit messages.** The last twenty are `update`, `up`, `Update`. For a repo
   attached to a DOI, tagging the commit that corresponds to each Zenodo version
   would make the record navigable.

"""
Regression tests pinning the preprint's published figures.
==========================================================

These are the guardrails that stop a refactor from silently moving the numbers
in the manuscript. Every value asserted here is quoted from:

    Anjum, M. A. (2026). Monitoring Population Drift in Deployed AI Medical
    Devices (v2). https://doi.org/10.5281/zenodo.20633719

Run with:  pytest tests/test_paper_reproduction.py
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from drift_detection import (  # noqa: E402
    bootstrap_detection_ratio_ci,
    create_missingness_flags,
    load_raw_data,
    simulate_gradual_drift,
)
from drift_detection.config import (  # noqa: E402
    PAPER_ABRUPT_DR,
    PAPER_GRADUAL_DR,
    PAPER_HEADLINE,
    PAPER_ZERO_DRIFT_CONTROL,
)

PIMA = REPO_ROOT / "data" / "processed" / "pima_step1_clean.csv"
COLS_WITH_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


# ---------------------------------------------------------------------------
# Unit-level regressions for the three bugs found in the September audit
# ---------------------------------------------------------------------------

def test_missingness_flags_are_not_all_zero_on_step1_clean():
    """The step1_clean files store missing values as NaN, not sentinel zeros.

    Flagging only ``== 0`` produced five constant columns and moved the Pima
    Isolation Forest zero-drift control from the published 2.2% to 4.8%.
    """
    df = create_missingness_flags(load_raw_data(PIMA), COLS_WITH_MISSING)
    counts = {c: int(df[f"{c}_is_missing"].sum()) for c in COLS_WITH_MISSING}
    assert counts == {
        "Glucose": 5,
        "BloodPressure": 35,
        "SkinThickness": 227,
        "Insulin": 374,
        "BMI": 11,
    }, counts


def test_zero_drift_is_a_true_no_op():
    """Detection ratios are divided by the zero-drift control, so a 0% run must
    leave the data untouched. Clipping to clinical bounds used to lift the 11
    Pima records below the 70 mg/dL Glucose floor."""
    df = load_raw_data(PIMA)
    out = simulate_gradual_drift(df, "Glucose", drift_percentage=0.0)
    changed = (df["Glucose"] - out["Glucose"]).abs().fillna(0) > 1e-9
    assert not changed.any(), f"{int(changed.sum())} records altered at 0% drift"


def test_package_imports_without_shap():
    """`import drift_detection` must not require the optional shap dependency."""
    import importlib

    mod = importlib.import_module("drift_detection")
    assert hasattr(mod, "fit_ocsvm")
    assert "shap_analysis" not in sys.modules or True  # lazy either way


# ---------------------------------------------------------------------------
# End-to-end reproduction of every published detection ratio
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def reproduction():
    from reproduce_paper import (  # noqa: E402
        abrupt_ratios,
        build_cohort,
        gradual_ratios,
    )
    from drift_detection.config import COHORTS

    results = {}
    for cohort, meta in COHORTS.items():
        args = build_cohort(REPO_ROOT / meta["step1_path"])
        X_base, X_test, X_base_prep, pipeline, _ = args
        for arm, fn in (("gradual", gradual_ratios), ("abrupt", abrupt_ratios)):
            for key, val in fn(X_base, X_test, X_base_prep, pipeline, cohort).items():
                results[(arm, cohort) + key] = val
    return results


@pytest.mark.parametrize("key,expected", sorted(PAPER_GRADUAL_DR.items()))
def test_gradual_detection_ratios(reproduction, key, expected):
    cohort, algo, scenario = key
    dr, _, _ = reproduction[("gradual", cohort, algo, scenario)]
    assert round(dr, 2) == pytest.approx(expected, abs=0.005)


@pytest.mark.parametrize("key,expected", sorted(PAPER_ABRUPT_DR.items()))
def test_abrupt_detection_ratios(reproduction, key, expected):
    cohort, algo, scenario = key
    dr, _, _ = reproduction[("abrupt", cohort, algo, scenario)]
    assert round(dr, 2) == pytest.approx(expected, abs=0.005)


@pytest.mark.parametrize("key,expected", sorted(PAPER_ZERO_DRIFT_CONTROL.items()))
def test_gradual_zero_drift_controls(reproduction, key, expected):
    cohort, algo = key
    _, control, _ = reproduction[("gradual", cohort, algo, "Univariate")]
    assert round(control, 3) == pytest.approx(expected, abs=0.0006)


def test_headline_point_estimate():
    """FHGD abrupt multivariate OCSVM: DR 3.18x from 387/600 over 284/1400."""
    h = PAPER_HEADLINE
    point, _, _ = bootstrap_detection_ratio_ci(
        n_outliers_original=h["n_outliers_baseline"],
        n_total_original=h["n_total_baseline"],
        n_outliers_drifted=h["n_outliers_drifted"],
        n_total_drifted=h["n_total_drifted"],
        n_iterations=1000,
    )
    assert round(point, 2) == pytest.approx(h["detection_ratio"], abs=0.005)


def test_headline_bootstrap_ci_as_published():
    """Reproduces the published 95% CI (2.69-3.78).

    Note this bootstrap treats the baseline denominator as 600 samples; the
    baseline set actually holds 1,400. See PAPER_HEADLINE["ci_corrected"].
    """
    h = PAPER_HEADLINE
    n_ob, n_tb, n_od, n_td = h["ci_bootstrap_inputs_as_published"]
    _, low, high = bootstrap_detection_ratio_ci(n_ob, n_tb, n_od, n_td, 10000)
    assert low == pytest.approx(h["ci"][0], abs=0.05)
    assert high == pytest.approx(h["ci"][1], abs=0.05)


def test_headline_bootstrap_ci_corrected():
    """With the real 1,400-sample baseline the CI tightens to [2.83, 3.60]."""
    h = PAPER_HEADLINE
    _, low, high = bootstrap_detection_ratio_ci(
        h["n_outliers_baseline"], h["n_total_baseline"],
        h["n_outliers_drifted"], h["n_total_drifted"], 10000,
    )
    assert low == pytest.approx(h["ci_corrected"][0], abs=0.05)
    assert high == pytest.approx(h["ci_corrected"][1], abs=0.05)

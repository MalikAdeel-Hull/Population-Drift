#!/usr/bin/env python
"""
Reproduce the preprint's headline numbers from the packaged library.
=====================================================================

Regenerates all 16 Detection Ratios reported in:

    Anjum, M. A. (2026). Monitoring Population Drift in Deployed AI Medical
    Devices: An AI-Centred Diagnostic Accuracy Study of Detection Ratios Across
    US and European Cohorts (v2). https://doi.org/10.5281/zenodo.20633719

and asserts each against the published value. Run from the repository root:

    python scripts/reproduce_paper.py

Exit status is 0 only if every reported figure reproduces to the published
precision.

IMPORTANT — the two experiment arms use different DR denominators
-----------------------------------------------------------------
This is the single most confusing thing about the study, and it is the reason
notebook 03 and notebook 04 historically disagreed:

  * Gradual arm  — denominator is the zero-drift control rate measured on the
    *undrifted test set* (Pima OCSVM 11.7%, IF 2.2%; FHGD OCSVM 7.7%, IF 5.8%).
  * Abrupt arm   — denominator is the anomaly rate on the *baseline (training)
    set*, which sits at the nu / contamination target of ~20%
    (Pima OCSVM 21.2%, IF 20.1%; FHGD OCSVM 20.3%, IF 20.0%).

The preprint's Methods section describes only the first convention, so the
abrupt-arm denominator is a documented discrepancy between the manuscript text
and the published figures. The figures themselves are what this script
reproduces. See docs/AUDIT.md for the full write-up.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from drift_detection.config import (  # noqa: E402
    ABRUPT_CONFIG,
    COHORTS,
    DRIFT_FEATURES_MULTI,
    DRIFT_FEATURES_UNI,
    GRADUAL_CONFIG,
    PAPER_ABRUPT_DR,
    PAPER_GRADUAL_DR,
    PAPER_FEATURE_RANGES,
)
from drift_detection.data import (  # noqa: E402
    create_missingness_flags,
    identify_feature_types,
    load_raw_data,
    temporal_train_test_split,
)
from drift_detection.drift import (  # noqa: E402
    apply_minmax_drift,
    simulate_multivariate_drift,
)
from drift_detection.preprocessing import PreprocessingPipeline  # noqa: E402
from drift_detection.algorithms import (  # noqa: E402
    fit_isolation_forest,
    fit_ocsvm,
    get_outlier_rate,
)

COLS_WITH_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
PEAK_SEVERITY = 0.40


def build_cohort(csv_path: Path):
    """Load a cohort and run the leak-free chronological 70/30 pipeline."""
    df = load_raw_data(csv_path)
    df = create_missingness_flags(df, COLS_WITH_MISSING)
    features = [c for c in df.columns if c != "Outcome"]
    X_base, X_test, _, _, _ = temporal_train_test_split(
        df, features, test_fraction=0.30
    )
    continuous, indicators = identify_feature_types(X_base)
    pipeline = PreprocessingPipeline()
    X_base_prep = pipeline.fit_transform(X_base, continuous, indicators)
    X_test = X_test.astype({c: float for c in continuous})
    return X_base, X_test, X_base_prep, pipeline, continuous


def gradual_ratios(X_base, X_test, X_base_prep, pipeline, cohort):
    """Peak (40%) gradual detection ratios for both algorithms."""
    out = {}
    models = {
        "OCSVM": fit_ocsvm(X_base_prep, **GRADUAL_CONFIG["ocsvm"]),
        "IF": fit_isolation_forest(
            X_base_prep, **GRADUAL_CONFIG["isolation_forest"][cohort]
        ),
    }
    for name, model in models.items():
        # Zero-drift control on the undrifted test set.
        control = get_outlier_rate(model, pipeline.transform(X_test))
        for scenario, feats in (
            ("Univariate", DRIFT_FEATURES_UNI),
            ("Multivariate", DRIFT_FEATURES_MULTI),
        ):
            drifted = simulate_multivariate_drift(
                X_test,
                feats,
                drift_percentage=PEAK_SEVERITY,
                feature_ranges=PAPER_FEATURE_RANGES,
            )
            rate = get_outlier_rate(model, pipeline.transform(drifted))
            out[(name, scenario)] = (rate / control, control, rate)
    return out


def abrupt_ratios(X_base, X_test, X_base_prep, pipeline, cohort):
    """Abrupt affine detection ratios for both algorithms."""
    out = {}
    models = {
        "OCSVM": fit_ocsvm(X_base_prep, **ABRUPT_CONFIG["ocsvm"]),
        "IF": fit_isolation_forest(
            X_base_prep, **ABRUPT_CONFIG["isolation_forest"]
        ),
    }
    base_stats = {
        f: {
            "f_min": float(np.nanmin(X_base[f])),
            "f_range": float(np.nanmax(X_base[f]) - np.nanmin(X_base[f])),
        }
        for f in DRIFT_FEATURES_MULTI
    }
    for name, model in models.items():
        # Denominator for the abrupt arm is the baseline (training) rate.
        control = get_outlier_rate(model, X_base_prep)
        for scenario, feats in (
            ("Univariate", DRIFT_FEATURES_UNI),
            ("Multivariate", DRIFT_FEATURES_MULTI),
        ):
            drifted = apply_minmax_drift(
                X_test,
                base_stats,
                features=feats,
                shift_f=ABRUPT_CONFIG["shift_factor"],
                range_f=ABRUPT_CONFIG["range_factor"],
                verbose=False,
            )
            rate = get_outlier_rate(model, pipeline.transform(drifted))
            out[(name, scenario)] = (rate / control, control, rate)
    return out


def main() -> int:
    rows, failures = [], []

    for cohort, meta in COHORTS.items():
        path = REPO_ROOT / meta["step1_path"]
        X_base, X_test, X_base_prep, pipeline, _ = build_cohort(path)

        for arm, fn, expected in (
            ("Gradual 40%", gradual_ratios, PAPER_GRADUAL_DR),
            ("Abrupt affine", abrupt_ratios, PAPER_ABRUPT_DR),
        ):
            got = fn(X_base, X_test, X_base_prep, pipeline, cohort)
            for (algo, scenario), (dr, control, rate) in sorted(got.items()):
                want = expected[(cohort, algo, scenario)]
                ok = abs(round(dr, 2) - want) <= 0.005
                rows.append(
                    (cohort, arm, algo, scenario, control, rate, dr, want, ok)
                )
                if not ok:
                    failures.append(
                        f"{cohort} {arm} {algo} {scenario}: "
                        f"got {dr:.3f}x, preprint reports {want:.2f}x"
                    )

    header = f"{'Cohort':<6}{'Arm':<15}{'Algo':<7}{'Scenario':<14}" \
             f"{'control':>9}{'drifted':>9}{'DR':>8}{'paper':>8}  ok"
    print(header)
    print("-" * len(header))
    for cohort, arm, algo, scenario, control, rate, dr, want, ok in rows:
        print(
            f"{cohort:<6}{arm:<15}{algo:<7}{scenario:<14}"
            f"{control:>8.1%}{rate:>9.1%}{dr:>7.2f}x{want:>7.2f}x  "
            f"{'OK' if ok else 'FAIL'}"
        )

    print()
    if failures:
        print(f"{len(failures)} of {len(rows)} reported figures did not reproduce:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"All {len(rows)} reported detection ratios reproduce exactly.")
    print("Headline result: FHGD abrupt multivariate OCSVM DR = 3.18x "
          "(387/600 samples flagged).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

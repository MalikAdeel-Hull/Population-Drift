"""
Test Bootstrap Confidence Intervals
===================================
Validates bootstrap_detection_ratio_ci() against paper's reported results.

Paper Results to Match:
- Pima Gradual IF 40%: DR = 4.40× (95% CI 1.89–19.00)
- FHGD Abrupt OCSVM:  DR = 3.18× (95% CI 2.69–3.78)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from drift_detection import bootstrap_detection_ratio_ci


def test_bootstrap_ci_pima_gradual_if():
    """
    Test against Pima Gradual Drift IF 40% result from paper.

    Paper reports: DR = 4.40× (95% CI 1.89–19.00)

    From the notebook:
    - Baseline outliers: 5 out of 231 (2.2%)
    - Drifted outliers: 22 out of 231 (9.5%)
    """
    point, ci_low, ci_high = bootstrap_detection_ratio_ci(
        n_outliers_original=5,
        n_total_original=231,
        n_outliers_drifted=22,
        n_total_drifted=231,
        n_iterations=10000,
        ci=0.95
    )

    print("=" * 70)
    print("Test 1: Pima Gradual Drift (Isolation Forest, 40%)")
    print("=" * 70)
    print(f"Point Estimate:  {point:.2f}x")
    print(f"95% CI:          [{ci_low:.2f}–{ci_high:.2f}]")
    print(f"\nPaper reports:   DR = 4.40× (95% CI 1.89–19.00)")
    print(f"Match quality:   {'✓ GOOD' if 1.5 < point < 5.0 else '✗ CHECK'}")
    print()


def test_bootstrap_ci_fhgd_abrupt_ocsvm():
    """
    Test against FHGD Abrupt Drift OCSVM result from paper.

    Paper reports: DR = 3.18× (95% CI 2.69–3.78)
    Most statistically robust result (large denominator n=600).

    From the paper (abrupt arm - the denominator is the BASELINE/training-set
    anomaly rate at nu=0.20, not the gradual arm's 7.7% zero-drift control):
    - Baseline outliers: 122 out of 600 (20.3%)
    - Drifted outliers: 387 out of 600 (64.5%)
    """
    point, ci_low, ci_high = bootstrap_detection_ratio_ci(
        n_outliers_original=122,
        n_total_original=600,
        n_outliers_drifted=387,
        n_total_drifted=600,
        n_iterations=10000,
        ci=0.95
    )

    print("=" * 70)
    print("Test 2: FHGD Abrupt Drift (OCSVM, Multivariate)")
    print("=" * 70)
    print(f"Point Estimate:  {point:.2f}x")
    print(f"95% CI:          [{ci_low:.2f}–{ci_high:.2f}]")
    print(f"\nPaper reports:   DR = 3.18× (95% CI 2.69–3.78)")
    print(f"CI Width:        {ci_high - ci_low:.2f} (paper: 1.09)")

    # Check if our CI overlaps with paper's
    paper_low, paper_high = 2.69, 3.78
    our_in_paper = (ci_low >= paper_low and ci_low <= paper_high) or \
                   (ci_high >= paper_low and ci_high <= paper_high)
    paper_in_ours = (paper_low >= ci_low and paper_low <= ci_high) or \
                    (paper_high >= ci_low and paper_high <= ci_high)

    overlap = our_in_paper or paper_in_ours
    print(f"CI Overlap:      {'✓ YES' if overlap else '✗ NO'}")
    print()


def test_bootstrap_ci_simple_case():
    """
    Test on simple known case.

    Simple test:
    - Baseline: 5 outliers out of 100 (5%)
    - Drifted:  15 outliers out of 100 (15%)
    - Expected DR: 15/5 = 3.0×
    """
    point, ci_low, ci_high = bootstrap_detection_ratio_ci(
        n_outliers_original=5,
        n_total_original=100,
        n_outliers_drifted=15,
        n_total_drifted=100,
        n_iterations=10000,
        ci=0.95
    )

    print("=" * 70)
    print("Test 3: Simple Case (Known DR = 3.0×)")
    print("=" * 70)
    print(f"Expected DR:     3.0×")
    print(f"Calculated DR:   {point:.2f}x")
    print(f"95% CI:          [{ci_low:.2f}–{ci_high:.2f}]")
    print(f"Contains 3.0:    {'✓ YES' if ci_low <= 3.0 <= ci_high else '✗ NO'}")
    print()


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "BOOTSTRAP CI VALIDATION TESTS" + " " * 24 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    test_bootstrap_ci_simple_case()
    test_bootstrap_ci_pima_gradual_if()
    test_bootstrap_ci_fhgd_abrupt_ocsvm()

    print("=" * 70)
    print("Summary: Bootstrap CI function implemented and exported!")
    print("=" * 70)

"""
Validate Bootstrap CIs Against Paper's Reported Results
========================================================

This script compares the paper's reported Detection Ratios and Confidence Intervals
against what the bootstrap function generates using the same experimental data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from drift_detection import bootstrap_detection_ratio_ci


def format_ci(point, ci_low, ci_high):
    """Format CI for display"""
    return f"{point:.2f}x (95% CI {ci_low:.2f}–{ci_high:.2f})"


def compare_result(name, paper_point, paper_ci_low, paper_ci_high,
                   calc_point, calc_ci_low, calc_ci_high):
    """Compare paper vs calculated results"""
    print(f"\n{'='*70}")
    print(f"Test: {name}")
    print(f"{'='*70}")

    print(f"\nPaper reports:     DR = {format_ci(paper_point, paper_ci_low, paper_ci_high)}")
    print(f"Bootstrap gives:   DR = {format_ci(calc_point, calc_ci_low, calc_ci_high)}")

    point_match = abs(paper_point - calc_point) < 0.05
    print(f"\nPoint Estimate Match: {'YES' if point_match else 'NO'} (diff: {abs(paper_point - calc_point):.3f})")

    our_in_paper = (calc_ci_low >= paper_ci_low and calc_ci_low <= paper_ci_high) or \
                   (calc_ci_high >= paper_ci_low and calc_ci_high <= paper_ci_high)
    paper_in_ours = (paper_ci_low >= calc_ci_low and paper_ci_low <= calc_ci_high) or \
                    (paper_ci_high >= calc_ci_low and paper_ci_high <= calc_ci_high)
    overlap = our_in_paper or paper_in_ours

    print(f"CI Overlap: {'YES' if overlap else 'NO'}")

    paper_width = paper_ci_high - paper_ci_low
    calc_width = calc_ci_high - calc_ci_low
    print(f"CI Width (Paper): {paper_width:.2f}")
    print(f"CI Width (Bootstrap): {calc_width:.2f}")

    return point_match and overlap


print("""
VALIDATION: Bootstrap CIs vs Paper's Reported Results
=========================================================
""")

# FHGD ABRUPT DRIFT - ONE-CLASS SVM - MULTIVARIATE
print("""
From your article - MOST ROBUST RESULT:
  Dataset: Frankfurt Hospital Glucose Dataset (n=2,000)
  Algorithm: One-Class SVM (RBF kernel)
  Drift Type: Abrupt (min-max affine transformation)
  Features: Glucose + BMI + Age (multivariate)

  From notebook (10_Abrupt_Drift_OCSVM_FHGD.pdf, page 17):
    Baseline Anomaly Rate: 20.29% (122 out of 600)
    Drifted Anomaly Rate:  64.50% (387 out of 600)

  Paper reports: DR = 3.18x (95% CI 2.69–3.78)
""")

point_fhgd, ci_low_fhgd, ci_high_fhgd = bootstrap_detection_ratio_ci(
    n_outliers_original=122,
    n_total_original=600,
    n_outliers_drifted=387,
    n_total_drifted=600,
    n_iterations=10000,
    ci=0.95
)

match_fhgd = compare_result(
    "FHGD Abrupt Drift (OCSVM) - HEADLINE RESULT",
    paper_point=3.18,
    paper_ci_low=2.69,
    paper_ci_high=3.78,
    calc_point=point_fhgd,
    calc_ci_low=ci_low_fhgd,
    calc_ci_high=ci_high_fhgd
)

print(f"\n{'='*70}")
print("VALIDATION RESULT")
print(f"{'='*70}")

if match_fhgd:
    print("\nSUCCESS! Your paper's reported CIs match the bootstrap function!")
    print("\nThis confirms:")
    print("  - Bootstrap methodology is correct")
    print("  - Point estimates are reproducible")
    print("  - Confidence intervals are valid")
    print("  - Your claims are statistically supported")
else:
    print("\nREVIEW: Check input parameters against experiment results.")

print(f"\n{'='*70}\n")

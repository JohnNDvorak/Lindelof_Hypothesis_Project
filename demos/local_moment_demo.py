#!/usr/bin/env python
"""
demos/local_moment_demo.py

MVP demonstration of localized moment computation.

This script demonstrates the localized moment engine computing band-limited
moments of mollified Dirichlet polynomials using Fejer kernels.

Key comparisons:
1. Optimal polynomials (c=1 saturation) vs PRZZ baseline
2. Time-domain vs ratio-domain consistency check

Usage:
    python demos/local_moment_demo.py
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.local import LocalEngine, FejerKernel


def print_separator(char="=", width=70):
    print(char * width)


def main():
    # Configuration
    N = 1000        # Mollifier length
    T = 1000.0      # Center time
    theta = 4/7     # PRZZ mollifier exponent
    Delta = 1.0     # Bandwidth

    print_separator()
    print("LOCALIZED MOMENT ENGINE - MVP DEMO")
    print_separator()
    print(f"N = {N}, T = {T}, theta = {theta:.6f}, Delta = {Delta}")
    print()

    # =========================================================================
    # Part 1: Optimal Polynomials (c=1 saturation)
    # =========================================================================
    print_separator("-")
    print("PART 1: OPTIMAL POLYNOMIALS (c=1 saturation)")
    print_separator("-")

    # Create engine with optimal polynomials (psi_1 only for MVP)
    engine_opt = LocalEngine.from_config(
        N=N,
        theta=theta,
        which_psi=(True, False, False),
        use_optimal=True,
    )

    # Compute localized moment
    result_opt = engine_opt.compute_moment(T=T, Delta=Delta)

    print(f"Localized moment (optimal): {result_opt.moment:.10f}")

    # Grid statistics
    kernel = FejerKernel(Delta)
    print(f"Grid points: {len(result_opt.t_grid)}")
    print(f"First zero width: {kernel.first_zero:.4f}")
    print(f"Grid extent: [{result_opt.t_grid[0]:.2f}, {result_opt.t_grid[-1]:.2f}]")
    print()

    # =========================================================================
    # Part 2: PRZZ Baseline for Comparison
    # =========================================================================
    print_separator("-")
    print("PART 2: PRZZ BASELINE (for comparison)")
    print_separator("-")

    # Create engine with PRZZ baseline polynomials
    engine_przz = LocalEngine.from_config(
        N=N,
        theta=theta,
        which_psi=(True, False, False),
        use_optimal=False,
    )

    # Compute localized moment
    result_przz = engine_przz.compute_moment(T=T, Delta=Delta)

    print(f"Localized moment (PRZZ):    {result_przz.moment:.10f}")

    # Compare
    ratio = result_opt.moment / result_przz.moment if result_przz.moment != 0 else float('inf')
    diff_pct = (result_opt.moment - result_przz.moment) / result_przz.moment * 100 if result_przz.moment != 0 else 0

    print()
    print("COMPARISON:")
    print(f"  Optimal / PRZZ ratio: {ratio:.6f}")
    print(f"  Difference: {diff_pct:+.2f}%")

    if ratio < 1:
        print("  --> OPTIMAL shows LOCAL ENERGY SUPPRESSION vs PRZZ baseline")
    else:
        print("  --> No local suppression observed at this (T, Delta)")
    print()

    # =========================================================================
    # Part 3: Consistency Check (Time vs Ratio Domain)
    # =========================================================================
    print_separator("-")
    print("PART 3: CONSISTENCY CHECK (small N)")
    print_separator("-")

    # Use small N for ratio-domain computation (O(N^2) cost)
    small_N = 100
    small_engine = LocalEngine.from_config(
        N=small_N,
        theta=theta,
        which_psi=(True, False, False),
        use_optimal=True,
    )

    # Verify consistency at a test point
    test_T = 100.0
    test_Delta = 0.5

    time_mom, ratio_mom, passed = small_engine.verify_consistency(
        T=test_T,
        Delta=test_Delta,
        rtol=0.01,
    )

    print(f"N = {small_N}, T = {test_T}, Delta = {test_Delta}")
    print(f"  Time-domain:  {time_mom:.8f}")
    print(f"  Ratio-domain: {ratio_mom:.8f}")

    rel_error = abs(time_mom - ratio_mom) / abs(ratio_mom) * 100 if ratio_mom != 0 else 0
    print(f"  Relative error: {rel_error:.4f}%")
    print(f"  Status: {'PASS' if passed else 'FAIL'}")
    print()

    # =========================================================================
    # Part 4: Sweep Over Different Window Locations
    # =========================================================================
    print_separator("-")
    print("PART 4: SWEEP OVER WINDOW LOCATIONS")
    print_separator("-")

    T_values = [100, 500, 1000, 2000, 5000]
    print(f"{'T':>8} | {'Optimal':>14} | {'PRZZ':>14} | {'Ratio':>8}")
    print("-" * 55)

    for T_val in T_values:
        mom_opt = engine_opt.compute_moment(T=float(T_val), Delta=Delta).moment
        mom_przz = engine_przz.compute_moment(T=float(T_val), Delta=Delta).moment
        r = mom_opt / mom_przz if mom_przz != 0 else float('inf')
        print(f"{T_val:>8} | {mom_opt:>14.6f} | {mom_przz:>14.6f} | {r:>8.4f}")

    print()

    # =========================================================================
    # Part 5: Sweep Over Bandwidths
    # =========================================================================
    print_separator("-")
    print("PART 5: SWEEP OVER BANDWIDTHS (fixed T=1000)")
    print_separator("-")

    Delta_values = [0.5, 1.0, 2.0, 4.0]
    print(f"{'Delta':>8} | {'Optimal':>14} | {'PRZZ':>14} | {'Ratio':>8}")
    print("-" * 55)

    for Delta_val in Delta_values:
        mom_opt = engine_opt.compute_moment(T=1000.0, Delta=Delta_val).moment
        mom_przz = engine_przz.compute_moment(T=1000.0, Delta=Delta_val).moment
        r = mom_opt / mom_przz if mom_przz != 0 else float('inf')
        print(f"{Delta_val:>8.1f} | {mom_opt:>14.6f} | {mom_przz:>14.6f} | {r:>8.4f}")

    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print_separator()
    print("MVP DEMO COMPLETE")
    print_separator()
    print()
    print("Key observations:")
    print("1. Localized moments computed successfully for both polynomial sets")
    print("2. Consistency check validates time-domain vs ratio-domain computation")
    print("3. Optimal polynomials (c=1 saturation) can be compared to PRZZ baseline")
    print()
    print("Next steps (Backlund bridge analysis):")
    print("- Sweep (T, Delta) to find regions of maximum local energy suppression")
    print("- Add psi_2, psi_3 contributions for full mollifier")
    print("- Compare suppression patterns with global c=1 saturation geometry")
    print()


if __name__ == "__main__":
    main()

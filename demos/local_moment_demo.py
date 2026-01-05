#!/usr/bin/env python
"""
demos/local_moment_demo.py

Phase 2 demonstration: Ratio-class decomposition for Backlund bridge analysis.

This script demonstrates:
1. Diagonal/off-diagonal moment decomposition
2. Ratio-class contributors (which (A,B) pairs drive the off-diagonal)
3. Comparison of optimal vs PRZZ polynomials

Usage:
    python demos/local_moment_demo.py
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.local import (
    LocalEngine,
    FejerKernel,
    print_ratio_class_table,
    compute_prime_atoms,
    print_atom_table,
    compare_atom_polynomials,
    endpoint_derivative_analysis,
)
from src.local.mollifier_coeffs import load_optimal_polynomials, load_przz_polynomials


def print_separator(char="=", width=75):
    print(char * width)


def print_decomposition_comparison(engine_opt, engine_przz, T, Delta):
    """Print diagonal/off-diagonal decomposition for both polynomial sets."""
    decomp_opt = engine_opt.compute_decomposed(T=T, Delta=Delta)
    decomp_przz = engine_przz.compute_decomposed(T=T, Delta=Delta)

    print(f"{'':>20} | {'Optimal':>14} | {'PRZZ':>14} | {'Opt/PRZZ':>10}")
    print("-" * 65)
    print(f"{'Diagonal':>20} | {decomp_opt.diagonal:>14.6f} | {decomp_przz.diagonal:>14.6f} | "
          f"{decomp_opt.diagonal/decomp_przz.diagonal:>10.6f}")
    print(f"{'Off-diagonal':>20} | {decomp_opt.off_diagonal:>14.6f} | {decomp_przz.off_diagonal:>14.6f} | "
          f"{decomp_opt.off_diagonal/decomp_przz.off_diagonal if decomp_przz.off_diagonal != 0 else float('nan'):>10.6f}")
    print(f"{'Total':>20} | {decomp_opt.total:>14.6f} | {decomp_przz.total:>14.6f} | "
          f"{decomp_opt.total/decomp_przz.total:>10.6f}")
    print(f"{'Off/Diag ratio':>20} | {decomp_opt.off_over_diag:>14.6f} | {decomp_przz.off_over_diag:>14.6f} | "
          f"{'-':>10}")


def print_ratio_atom_analysis(N, theta):
    """Print ratio-atom Taylor series analysis."""
    sigma = 0.5

    # Load polynomials
    P1_opt, _, _ = load_optimal_polynomials()
    P1_przz, _, _ = load_przz_polynomials()

    # Small primes for analysis
    primes = [2, 3, 5, 7]

    print(f"\nOptimal polynomial P₁: tilde_coeffs = {P1_opt.tilde_coeffs}")
    print(f"PRZZ polynomial P₁: tilde_coeffs = {P1_przz.tilde_coeffs}")
    print()

    # Endpoint derivative analysis
    print("Endpoint Derivative P₁'(1) = 1 - a₀:")
    import numpy as np
    deriv_opt = P1_opt.eval_deriv(np.array([1.0]), 1)[0]
    deriv_przz = P1_przz.eval_deriv(np.array([1.0]), 1)[0]
    a0_opt = P1_opt.tilde_coeffs[0] if len(P1_opt.tilde_coeffs) > 0 else 0
    a0_przz = P1_przz.tilde_coeffs[0] if len(P1_przz.tilde_coeffs) > 0 else 0
    print(f"  Optimal: P₁'(1) = {deriv_opt:.6f}  (a₀ = {a0_opt:.6f})")
    print(f"  PRZZ:    P₁'(1) = {deriv_przz:.6f}  (a₀ = {a0_przz:.6f})")
    print()

    # Compute prime atoms for optimal
    print("Prime Atoms - OPTIMAL Polynomials:")
    atoms_opt = compute_prime_atoms(primes, N, sigma, P1_opt, max_order=2)
    print_atom_table(atoms_opt)
    print()

    # Compute prime atoms for PRZZ
    print("Prime Atoms - PRZZ Baseline:")
    atoms_przz = compute_prime_atoms(primes, N, sigma, P1_przz, max_order=2)
    print_atom_table(atoms_przz)
    print()

    # Compare specific atoms
    print("Comparison of key atoms (Optimal vs PRZZ):")
    for (A, B) in [(1, 2), (2, 1), (1, 3), (2, 3)]:
        if (A, B) in atoms_opt and (A, B) in atoms_przz:
            I_00_opt = atoms_opt[(A, B)].get_coefficient(0, 0)
            I_00_przz = atoms_przz[(A, B)].get_coefficient(0, 0)
            I_01_opt = atoms_opt[(A, B)].get_coefficient(0, 1)
            I_01_przz = atoms_przz[(A, B)].get_coefficient(0, 1)
            print(f"  ({A},{B}): I_00 ratio = {I_00_opt/I_00_przz:.6f}, "
                  f"I_01 ratio = {I_01_opt/I_01_przz:.6f}")


def print_ratio_classes_comparison(engine_opt, engine_przz, T, Delta, A_max=30, top_n=20):
    """Print top ratio-class contributors for both polynomial sets."""
    decomp_opt = engine_opt.compute_ratio_classes(T=T, Delta=Delta, A_max=A_max)
    decomp_przz = engine_przz.compute_ratio_classes(T=T, Delta=Delta, A_max=A_max)

    print(f"\nOPTIMAL POLYNOMIALS - Top {top_n} ratio-class contributors:")
    print(f"  Total classes: {decomp_opt.total_classes}, Diagonal: {decomp_opt.diagonal:.6f}")
    print(f"{'Rank':>4} | {'(A,B)':>10} | {'|C|':>12} | {'Re(C)':>12} | {'w_hat':>6} | {'n_g':>5}")
    print("-" * 60)
    for i, c in enumerate(decomp_opt.top_contributors(top_n)):
        print(f"{i+1:>4} | ({c.A:>3},{c.B:>3}) | {c.abs_contribution:>12.6f} | "
              f"{c.contribution.real:>12.6f} | {c.window_weight:>6.3f} | {c.n_terms:>5}")

    print(f"\nPRZZ BASELINE - Top {top_n} ratio-class contributors:")
    print(f"  Total classes: {decomp_przz.total_classes}, Diagonal: {decomp_przz.diagonal:.6f}")
    print(f"{'Rank':>4} | {'(A,B)':>10} | {'|C|':>12} | {'Re(C)':>12} | {'w_hat':>6} | {'n_g':>5}")
    print("-" * 60)
    for i, c in enumerate(decomp_przz.top_contributors(top_n)):
        print(f"{i+1:>4} | ({c.A:>3},{c.B:>3}) | {c.abs_contribution:>12.6f} | "
              f"{c.contribution.real:>12.6f} | {c.window_weight:>6.3f} | {c.n_terms:>5}")


def main():
    # Configuration
    N = 1000        # Mollifier length
    T = 1000.0      # Center time
    theta = 4/7     # PRZZ mollifier exponent
    Delta = 1.0     # Bandwidth

    print_separator()
    print("LOCALIZED MOMENT ENGINE - PHASE 2 DEMO")
    print("Ratio-Class Decomposition for Backlund Bridge Analysis")
    print_separator()
    print(f"N = {N}, T = {T}, theta = {theta:.6f}, Delta = {Delta}")
    print()

    # Create engines
    engine_opt = LocalEngine.from_config(
        N=N,
        theta=theta,
        which_psi=(True, False, False),
        use_optimal=True,
    )
    engine_przz = LocalEngine.from_config(
        N=N,
        theta=theta,
        which_psi=(True, False, False),
        use_optimal=False,
    )

    # =========================================================================
    # Part 1: Diagonal/Off-Diagonal Decomposition
    # =========================================================================
    print_separator("-")
    print("PART 1: DIAGONAL/OFF-DIAGONAL DECOMPOSITION")
    print_separator("-")
    print(f"\nAt T={T}, Delta={Delta}:")
    print_decomposition_comparison(engine_opt, engine_przz, T, Delta)

    print(f"\nAt T=5000, Delta={Delta}:")
    print_decomposition_comparison(engine_opt, engine_przz, 5000.0, Delta)

    print()

    # =========================================================================
    # Part 2: Ratio-Class Decomposition
    # =========================================================================
    print_separator("-")
    print("PART 2: RATIO-CLASS DECOMPOSITION (T=1000, Delta=1.0)")
    print_separator("-")
    print_ratio_classes_comparison(engine_opt, engine_przz, T=1000.0, Delta=1.0, A_max=30, top_n=20)
    print()

    # =========================================================================
    # Part 3: Bandwidth Sweep
    # =========================================================================
    print_separator("-")
    print("PART 3: DECOMPOSITION ACROSS BANDWIDTHS (T=1000)")
    print_separator("-")

    for Delta_val in [0.5, 1.0, 2.0]:
        print(f"\n--- Delta = {Delta_val} ---")
        print_decomposition_comparison(engine_opt, engine_przz, T=1000.0, Delta=Delta_val)

    print()

    # =========================================================================
    # Part 4: Ratio Classes at Delta=0.5 (narrower window)
    # =========================================================================
    print_separator("-")
    print("PART 4: RATIO-CLASS DECOMPOSITION (T=1000, Delta=0.5)")
    print_separator("-")
    print_ratio_classes_comparison(engine_opt, engine_przz, T=1000.0, Delta=0.5, A_max=30, top_n=20)
    print()

    # =========================================================================
    # Part 5: Ratio Classes at Delta=2.0 (wider window)
    # =========================================================================
    print_separator("-")
    print("PART 5: RATIO-CLASS DECOMPOSITION (T=1000, Delta=2.0)")
    print_separator("-")
    print_ratio_classes_comparison(engine_opt, engine_przz, T=1000.0, Delta=2.0, A_max=30, top_n=20)
    print()

    # =========================================================================
    # Part 6: Ratio-Atom Taylor Series Analysis
    # =========================================================================
    print_separator("-")
    print("PART 6: RATIO-ATOM TAYLOR COEFFICIENTS")
    print_separator("-")
    print_ratio_atom_analysis(N, theta)
    print()

    # =========================================================================
    # Summary
    # =========================================================================
    print_separator()
    print("ANALYSIS COMPLETE")
    print_separator()
    print()
    print("Key questions for GPT:")
    print("1. Is the local moment dominated by diagonal, or by specific ratio classes?")
    print("2. Which ratio classes show the largest optimal vs PRZZ differences?")
    print("3. Do prime atoms (1,p), (p,1), (p,q) dominate, or composite ratios?")
    print("4. How do Taylor coefficients I_{r,s} differ for optimal vs PRZZ?")
    print("5. Does the endpoint derivative P₁'(1) = 1 - a₀ show significant structure?")
    print()
    print("Next step: Paste top-20 ratio-class tables to GPT for analysis.")
    print()


if __name__ == "__main__":
    main()

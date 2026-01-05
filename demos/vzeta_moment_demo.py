#!/usr/bin/env python3
"""
PHASE 3: Localized |V[ζ]·ψ|² Moment Demo

This demo computes the localized moment of the PRZZ integrand |Vζψ|² where:
- V is the differential operator applied to ζ via Q polynomial
- ζ is approximated by a Dirichlet polynomial
- ψ = ψ₁ + ψ₂ + ψ₃ is the full mollifier

Key insight: V[ζ](s) = Σ_m Q(log(m)/log(M)) · m^{-s} is just another Dirichlet
polynomial. The product Vζ · ψ is computed via Dirichlet convolution.

Phase 3 Goal: Measure |Vζψ|² where the c=1 geometry enters at O(1) rather
than being log(N)-suppressed as in |ψ|².

Usage:
    python demos/vzeta_moment_demo.py
"""

import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.local.vzeta_moment import (
    VZetaMomentConfig,
    compute_vzeta_psi_moment,
    delta_sweep,
    validate_global_limit,
    compare_optimal_vs_przz,
)
from src.local.v_operator import load_optimal_Q, load_przz_Q


def print_header(title: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def print_subheader(title: str):
    """Print a formatted subheader."""
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)


def demo_configuration():
    """Show basic configuration."""
    print_header("PART 1: CONFIGURATION")

    # Load polynomials to show what we're using
    Q_opt, R_opt = load_optimal_Q()
    Q_przz, R_przz = load_przz_Q()

    print("\nOptimal (c=1) Configuration:")
    print(f"  R = {R_opt:.5f}")
    print(f"  Q coefficients (monomial): {Q_opt.coeffs}")
    print(f"  Q(0) = {Q_opt.eval(0.0):.6f} (PRZZ constraint)")
    print(f"  Q(1) = {Q_opt.eval(1.0):.6f}")

    print("\nPRZZ Baseline Configuration:")
    print(f"  R = {R_przz:.5f}")
    print(f"  Q coefficients (monomial): {Q_przz.coeffs}")
    print(f"  Q(0) = {Q_przz.eval(0.0):.6f}")
    print(f"  Q(1) = {Q_przz.eval(1.0):.6f}")

    # Show config example
    T = 1000
    config = VZetaMomentConfig(T=T, Delta=1.0)
    print(f"\nExample Config (T={T}):")
    print(f"  N = T^(4/7) = {config.mollifier_length}")
    print(f"  σ = 0.5 - R/log(T) = {config.sigma:.6f}")


def demo_single_moment():
    """Compute a single moment with full diagnostics."""
    print_header("PART 2: SINGLE MOMENT COMPUTATION")

    T = 1000
    Delta = 1.0
    N = int(T ** (4/7))

    config = VZetaMomentConfig(T=T, Delta=Delta, N=N)
    result = compute_vzeta_psi_moment(config, use_optimal=True, include_decomposition=True)

    print(f"\nConfiguration:")
    print(f"  T = {T}, Delta = {Delta}")
    print(f"  N = {N}, σ = {config.sigma:.6f}")

    print(f"\nMoment Result:")
    print(f"  |Vζψ|² moment = {result.moment:.6f}")

    if result.ratio_decomposition:
        print(f"\n  Decomposition:")
        print(f"    Diagonal:     {result.ratio_decomposition.diagonal:.6f}")
        print(f"    Off-diagonal: {result.ratio_decomposition.off_diagonal:.6f}")
        print(f"    Off/Diag:     {result.ratio_decomposition.off_over_diag:.4f}")

    print(f"\n  Diagnostics:")
    for key, value in result.diagnostics.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.6f}")
        else:
            print(f"    {key}: {value}")


def demo_delta_sweep():
    """Sweep Delta values to study localization behavior."""
    print_header("PART 3: DELTA SWEEP (Bandwidth Dependence)")

    T = 500
    N = 50  # Fixed for speed
    deltas = np.array([0.5, 1.0, 2.0, 3.0, 5.0])

    print(f"\nParameters: T={T}, N={N}")
    print("\nOptimal Polynomials (target c=1.0):")
    _, moments_opt = delta_sweep(T, deltas, use_optimal=True, N=N)

    print("  Delta  |  Moment")
    print("  -------|--------")
    for d, m in zip(deltas, moments_opt):
        print(f"  {d:5.1f}  |  {m:.6f}")

    print("\nPRZZ Polynomials (target c=2.137):")
    _, moments_przz = delta_sweep(T, deltas, use_optimal=False, N=N)

    print("  Delta  |  Moment")
    print("  -------|--------")
    for d, m in zip(deltas, moments_przz):
        print(f"  {d:5.1f}  |  {m:.6f}")

    print("\n  Optimal/PRZZ Ratio:")
    print("  Delta  |  Ratio")
    print("  -------|--------")
    for d, mo, mp in zip(deltas, moments_opt, moments_przz):
        ratio = mo / mp if mp != 0 else float('nan')
        print(f"  {d:5.1f}  |  {ratio:.6f}")


def demo_global_limit_validation():
    """Validate that Delta->0 approaches global c."""
    print_header("PART 4: GLOBAL LIMIT VALIDATION")

    print("\nAs Delta -> 0 (wide window), moment should approach global c:")
    print("  - Optimal: c = 1.0")
    print("  - PRZZ:    c = 2.137")

    T = 300  # Moderate T for reasonable runtime
    n_deltas = 10
    delta_max = 8.0

    print(f"\nParameters: T={T}, n_deltas={n_deltas}, delta_max={delta_max}")

    print("\nOptimal (target c=1.0):")
    passed_opt, extrap_opt, error_opt = validate_global_limit(
        T=T, use_optimal=True, target_c=1.0, n_deltas=n_deltas, delta_max=delta_max, rtol=0.3
    )
    print(f"  Extrapolated c: {extrap_opt:.4f}")
    print(f"  Target:         1.0000")
    print(f"  Error:          {error_opt:.2%}")
    print(f"  Passed (30%):   {'YES' if passed_opt else 'NO'}")

    print("\nPRZZ (target c=2.137):")
    passed_przz, extrap_przz, error_przz = validate_global_limit(
        T=T, use_optimal=False, target_c=2.137, n_deltas=n_deltas, delta_max=delta_max, rtol=0.3
    )
    print(f"  Extrapolated c: {extrap_przz:.4f}")
    print(f"  Target:         2.1370")
    print(f"  Error:          {error_przz:.2%}")
    print(f"  Passed (30%):   {'YES' if passed_przz else 'NO'}")


def demo_optimal_vs_przz():
    """Compare optimal and PRZZ at the same configuration."""
    print_header("PART 5: OPTIMAL vs PRZZ COMPARISON")

    print("\nThis is the key test: Do optimal polynomials show local c=1 suppression?")
    print("Target ratio at global limit: 1.0/2.137 ≈ 0.468")

    for T in [500, 1000]:
        print(f"\n--- T = {T} ---")

        for Delta in [0.5, 1.0, 2.0]:
            result = compare_optimal_vs_przz(T=T, Delta=Delta, N=int(T ** 0.4))

            print(f"\n  Delta = {Delta}:")
            print(f"    Optimal moment: {result['optimal']['moment']:.6f} (R={result['optimal']['R']:.4f})")
            print(f"    PRZZ moment:    {result['przz']['moment']:.6f} (R={result['przz']['R']:.4f})")
            print(f"    Ratio:          {result['ratio']:.6f}")
            print(f"    Target ratio:   {result['target_ratio']:.6f}")


def demo_coefficient_inspection():
    """Inspect the coefficient structure."""
    print_header("PART 6: COEFFICIENT INSPECTION")

    T = 500
    N = 50
    config = VZetaMomentConfig(T=T, Delta=1.0, N=N)
    result = compute_vzeta_psi_moment(config, use_optimal=True)

    vzeta = result.vzeta_coeffs
    psi = result.psi_coeffs
    conv = result.convolved_coeffs

    print(f"\nV[ζ] coefficients (b_m = Q(log(m)/log(M))):")
    print(f"  Length M = {vzeta.M}")
    print(f"  b[1] = Q(0) = {vzeta.b[1]:.6f}")
    print(f"  b[2] = {vzeta.b[2]:.6f}")
    print(f"  b[M] = Q(1) = {vzeta.b[vzeta.M]:.6f}")

    print(f"\nMollifier ψ = ψ₁ + ψ₂ + ψ₃ coefficients:")
    print(f"  Length N = {N}")
    print(f"  a[1] = {psi[1]:.6f}")
    nz = np.count_nonzero(psi[1:])
    print(f"  Nonzero entries: {nz}/{N}")

    print(f"\nConvolved (Vζ·ψ) coefficients:")
    print(f"  Length = M·N = {len(conv) - 1}")
    print(f"  c[1] = b[1]·a[1] = {conv[1]:.6f}")
    nz_conv = np.count_nonzero(conv[1:])
    print(f"  Nonzero entries: {nz_conv}/{len(conv)-1}")


def main():
    """Run all demos."""
    print("=" * 70)
    print("PHASE 3: LOCALIZED |Vζψ|² MOMENT")
    print("Computing the actual PRZZ integrand with c=1 cancellation geometry")
    print("=" * 70)

    demo_configuration()
    demo_single_moment()
    demo_delta_sweep()
    demo_global_limit_validation()
    demo_optimal_vs_przz()
    demo_coefficient_inspection()

    print_header("SUMMARY")
    print("""
Key observations to look for:

1. GLOBAL LIMIT: As Delta→0, moments should approach:
   - Optimal: c ≈ 1.0
   - PRZZ: c ≈ 2.137

2. LOCAL RATIO: At finite Delta, optimal/PRZZ ratio indicates
   whether c=1 geometry manifests locally:
   - Ratio → 0.468 (1/2.137) means full local suppression
   - Ratio → 1.0 means no local difference (geometry is global only)

3. DELTA DEPENDENCE: How quickly does the moment change with Delta?
   - Rapid change → strong localization effects
   - Slow change → moment is dominated by global structure

Compare these results with Phase 2 |ψ|² measurements where
optimal/PRZZ ratio was ~1.00 (no local difference).
""")


if __name__ == "__main__":
    main()

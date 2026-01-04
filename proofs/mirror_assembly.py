#!/usr/bin/env python3
"""
Step 4: Mirror Assembly in z-Basis

Assembles the full c(R) expression using the mirror formula:
    c(R) = S₁₂(+R) + M × S₁₂(-R) + S₃₄(+R)

Where:
    S₁₂ = Σ (I₁ + I₂) over all pairs with factorial weights
    S₃₄ = Σ (I₃ + I₄) over all pairs with factorial weights
    M = G × M₀  (full mirror multiplier)
    M₀ = e^R + 5 = z⁷ + 5  (for K=3)
    G = f_I1 × g_I1 + (1-f_I1) × g_I2

z-Basis Structure (z = e^{R/7}):
    z⁷ = e^R     (appears in M₀)
    z¹⁴ = e^{2R}  (appears in I₁, I₂, I₃, I₄)

For R → -R (mirror transformation):
    z → 1/z
    z¹⁴ → z⁻¹⁴
    z⁷ → z⁻⁷

Usage:
    python -m src.path_a.mirror_assembly
"""
import sympy as sp
from sympy import (
    Rational, symbols, exp, simplify, expand, together,
    fraction, N, factorial
)
from typing import Dict, Tuple

from src.path_a.optimal_coeffs import R, theta, R_star_approx

# Constants for K=3
K = 3

# Symbol for z-basis
z = symbols('z', positive=True)


def compute_g_I1() -> sp.Expr:
    """
    Compute g_I1 correction factor symbolically.

    Formula: g_I1 = 1 + θ(1-θ)(2(K-1)+θ) / [8K(2K+1)²]
    """
    numerator = theta * (1 - theta) * (2*(K-1) + theta)
    denominator = 8 * K * (2*K + 1)**2
    return 1 + numerator / denominator


def compute_g_I2() -> sp.Expr:
    """
    Compute g_I2 correction factor symbolically.

    Formula: g_I2 = 1 + (2-θ)θ / [2K(2K+1)]
    """
    return 1 + theta * (2 - theta) / (2 * K * (2*K + 1))


def compute_M0_symbolic() -> sp.Expr:
    """
    Compute M₀ = e^R + (2K-1) symbolically.

    In z-basis: M₀ = z⁷ + 5  (for K=3)

    Returns expression in terms of R (symbolic).
    """
    return exp(R) + (2*K - 1)


def compute_M0_z_basis() -> sp.Expr:
    """
    Express M₀ in z-basis.

    M₀ = z⁷ + 5  (since e^R = z⁷)
    """
    return z**7 + 5


def get_factorial_weight(ell1: int, ell2: int) -> sp.Expr:
    """
    Get the factorial weight for pair (ℓ₁, ℓ₂).

    Weight = 1/(ℓ₁! × ℓ₂!)

    For off-diagonal pairs, we include symmetry factor 2 in the sum.
    """
    return Rational(1, factorial(ell1) * factorial(ell2))


def compute_I2_all_pairs() -> Dict:
    """
    Compute I₂ for all pairs using the numeric paper regime.

    Returns dict with numeric I₂ values and symbolic expressions.
    """
    from src.path_a.u_integral_symbolic import compute_all_symbolic, _extract_piecewise_main_branch
    from src.path_a.c_in_y_basis import compute_T_symbolic, compute_I2_z_basis

    # Get U results
    U_results = compute_all_symbolic(verbose=False)

    # Get T components
    T_expr, T_z14, T_const, T_den = compute_T_symbolic(verbose=False)

    # Compute I₂ for all pairs
    I2_raw = compute_I2_z_basis(U_results, T_expr, T_z14, T_const, T_den, verbose=False)

    # Convert to our format with numeric values
    results = {}
    for (ell1, ell2), r in I2_raw.items():
        z_coeffs = r['z_coeffs']
        den = r['denominator']

        # Compute numeric value at R*
        z_star = float(N(exp(R_star_approx / 7), 20))
        den_val = float(N(den.subs(R, R_star_approx), 20))

        I2_val = 0.0
        for p, c in z_coeffs.items():
            if c != 0:
                c_val = float(N(c.subs(R, R_star_approx), 20))
                I2_val += c_val * (z_star ** p)
        I2_val /= den_val

        results[(ell1, ell2)] = {
            'I2_numeric': I2_val,
            'z_coeffs': z_coeffs,
            'denominator': den,
        }

    return results


def compute_S12_numeric(I1_results: Dict, I2_results: Dict) -> float:
    """
    Compute S₁₂(+R) = Σ weight × (I₁ + I₂) over all pairs.

    Returns numeric S12 at R*.
    """
    pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

    S12_numeric = 0.0

    for ell1, ell2 in pairs:
        weight = float(get_factorial_weight(ell1, ell2))
        sym_factor = 2 if ell1 != ell2 else 1

        I1_val = I1_results[(ell1, ell2)]['I1_numeric']
        I2_val = I2_results[(ell1, ell2)]['I2_numeric']

        contrib = weight * sym_factor * (I1_val + I2_val)
        S12_numeric += float(contrib)

    return S12_numeric


def compute_S34_numeric(I34_results: Dict) -> float:
    """
    Compute S₃₄(+R) = Σ weight × (I₃ + I₄) over all pairs.

    Args:
        I34_results: Dict from i34_symbolic with keys (ell1, ell2)

    Returns:
        S34_plus_numeric at R*
    """
    pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]

    S34_numeric = 0.0

    for ell1, ell2 in pairs:
        weight = float(get_factorial_weight(ell1, ell2))
        sym_factor = 2 if ell1 != ell2 else 1

        I3_val = I34_results[(ell1, ell2)]['I3_numeric']
        I4_val = I34_results[(ell1, ell2)]['I4_numeric']

        contrib = weight * sym_factor * (I3_val + I4_val)
        S34_numeric += float(contrib)

    return S34_numeric


def compute_S12_minus(I1_results: Dict, I2_results: Dict) -> float:
    """
    Compute S₁₂(-R) by evaluating at -R* numerically.

    For symbolic: this involves z → 1/z substitution.
    """
    from src.path_a.u_integral_symbolic import _extract_piecewise_main_branch

    pairs = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
    S12_minus = 0.0

    for ell1, ell2 in pairs:
        weight = float(get_factorial_weight(ell1, ell2))
        sym_factor = 2 if ell1 != ell2 else 1

        # Evaluate I₁ at -R*
        I1_expr = I1_results[(ell1, ell2)]['I1_expr']
        I1_main = _extract_piecewise_main_branch(I1_expr)
        I1_minus = float(N(I1_main.subs(R, -R_star_approx), 20))

        # Evaluate I₂ at -R* using z-basis
        z_coeffs = I2_results[(ell1, ell2)].get('z_coeffs', {})
        den = I2_results[(ell1, ell2)].get('denominator', sp.Integer(1))

        if z_coeffs:
            # z at -R* = e^{-R*/7}
            z_star_minus = float(N(exp(-R_star_approx / 7), 20))
            den_val = float(N(den.subs(R, -R_star_approx), 20))

            I2_minus = 0.0
            for p, c in z_coeffs.items():
                if c != 0:
                    c_val = float(N(c.subs(R, -R_star_approx), 20))
                    I2_minus += c_val * (z_star_minus ** p)
            I2_minus /= den_val
        else:
            I2_minus = 0.0

        contrib = weight * sym_factor * (I1_minus + I2_minus)
        S12_minus += contrib

    return S12_minus


def compute_full_assembly(verbose: bool = True) -> Dict:
    """
    Compute the full c(R) using mirror assembly.

    c(R) = S₁₂(+R) + M × S₁₂(-R) + S₃₄(+R)

    Returns dict with all components and final c value.
    """
    from src.path_a.i1_symbolic import compute_I1_all_pairs
    from src.path_a.i34_symbolic import compute_I34_all_pairs

    print("=" * 70)
    print("STEP 4: MIRROR ASSEMBLY")
    print("=" * 70)

    # Compute all integrals
    print("\n--- Computing I₁ for all pairs ---")
    I1_results = compute_I1_all_pairs(verbose=False)

    print("--- Computing I₂ for all pairs ---")
    I2_results = compute_I2_all_pairs()

    print("--- Computing I₃/I₄ for all pairs ---")
    I34_results = compute_I34_all_pairs(verbose=False)

    # Compute S₁₂(+R)
    S12_plus = compute_S12_numeric(I1_results, I2_results)
    print(f"\nS₁₂(+R*) = {S12_plus:.10f}")

    # Compute S₁₂(-R)
    S12_minus = compute_S12_minus(I1_results, I2_results)
    print(f"S₁₂(-R*) = {S12_minus:.10f}")

    # Compute S₃₄(+R)
    S34_plus = compute_S34_numeric(I34_results)
    print(f"S₃₄(+R*) = {S34_plus:.10f}")

    # Compute I₁(-R) / (I₁(-R) + I₂(-R)) for f_I1
    I1_minus_total = 0.0
    I2_minus_total = 0.0
    from src.path_a.u_integral_symbolic import _extract_piecewise_main_branch

    for ell1, ell2 in [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]:
        weight = float(get_factorial_weight(ell1, ell2))
        sym_factor = 2 if ell1 != ell2 else 1

        # I₁ at -R*
        I1_expr = I1_results[(ell1, ell2)]['I1_expr']
        I1_main = _extract_piecewise_main_branch(I1_expr)
        I1_minus_total += weight * sym_factor * float(N(I1_main.subs(R, -R_star_approx), 20))

        # I₂ at -R* from z-basis
        z_coeffs = I2_results[(ell1, ell2)].get('z_coeffs', {})
        den = I2_results[(ell1, ell2)].get('denominator', sp.Integer(1))

        if z_coeffs:
            z_star_minus = float(N(exp(-R_star_approx / 7), 20))
            den_val = float(N(den.subs(R, -R_star_approx), 20))

            I2_minus_val = 0.0
            for p, c in z_coeffs.items():
                if c != 0:
                    c_val = float(N(c.subs(R, -R_star_approx), 20))
                    I2_minus_val += c_val * (z_star_minus ** p)
            I2_minus_val /= den_val
            I2_minus_total += weight * sym_factor * I2_minus_val

    f_I1 = I1_minus_total / (I1_minus_total + I2_minus_total) if abs(I1_minus_total + I2_minus_total) > 1e-15 else 0.5
    print(f"\nf_I1 = I₁(-R)/(I₁(-R)+I₂(-R)) = {f_I1:.6f}")

    # Compute g factors
    g_I1 = float(compute_g_I1())
    g_I2 = float(compute_g_I2())
    G = f_I1 * g_I1 + (1 - f_I1) * g_I2
    print(f"\ng_I1 = {g_I1:.8f}")
    print(f"g_I2 = {g_I2:.8f}")
    print(f"G = f_I1*g_I1 + (1-f_I1)*g_I2 = {G:.8f}")

    # Compute M₀ and M
    M0 = float(compute_M0_symbolic().subs(R, R_star_approx))
    M = G * M0
    print(f"\nM₀ = e^R* + 5 = {M0:.8f}")
    print(f"M = G × M₀ = {M:.8f}")

    # Compute c(R*)
    c_value = S12_plus + M * S12_minus + S34_plus
    print(f"\n{'='*60}")
    print(f"ASSEMBLY: c(R*) = S₁₂(+R) + M × S₁₂(-R) + S₃₄(+R)")
    print(f"{'='*60}")
    print(f"  S₁₂(+R*) = {S12_plus:.10f}")
    print(f"  M        = {M:.10f}")
    print(f"  S₁₂(-R*) = {S12_minus:.10f}")
    print(f"  M×S₁₂(-R*) = {M * S12_minus:.10f}")
    print(f"  S₃₄(+R*) = {S34_plus:.10f}")
    print(f"  ─────────────────────────────")
    print(f"  c(R*)    = {c_value:.10f}")
    print(f"  Target   = 1.0000000000")
    print(f"  Gap      = {(c_value - 1.0)*100:.4f}%")

    return {
        'S12_plus': S12_plus,
        'S12_minus': S12_minus,
        'S34_plus': S34_plus,
        'f_I1': f_I1,
        'g_I1': g_I1,
        'g_I2': g_I2,
        'G': G,
        'M0': M0,
        'M': M,
        'c': c_value,
        'I1_results': I1_results,
        'I2_results': I2_results,
        'I34_results': I34_results,
    }


def main():
    result = compute_full_assembly(verbose=True)

    print("\n" + "=" * 70)
    print("STEP 4 COMPLETE: MIRROR ASSEMBLY")
    print("=" * 70)

    # Summary in z-basis
    print("\nMirror Multiplier in z-basis:")
    print(f"  M₀ = z⁷ + 5  (where z = e^{{R/7}})")
    print(f"  M = G × M₀ = {result['G']:.6f} × (z⁷ + 5)")

    print(f"\nc(R*) = {result['c']:.10f}")
    print(f"Target c = 1.0")

    return result


if __name__ == "__main__":
    result = main()

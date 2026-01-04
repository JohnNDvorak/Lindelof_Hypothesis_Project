#!/usr/bin/env python3
"""
Optimal polynomial coefficients that achieve c(R*) = 1.

These are the coefficients from our optimization that give:
- κ_main = 1 (i.e., c = 1 exactly at R*)
- R* ≈ 1.149760231537150...

Path A goal: Prove c(R*) = 1 algebraically using these exact coefficients.
"""
from sympy import Rational, symbols

# =============================================================================
# Symbol
# =============================================================================
R = symbols('R', real=True)

# =============================================================================
# θ = 4/7 (PRZZ universal)
# =============================================================================
theta = Rational(4, 7)

# =============================================================================
# P̃₁ coefficients - EXACT RATIONALS
# =============================================================================
# These are the "universal tilde basis" coefficients from our optimization.
# P₁(x) = x + x(1-x)·P̃₁(1-x) where P̃₁(v) = a₀ + a₁v + a₂v² + a₃v³
#
# From precomputed: [-2.0, 0.9375, 1.0, -0.6]
# Exact rational forms:
P1_tilde = {
    'a0': Rational(-2, 1),     # = -2
    'a1': Rational(15, 16),    # = 0.9375
    'a2': Rational(1, 1),      # = 1
    'a3': Rational(-3, 5),     # = -0.6
}

# =============================================================================
# P̃₂ coefficients
# =============================================================================
# From precomputed: [0.5241, 1.3199, -0.9401]
# P₂(x) = x·P̃₂(x) where P̃₂(x) = b₀ + b₁x + b₂x²
#
# These need rational reconstruction for exact Path A proof.
# For now, use approximate rationals:
P2_tilde = {
    'b0': Rational(5241, 10000),      # ≈ 0.5241
    'b1': Rational(13199, 10000),     # ≈ 1.3199
    'b2': Rational(-9401, 10000),     # ≈ -0.9401
}

# =============================================================================
# P̃₃ coefficients
# =============================================================================
# From precomputed: [0.1367, -0.6865, -0.0499]
# P₃(x) = x·P̃₃(x) where P̃₃(x) = c₀ + c₁x + c₂x²
#
# These need rational reconstruction for exact Path A proof.
P3_tilde = {
    'c0': Rational(1367, 10000),      # ≈ 0.1367
    'c1': Rational(-6865, 10000),     # ≈ -0.6865
    'c2': Rational(-499, 10000),      # ≈ -0.0499
}

# =============================================================================
# Q coefficients in (1-2t)^k basis
# =============================================================================
# Q(t) = q₀ + q₁(1-2t) + q₃(1-2t)³ + q₅(1-2t)⁵
# with Q(0) = 1 constraint: q₀ + q₁ + q₃ + q₅ = 1
#
# From precomputed: {"0": 0.490464, "1": 0.636851, "3": -0.159327, "5": 0.032011}
# Note: 0.490464 + 0.636851 - 0.159327 + 0.032011 = 0.999999 ≈ 1
Q_coeffs = {
    'q0': Rational(490464, 1000000),   # ≈ 0.490464
    'q1': Rational(636851, 1000000),   # ≈ 0.636851
    'q3': Rational(-159327, 1000000),  # ≈ -0.159327
    'q5': Rational(32011, 1000000),    # ≈ 0.032011
}

# Enforce Q(0) = 1 exactly
Q_coeffs['q0'] = 1 - (Q_coeffs['q1'] + Q_coeffs['q3'] + Q_coeffs['q5'])

# =============================================================================
# R* - the optimal value where c = 1
# =============================================================================
# From numerical optimization: R* ≈ 1.149760231537150...
#
# For Path A, we'll define R* as the root of F(R) = numerator(c(R) - 1)
# This gives an EXACT definition once we have the algebraic formula.
R_star_approx = 1.149760231537150


# =============================================================================
# Polynomial builders (symbolic)
# =============================================================================
def build_P1(x):
    """Build P₁(x) = x + x(1-x)·P̃₁(1-x)."""
    v = 1 - x
    tilde = (P1_tilde['a0'] + P1_tilde['a1']*v +
             P1_tilde['a2']*v**2 + P1_tilde['a3']*v**3)
    return x + x*v*tilde


def build_P2(x):
    """Build P₂(x) = x·P̃₂(x)."""
    return x * (P2_tilde['b0'] + P2_tilde['b1']*x + P2_tilde['b2']*x**2)


def build_P3(x):
    """Build P₃(x) = x·P̃₃(x)."""
    return x * (P3_tilde['c0'] + P3_tilde['c1']*x + P3_tilde['c2']*x**2)


def build_Q(t):
    """Build Q(t) in (1-2t)^k basis."""
    w = 1 - 2*t
    return (Q_coeffs['q0'] + Q_coeffs['q1']*w +
            Q_coeffs['q3']*w**3 + Q_coeffs['q5']*w**5)


def get_P(ell, x):
    """Get P_ℓ(x) for ℓ ∈ {1, 2, 3}."""
    if ell == 1:
        return build_P1(x)
    elif ell == 2:
        return build_P2(x)
    elif ell == 3:
        return build_P3(x)
    else:
        raise ValueError(f"Invalid piece index: {ell}")


# =============================================================================
# Display coefficients
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("OPTIMAL COEFFICIENTS FOR PATH A (c = 1)")
    print("=" * 60)

    print(f"\nθ = {theta} = {float(theta):.10f}")

    print(f"\n--- P̃₁ coefficients (exact rationals) ---")
    for name, val in P1_tilde.items():
        print(f"  {name} = {val} = {float(val):.10f}")

    print(f"\n--- P̃₂ coefficients ---")
    for name, val in P2_tilde.items():
        print(f"  {name} = {val} = {float(val):.10f}")

    print(f"\n--- P̃₃ coefficients ---")
    for name, val in P3_tilde.items():
        print(f"  {name} = {val} = {float(val):.10f}")

    print(f"\n--- Q coefficients (with Q(0) = 1 enforced) ---")
    for name, val in Q_coeffs.items():
        print(f"  {name} = {val} = {float(val):.10f}")

    Q0_check = sum(Q_coeffs.values())
    print(f"\n  Q(0) check: sum = {Q0_check} = {float(Q0_check):.15f}")

    print(f"\nR* ≈ {R_star_approx}")
    print("=" * 60)

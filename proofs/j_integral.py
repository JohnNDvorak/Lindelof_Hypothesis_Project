"""
J_n(λ) integral family: the core analytic ingredient for Path A.

All PRZZ integrals reduce to:
    J_n(λ) = ∫₀¹ t^n e^{λt} dt = (e^λ·P_n(λ) - n!) / λ^{n+1}

where P_n(λ) is an explicit polynomial of degree n with integer coefficients.

Recurrence:
    J_0(λ) = (e^λ - 1) / λ
    J_n(λ) = e^λ/λ - (n/λ)·J_{n-1}(λ)

For integrals with (1-t)^m factor:
    ∫₀¹ t^n (1-t)^m e^{λt} dt = Σⱼ₌₀^m (-1)^j C(m,j) J_{n+j}(λ)
"""
import sympy as sp
from sympy import Rational, factorial, binomial, exp, symbols, simplify, expand, Poly

# Symbolic variable for the exponential argument
R = symbols('R', real=True)


def J_n_closed_form(n: int, lam):
    """
    Compute J_n(λ) = ∫₀¹ t^n e^{λt} dt in closed form.

    Uses recurrence relation:
        J_0(λ) = (e^λ - 1) / λ
        J_n(λ) = e^λ/λ - n·J_{n-1}(λ)/λ

    Closed form: J_n = (e^λ Q_n(λ) - R_n) / λ^{n+1}
    where Q_n and R_n are polynomials built via recurrence.
    """
    if n < 0:
        raise ValueError("n must be non-negative")

    # Use the recurrence to build the expression
    # J_0 = (e^λ - 1) / λ
    # J_n = e^λ/λ - n J_{n-1}/λ

    # Build the closed form by unrolling the recurrence
    # J_n = (A_n e^λ + B_n) / λ^{n+1}
    # where A_n, B_n are polynomials in λ

    if n == 0:
        return (exp(lam) - 1) / lam

    # For n >= 1, use recurrence
    # Start with J_0 = (e^λ - 1)/λ => A_0 = 1, B_0 = -1 (constant)
    # J_n = e^λ/λ - n J_{n-1}/λ
    # λ^{n+1} J_n = λ^n e^λ - n λ^n J_{n-1}
    #             = λ^n e^λ - n (A_{n-1} e^λ + B_{n-1})
    #             = (λ^n - n A_{n-1}) e^λ - n B_{n-1}
    # So A_n = λ^n - n A_{n-1} and B_n = -n B_{n-1}

    # Initialize: for J_0, we have A_0 = 1, B_0 = -1
    A = sp.Integer(1)  # A_0
    B = sp.Integer(-1)  # B_0

    for k in range(1, n + 1):
        A_new = lam**k - k * A
        B_new = -k * B
        A = A_new
        B = B_new

    return (A * exp(lam) + B) / lam**(n + 1)


def J_n_recurrence(n: int, lam, cache=None):
    """
    Compute J_n(λ) using the recurrence relation.

    J_0(λ) = (e^λ - 1) / λ
    J_n(λ) = e^λ/λ - n·J_{n-1}(λ)/λ

    This is mathematically equivalent to J_n_closed_form but may have
    different simplification behavior.
    """
    if cache is None:
        cache = {}

    if n in cache:
        return cache[n]

    if n == 0:
        result = (exp(lam) - 1) / lam
    else:
        J_prev = J_n_recurrence(n - 1, lam, cache)
        result = exp(lam) / lam - n * J_prev / lam

    cache[n] = result
    return result


def integral_t_n_1mt_m_exp(n: int, m: int, lam):
    """
    Compute ∫₀¹ t^n (1-t)^m e^{λt} dt exactly.

    Uses binomial expansion:
        (1-t)^m = Σⱼ₌₀^m (-1)^j C(m,j) t^j

    So the integral becomes:
        Σⱼ₌₀^m (-1)^j C(m,j) J_{n+j}(λ)
    """
    result = 0
    for j in range(m + 1):
        coeff = ((-1)**j) * binomial(m, j)
        result += coeff * J_n_closed_form(n + j, lam)
    return result


def split_exp_form(expr, lam=None):
    """
    Given an expression in e^λ and rational functions of λ,
    rewrite as (A·e^λ + B) / D where A, B, D are polynomials in λ.

    Returns (A, B, D) as sympy expressions.
    """
    if lam is None:
        lam = R

    # Substitute z = e^λ to work with polynomials
    z = symbols('z')
    expr_sub = expr.rewrite(exp).subs(exp(lam), z)

    # Combine into single fraction
    expr_together = sp.together(expr_sub)
    num, den = sp.fraction(expr_together)

    # Expand numerator and collect powers of z
    num = expand(num)

    # The numerator should be of form A(λ)·z + B(λ) for our integrals
    # (or A(λ)·z^2 + B(λ)·z + C(λ) if we have e^{2λ} terms)
    try:
        poly = Poly(num, z)
        coeffs = poly.all_coeffs()

        if poly.degree() == 1:
            # Form: A·z + B
            A, B = coeffs
            return A, B, den
        elif poly.degree() == 2:
            # Form: A·z^2 + B·z + C
            # For e^{2R} terms, this becomes A·e^{2R} + B·e^R + C
            return coeffs[0], coeffs[1], coeffs[2], den
        else:
            # Return raw coefficients
            return tuple(coeffs) + (den,)
    except:
        # Fallback: return the expression as-is
        return expr, sp.Integer(0), sp.Integer(1)


# ============================================================
# Test / demo
# ============================================================
if __name__ == "__main__":
    print("Testing J_n integral family...")

    lam = symbols('lambda', real=True)

    # Test J_0
    J0 = J_n_closed_form(0, lam)
    print(f"\nJ_0(λ) = {simplify(J0)}")
    print(f"  Expected: (e^λ - 1)/λ")

    # Test J_1
    J1 = J_n_closed_form(1, lam)
    print(f"\nJ_1(λ) = {simplify(J1)}")

    # Test J_2
    J2 = J_n_closed_form(2, lam)
    print(f"\nJ_2(λ) = {simplify(J2)}")

    # Test integral with (1-t) factor
    I_01 = integral_t_n_1mt_m_exp(0, 1, lam)
    print(f"\n∫₀¹ (1-t) e^λt dt = {simplify(I_01)}")

    # Verify: should equal (e^λ(λ-1) + 1)/λ^2
    expected = (exp(lam) * (lam - 1) + 1) / lam**2
    print(f"  Expected: {simplify(expected)}")
    print(f"  Match: {simplify(I_01 - expected) == 0}")

    # Test with λ = 2R (the PRZZ case)
    print("\n--- With λ = 2R ---")
    J0_2R = J_n_closed_form(0, 2*R)
    print(f"J_0(2R) = {simplify(J0_2R)}")

    print("\nJ_n module ready for Path A.")

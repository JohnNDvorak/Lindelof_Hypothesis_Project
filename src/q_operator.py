"""
src/q_operator.py
Q operator binomial-lift implementation (Priority C - GPT Guidance 2025-12-19)

The key insight from PRZZ mirror assembly:
- Q is applied as Q(D) where D is a differential operator
- For mirror terms, we need Q(1+D) instead of Q(D)
- Q(1+D) = sum_n q_n (1+D)^n where q_n are Q's coefficients

This module provides:
1. binomial_lift_coeffs: Pure math - coefficients of Q(1+x) from Q(x)
2. apply_Q_shift: Operator mode - Q(1+D) application to integrand
"""

from __future__ import annotations

from typing import List, Tuple
from math import comb
import numpy as np


def poly_eval(coeffs: List[float], x: float) -> float:
    """Evaluate a polynomial in monomial-coefficient form at x.

    coeffs represent Q(x) = sum_j coeffs[j] * x^j.
    """
    return sum(c * (x ** j) for j, c in enumerate(coeffs))


def binomial_lift_coeffs(q_coeffs: List[float]) -> List[float]:
    """
    Compute coefficients for Q(1+x) from Q(x).

    If Q(x) = sum_j q_j x^j, then Q(1+x) = sum_r q'_r x^r
    where:
        q'_r = sum_{j>=r} q_j * C(j,r)

    This is the "argument shift" transform: Q(x) -> Q(1+x).

    Args:
        q_coeffs: Coefficients [q_0, q_1, q_2, ...] of Q(x) = sum q_j x^j

    Returns:
        Coefficients [q'_0, q'_1, q'_2, ...] of Q(1+x) = sum q'_r x^r

    Example:
        If Q(x) = 1 + 2x + 3x^2, then
        Q(1+x) = Q at (1+x)
               = 1 + 2(1+x) + 3(1+x)^2
               = 1 + 2 + 2x + 3(1 + 2x + x^2)
               = 1 + 2 + 3 + (2 + 6)x + 3x^2
               = 6 + 8x + 3x^2

        So binomial_lift_coeffs([1, 2, 3]) == [6, 8, 3]
    """
    if not q_coeffs:
        return []

    n = len(q_coeffs)
    result = [0.0] * n

    # q'_r = sum_{j>=r} q_j * C(j, r)
    for r in range(n):
        total = 0.0
        for j in range(r, n):
            total += q_coeffs[j] * comb(j, r)
        result[r] = total

    return result


def binomial_shift_coeffs(q_coeffs: List[float], shift: float = 1.0) -> List[float]:
    """
    Compute coefficients for Q(shift + x) from Q(x).

    Generalization of binomial_lift_coeffs for arbitrary shift.

    If Q(x) = sum_j q_j x^j, then Q(shift+x) = sum_r q'_r x^r
    where:
        q'_r = sum_{j>=r} q_j * C(j,r) * shift^{j-r}

    Args:
        q_coeffs: Coefficients [q_0, q_1, ...] of Q(x)
        shift: The constant to add (default 1.0)

    Returns:
        Coefficients of Q(shift+x)
    """
    if not q_coeffs:
        return []

    n = len(q_coeffs)
    result = [0.0] * n

    for r in range(n):
        total = 0.0
        for j in range(r, n):
            # shift^{j-r} * C(j,r) * q_j
            total += q_coeffs[j] * comb(j, r) * (shift ** (j - r))
        result[r] = total

    return result


def lift_poly_by_shift(poly, *, shift: float = 1.0):
    """Return a monomial-basis Polynomial representing x -> poly(x + shift).

    This is a convenience wrapper around `binomial_shift_coeffs` for PRZZ's
    polynomial objects (which expose `.to_monomial()`).
    """
    from src.polynomials import Polynomial
    import math

    to_monomial = getattr(poly, "to_monomial", None)
    if callable(to_monomial):
        mono = to_monomial()
        coeffs = getattr(mono, "coeffs", None)
        if coeffs is None:
            raise TypeError("to_monomial() did not return an object with .coeffs")
        q_coeffs = list(np.asarray(coeffs, dtype=float))
        shifted_coeffs = binomial_shift_coeffs(q_coeffs, shift=shift)
        return Polynomial(np.asarray(shifted_coeffs, dtype=float))

    degree = getattr(poly, "degree", None)
    if degree is None:
        raise TypeError("lift_poly_by_shift requires poly.to_monomial() or poly.degree")

    x0 = np.asarray([0.0], dtype=float)
    q_coeffs = [
        float(poly.eval_deriv(x0, k)[0]) / math.factorial(k)
        for k in range(int(degree) + 1)
    ]
    shifted_coeffs = binomial_shift_coeffs(q_coeffs, shift=shift)
    return Polynomial(np.asarray(shifted_coeffs, dtype=float))


def verify_binomial_lift(q_coeffs: List[float], test_points: List[float] = None) -> Tuple[bool, float]:
    """
    Verify binomial_lift_coeffs by comparing Q(1+x) vs Q_lift(x) at test points.

    Args:
        q_coeffs: Coefficients of Q(x)
        test_points: Points to test (default: [0.0, 0.5, 1.0, -0.5])

    Returns:
        (success, max_error): Boolean and maximum absolute error
    """
    if test_points is None:
        test_points = [0.0, 0.5, 1.0, -0.5, 0.25, -0.25]

    lifted_coeffs = binomial_lift_coeffs(q_coeffs)

    max_error = 0.0

    for x in test_points:
        # Evaluate Q(1+x) directly
        q_at_1_plus_x = sum(c * ((1.0 + x) ** j) for j, c in enumerate(q_coeffs))

        # Evaluate Q_lift(x)
        q_lift_at_x = sum(c * (x ** r) for r, c in enumerate(lifted_coeffs))

        error = abs(q_at_1_plus_x - q_lift_at_x)
        max_error = max(max_error, error)

    success = max_error < 1e-10
    return success, max_error


def przz_basis_to_standard_coeffs(basis_coeffs: List[Tuple[int, float]]) -> List[float]:
    """
    Convert PRZZ (1-2x)^k basis coefficients to standard polynomial coefficients.

    PRZZ Q(x) = sum_k c_k (1-2x)^k

    We need standard form Q(x) = sum_j q_j x^j for the binomial lift.

    Args:
        basis_coeffs: List of (k, c_k) pairs for (1-2x)^k terms

    Returns:
        Standard coefficients [q_0, q_1, q_2, ...]
    """
    if not basis_coeffs:
        return [0.0]

    # Find max power
    max_k = max(k for k, _ in basis_coeffs)
    max_degree = max_k  # (1-2x)^k has degree k

    result = np.zeros(max_degree + 1)

    for k, c_k in basis_coeffs:
        # (1-2x)^k = sum_{j=0}^{k} C(k,j) * 1^{k-j} * (-2x)^j
        #          = sum_{j=0}^{k} C(k,j) * (-2)^j * x^j
        for j in range(k + 1):
            coeff = comb(k, j) * ((-2) ** j)
            result[j] += c_k * coeff

    return list(result)


def poly_multiply_coeffs(a_coeffs: List[float], b_coeffs: List[float]) -> List[float]:
    """
    Multiply two polynomials given their coefficients.

    If A(x) = sum_i a_i x^i and B(x) = sum_j b_j x^j,
    then (A·B)(x) = sum_k c_k x^k where c_k = sum_{i+j=k} a_i b_j.

    Args:
        a_coeffs: Coefficients of A(x)
        b_coeffs: Coefficients of B(x)

    Returns:
        Coefficients of (A·B)(x)
    """
    if not a_coeffs or not b_coeffs:
        return [0.0]

    n_a = len(a_coeffs)
    n_b = len(b_coeffs)
    n_result = n_a + n_b - 1

    result = [0.0] * n_result
    for i, a_i in enumerate(a_coeffs):
        for j, b_j in enumerate(b_coeffs):
            result[i + j] += a_i * b_j

    return result


def binomial_lift_product_coeffs(q_coeffs: List[float]) -> List[float]:
    """
    Compute coefficients for (Q·Q)(1+x) from Q(x).

    This is the "joint" Q-shift: first form Q², then apply binomial lift.

    (Q·Q)_lift(x) = Q(1+x)·Q(1+x) = (Q_lift(x))²

    Note: This is mathematically equivalent to lifting each Q independently
    and then multiplying. The difference only matters if we're doing
    something special with the product structure.

    Args:
        q_coeffs: Coefficients of Q(x)

    Returns:
        Coefficients of (Q²)(1+x) = (Q(1+x))²
    """
    # First lift Q
    q_lift = binomial_lift_coeffs(q_coeffs)

    # Then square the lifted polynomial
    return poly_multiply_coeffs(q_lift, q_lift)


def binomial_lift_ratio(q_coeffs: List[float]) -> float:
    """
    Compute the ratio Q_lift(0) / Q(0) = Q(1) / Q(0).

    This is useful for understanding the scaling effect of the lift.

    Args:
        q_coeffs: Coefficients of Q(x)

    Returns:
        Q(1) / Q(0) if Q(0) != 0, else inf
    """
    q_0 = q_coeffs[0] if q_coeffs else 0.0
    q_1 = sum(q_coeffs) if q_coeffs else 0.0  # Q(1) = sum of all coeffs

    if abs(q_0) < 1e-15:
        return float('inf')
    return q_1 / q_0


def standard_to_przz_basis_coeffs(
    standard_coeffs: List[float],
    max_k: int = None
) -> List[Tuple[int, float]]:
    """
    Convert standard polynomial coefficients to PRZZ (1-2x)^k basis.

    This is the inverse of przz_basis_to_standard_coeffs.

    Note: This requires solving a linear system. The PRZZ basis may not
    represent all polynomials exactly (only odd powers k=1,3,5,...).

    Args:
        standard_coeffs: Standard coefficients [q_0, q_1, ...]
        max_k: Maximum power k to use (default: len(standard_coeffs)-1)

    Returns:
        List of (k, c_k) pairs for non-zero terms
    """
    n = len(standard_coeffs)
    if max_k is None:
        max_k = n - 1

    # Build matrix A where A[j,k] = coefficient of x^j in (1-2x)^k
    A = np.zeros((n, max_k + 1))
    for k in range(max_k + 1):
        for j in range(min(k + 1, n)):
            A[j, k] = comb(k, j) * ((-2) ** j)

    # Solve for c_k: A @ c = standard_coeffs
    # Use least squares since system may be overdetermined
    c_vec, residuals, rank, s = np.linalg.lstsq(A, standard_coeffs, rcond=None)

    # Return non-zero terms
    result = []
    for k, c_k in enumerate(c_vec):
        if abs(c_k) > 1e-12:
            result.append((k, float(c_k)))

    return result

"""src/mollifier_profiles.py

Mollifier profile Taylor coefficient generators for PRZZ omega-case handling.

This module provides Taylor coefficient generators for profile functions that
appear in the Term DSL (see `src/term_dsl.py`).

The Term DSL composes factors of the form F(u + δ) where δ is a nilpotent
perturbation in formal variables. The composition code uses the "factorial
basis":

    taylor_coeffs[j] == d^j/du^j F(u)   (NOT divided by j!)

Case B (ω = 0):
    F(u) = P(u)

Case C (ω > 0):
    F(u) = K_ω(u; R, θ)

    K_ω(u; R, θ) = u^ω/(ω-1)! · ∫₀¹ a^{ω-1} P((1-a)u) exp(R θ u a) da

Important: For Case C, derivatives must include:
  - derivatives of P((1-a)u) via chain rule
  - derivatives of exp(R θ u a)
  - derivatives of the prefactor u^ω

The earlier archived implementation differentiated only the P term and is
mathematically incorrect even for P ≡ 1.
"""

from __future__ import annotations

from math import comb, factorial
from typing import Protocol

import numpy as np

from src.quadrature import gauss_legendre_01


class PolyLike(Protocol):
    """Protocol for polynomial-like objects used by the DSL."""

    def eval_deriv(self, x: np.ndarray, k: int) -> np.ndarray: ...


def case_b_taylor_coeffs(P: PolyLike, u: float, max_order: int) -> np.ndarray:
    """Case B (ω=0): Taylor coefficients for P(u+δ).

    Returns:
        coeff[j] = P^{(j)}(u)  (factorial basis; NOT divided by j!)
    """

    u_arr = np.array([float(u)], dtype=float)
    return np.array([float(P.eval_deriv(u_arr, j)[0]) for j in range(max_order + 1)], dtype=float)


def case_c_taylor_coeffs(
    P: PolyLike,
    u: float,
    omega: int,
    R: float,
    theta: float,
    max_order: int,
    n_quad_a: int = 40,
) -> np.ndarray:
    """Case C (ω>0): Taylor coefficients for K_ω(u+δ; R, θ).

    We compute derivatives under the integral sign.

    Let γ := Rθ and
        I(u) := ∫₀¹ a^{ω-1} P((1-a)u) exp(γ u a) da
        pref(u) := u^ω/(ω-1)!
        K(u) := pref(u) · I(u)

    Then:
        K^{(j)}(u) = Σ_{m=0}^{min(j,ω)} C(j,m) · pref^{(m)}(u) · I^{(j-m)}(u)

    And for r ≥ 0:
        I^{(r)}(u) = ∫₀¹ a^{ω-1} exp(γ u a)
                     Σ_{k=0}^r C(r,k) (1-a)^k P^{(k)}((1-a)u) (γ a)^{r-k} da

    Returns:
        coeff[j] = K_ω^{(j)}(u; R, θ)  (factorial basis; NOT divided by j!)
    """

    if omega <= 0:
        raise ValueError(f"Case C requires omega > 0, got {omega}")
    if max_order < 0:
        raise ValueError(f"max_order must be non-negative, got {max_order}")

    u = float(u)
    gamma = float(R) * float(theta)

    a_nodes, a_weights = gauss_legendre_01(int(n_quad_a))
    a_nodes = np.asarray(a_nodes, dtype=float)
    a_weights = np.asarray(a_weights, dtype=float)

    one_minus_a = 1.0 - a_nodes
    a_power = a_nodes ** (omega - 1)  # ω=1 -> 1

    # Arguments for P: (1-a)u
    args = one_minus_a * u

    # Precompute P^{(k)}((1-a)u) for k=0..max_order
    P_derivs = [P.eval_deriv(args, k).astype(float, copy=False) for k in range(max_order + 1)]

    exp_factor = np.exp(gamma * u * a_nodes)

    # I^{(r)}(u) for r=0..max_order
    I_derivs = np.zeros(max_order + 1, dtype=float)
    for r in range(max_order + 1):
        sum_terms = np.zeros_like(a_nodes, dtype=float)
        for k in range(r + 1):
            sum_terms += (
                comb(r, k)
                * (one_minus_a ** k)
                * P_derivs[k]
                * ((gamma * a_nodes) ** (r - k))
            )
        I_derivs[r] = float(np.sum(a_weights * a_power * exp_factor * sum_terms))

    # Combine with prefactor derivatives
    denom = factorial(omega - 1)
    omega_fact = factorial(omega)

    K_derivs = np.zeros(max_order + 1, dtype=float)
    for j in range(max_order + 1):
        total = 0.0
        for m in range(0, min(j, omega) + 1):
            fall = omega_fact / factorial(omega - m)
            pref_m = (fall * (u ** (omega - m))) / denom
            total += comb(j, m) * pref_m * I_derivs[j - m]
        K_derivs[j] = total

    return K_derivs

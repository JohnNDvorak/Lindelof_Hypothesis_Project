"""
src/unified_i2_paper.py
Phase 29: Unified I2 Evaluator with Paper Regime Kernel Attenuation

I2 is the simplest term:
- No variables (no derivatives)
- No (1-u) factor
- Uses Q(t)² not Q(A_α)×Q(A_β)

MATHEMATICAL STRUCTURE (RAW regime):
====================================
I₂_{ℓ₁,ℓ₂} = (1/θ) × ∫₀¹∫₀¹ exp(2Rt) × P_{ℓ₁}(u) × P_{ℓ₂}(u) × Q(t)² du dt

PAPER REGIME DIFFERENCE:
========================
For ell >= 2, we use K_ω(u) instead of P(u):

K_ω(u; R, θ) = u^ω/(ω-1)! × ∫₀¹ a^{ω-1} P((1-a)u) exp(Rθua) da

where omega = ell - 1.

Created: 2025-12-26 (Phase 29)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import math
import numpy as np

from src.quadrature import gauss_legendre_01
from src.mollifier_profiles import case_c_taylor_coeffs, PolyLike


@dataclass
class UnifiedI2PaperResult:
    """Result of unified I2 evaluation using paper regime."""

    ell1: int
    ell2: int
    I2_value: float

    omega1: int
    omega2: int

    n_quad_u: int
    n_quad_t: int
    n_quad_a: int
    include_Q: bool


def omega_for_ell(ell: int) -> int:
    """Get omega value for piece index: omega = ell - 1."""
    return ell - 1


def eval_P_paper(
    poly: PolyLike,
    u: float,
    omega: int,
    R: float,
    theta: float,
    n_quad_a: int = 40,
) -> float:
    """
    Evaluate polynomial using paper regime kernel.

    For omega=0 (Case B): returns P(u)
    For omega>0 (Case C): returns K_omega(u; R, theta)

    K_ω(u) is the 0th order Taylor coefficient (the value itself).
    """
    if omega == 0:
        # Case B: direct polynomial evaluation
        u_arr = np.array([u], dtype=float)
        return float(poly.eval_deriv(u_arr, 0)[0])
    else:
        # Case C: compute K_omega(u) - the 0th Taylor coefficient
        taylor_coeffs = case_c_taylor_coeffs(
            poly, u, omega, R, theta, max_order=0, n_quad_a=n_quad_a
        )
        return float(taylor_coeffs[0])


def _extract_poly_coeffs(poly: PolyLike) -> List[float]:
    """Extract standard basis polynomial coefficients."""
    if hasattr(poly, 'standard_coeffs'):
        return list(poly.standard_coeffs)
    if hasattr(poly, 'tilde_coeffs') and poly.tilde_coeffs is not None:
        n_coeffs = len(poly.tilde_coeffs) + 2
        u_points = np.linspace(0, 1, n_coeffs + 5)
        y_points = poly.eval(u_points)
        coeffs = np.polyfit(u_points, y_points, len(poly.tilde_coeffs) + 1)
        return list(coeffs[::-1])
    if hasattr(poly, 'eval'):
        u_points = np.linspace(0, 1, 15)
        y_points = poly.eval(u_points)
        coeffs = np.polyfit(u_points, y_points, 5)
        return list(coeffs[::-1])
    raise ValueError(f"Cannot extract coefficients from polynomial: {type(poly)}")


def eval_polynomial(coeffs: List[float], x: float) -> float:
    """Evaluate polynomial using Horner's method."""
    result = 0.0
    for c in reversed(coeffs):
        result = result * x + c
    return result


def compute_I2_unified_paper(
    R: float,
    theta: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    n_quad_u: int = 60,
    n_quad_t: int = 60,
    n_quad_a: int = 40,
    include_Q: bool = True,
) -> UnifiedI2PaperResult:
    """
    Compute I2 for pair (ell1, ell2) using PAPER regime kernels.

    I2 = (1/θ) × ∫₀¹∫₀¹ exp(2Rt) × F_{ℓ₁}(u) × F_{ℓ₂}(u) × Q(t)² du dt

    Where F is either P (omega=0) or K_omega (omega>0).

    Args:
        R: PRZZ R parameter
        theta: PRZZ theta parameter
        ell1, ell2: Piece indices
        polynomials: Dict with P1, P2, P3, Q
        n_quad_u, n_quad_t: Quadrature points for u, t integrals
        n_quad_a: Quadrature points for Case C a-integral
        include_Q: Whether to include Q factor

    Returns:
        UnifiedI2PaperResult with I2 value
    """
    omega1 = omega_for_ell(ell1)
    omega2 = omega_for_ell(ell2)

    u_nodes, u_weights = gauss_legendre_01(n_quad_u)
    t_nodes, t_weights = gauss_legendre_01(n_quad_t)

    P_ell1 = polynomials.get(f"P{ell1}")
    P_ell2 = polynomials.get(f"P{ell2}")
    Q = polynomials.get("Q")

    if P_ell1 is None or P_ell2 is None:
        raise ValueError(f"Missing polynomial P{ell1} or P{ell2}")

    Q_coeffs = _extract_poly_coeffs(Q) if Q is not None and include_Q else None

    total = 0.0

    for u, u_w in zip(u_nodes, u_weights):
        # Evaluate P/K factors at u using paper regime
        F1_val = eval_P_paper(P_ell1, u, omega1, R, theta, n_quad_a)
        F2_val = eval_P_paper(P_ell2, u, omega2, R, theta, n_quad_a)
        P_product = F1_val * F2_val

        for t, t_w in zip(t_nodes, t_weights):
            # exp(2Rt)
            exp_val = math.exp(2 * R * t)

            # Q(t)²
            if include_Q and Q_coeffs is not None:
                Q_val = eval_polynomial(Q_coeffs, t)
                Q_sq = Q_val ** 2
            else:
                Q_sq = 1.0

            total += exp_val * P_product * Q_sq * u_w * t_w

    # 1/θ prefactor
    total *= 1.0 / theta

    return UnifiedI2PaperResult(
        ell1=ell1,
        ell2=ell2,
        I2_value=total,
        omega1=omega1,
        omega2=omega2,
        n_quad_u=n_quad_u,
        n_quad_t=n_quad_t,
        n_quad_a=n_quad_a,
        include_Q=include_Q,
    )


def compare_I2_raw_vs_paper(
    R: float,
    theta: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    n_quad: int = 60,
) -> Dict:
    """Compare raw vs paper regime for I2."""
    from src.unified_i2_general import compute_I2_unified_general

    raw_result = compute_I2_unified_general(
        R, theta, ell1, ell2, polynomials,
        n_quad_u=n_quad, n_quad_t=n_quad, include_Q=True,
    )

    paper_result = compute_I2_unified_paper(
        R, theta, ell1, ell2, polynomials,
        n_quad_u=n_quad, n_quad_t=n_quad, n_quad_a=40, include_Q=True,
    )

    raw_val = raw_result.I2_value
    paper_val = paper_result.I2_value

    ratio = paper_val / raw_val if abs(raw_val) > 1e-15 else float('inf')
    sign_match = (raw_val > 0) == (paper_val > 0)

    return {
        "ell1": ell1,
        "ell2": ell2,
        "raw": raw_val,
        "paper": paper_val,
        "ratio": ratio,
        "sign_match": sign_match,
        "omega1": paper_result.omega1,
        "omega2": paper_result.omega2,
    }

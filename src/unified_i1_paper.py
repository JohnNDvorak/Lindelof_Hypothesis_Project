"""
src/unified_i1_paper.py
Phase 29: Unified I1 Evaluator with Paper Regime Kernel Attenuation

This module computes I1 for any pair (ell1, ell2) using PAPER regime kernels.

PAPER REGIME vs RAW REGIME:
===========================
- RAW regime: All polynomials use Case B (direct P(u+x))
- PAPER regime: P2/P3 use Case C kernel attenuation K_omega(u+x; R, theta)

The key difference is that for ell >= 2, we use:
    K_omega(u; R, theta) = u^omega/(omega-1)! * integral_0^1 a^{omega-1} P((1-a)u) exp(R*theta*u*a) da

where omega = ell - 1 (so P2 uses omega=1, P3 uses omega=2).

This creates fundamentally different numerical values - not just a convention difference.

CRITICAL PAIRS (from Phase 28 discovery):
=========================================
- (2,2): paper regime shrinks by ~4x vs raw (ratio ~0.24)
- (1,3), (2,3): paper regime FLIPS SIGN vs raw

These are the validation targets for this implementation.

Created: 2025-12-26 (Phase 29)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import math
import numpy as np

from src.quadrature import gauss_legendre_01
from src.series_bivariate import (
    BivariateSeries,
    build_exp_bracket,
    build_log_factor,
    build_P_factor,
    build_Q_factor,
)
from src.mollifier_profiles import case_c_taylor_coeffs, PolyLike


@dataclass
class UnifiedI1PaperResult:
    """Result of unified I1 evaluation using paper regime."""

    ell1: int
    ell2: int
    I1_value: float

    # Regime information
    omega1: int  # omega for P_ell1 (0 = Case B, >0 = Case C)
    omega2: int  # omega for P_ell2

    # Diagnostics
    n_quad_u: int
    n_quad_t: int
    n_quad_a: int  # Quadrature for Case C a-integral
    include_Q: bool
    factorial_norm_applied: bool


def omega_for_ell(ell: int) -> int:
    """
    Get the omega value for a given piece index.

    omega = ell - 1:
    - P1 (ell=1): omega=0 -> Case B (raw polynomial)
    - P2 (ell=2): omega=1 -> Case C (kernel attenuation)
    - P3 (ell=3): omega=2 -> Case C (kernel attenuation)
    """
    return ell - 1


def build_P_factor_paper(
    poly: PolyLike,
    u: float,
    var: str,
    omega: int,
    R: float,
    theta: float,
    max_dx: int,
    max_dy: int,
    n_quad_a: int = 40,
) -> BivariateSeries:
    """
    Build P-factor using paper regime kernels.

    For omega=0 (Case B): P(u + var) - standard polynomial composition
    For omega>0 (Case C): K_omega(u + var; R, theta) - kernel attenuation

    Args:
        poly: Polynomial object with eval_deriv method
        u: Evaluation point
        var: Variable name ("x" or "y")
        omega: Kernel order (0=Case B, 1,2=Case C)
        R: PRZZ R parameter
        theta: PRZZ theta parameter
        max_dx: Maximum x-degree
        max_dy: Maximum y-degree
        n_quad_a: Quadrature points for Case C a-integral

    Returns:
        BivariateSeries representing P(u+var) or K_omega(u+var)
    """
    if var == "x":
        ax, ay = 1.0, 0.0
        max_order = max_dx
    elif var == "y":
        ax, ay = 0.0, 1.0
        max_order = max_dy
    else:
        raise ValueError(f"var must be 'x' or 'y', got '{var}'")

    if omega == 0:
        # Case B: Standard polynomial composition
        # Get polynomial coefficients and use standard build_P_factor
        poly_coeffs = _extract_poly_coeffs(poly)
        dummy = BivariateSeries.zero(max_dx, max_dy)
        return dummy.compose_polynomial(poly_coeffs, a0=u, ax=ax, ay=ay)
    else:
        # Case C: Use kernel-attenuated Taylor coefficients
        # Get K_omega^{(j)}(u) for j=0..max_order (in factorial basis)
        taylor_coeffs = case_c_taylor_coeffs(
            poly, u, omega, R, theta, max_order, n_quad_a
        )
        # Build series from Taylor coefficients
        # K_omega(u + delta) = sum_{j=0}^{max_order} K_omega^{(j)}(u)/j! * delta^j
        return _build_series_from_taylor(
            taylor_coeffs, var, max_dx, max_dy
        )


def _extract_poly_coeffs(poly: PolyLike) -> List[float]:
    """Extract standard basis polynomial coefficients [c0, c1, c2, ...]."""
    # Check for standard_coeffs attribute
    if hasattr(poly, 'standard_coeffs'):
        return list(poly.standard_coeffs)

    # Check for tilde_coeffs (PRZZ Chebyshev-based)
    if hasattr(poly, 'tilde_coeffs') and poly.tilde_coeffs is not None:
        # Convert via sampling and fitting
        n_coeffs = len(poly.tilde_coeffs) + 2
        u_points = np.linspace(0, 1, n_coeffs + 5)
        y_points = poly.eval(u_points)
        coeffs = np.polyfit(u_points, y_points, len(poly.tilde_coeffs) + 1)
        return list(coeffs[::-1])

    # Fallback: sample and fit
    if hasattr(poly, 'eval'):
        u_points = np.linspace(0, 1, 15)
        y_points = poly.eval(u_points)
        for deg in range(10, 0, -1):
            coeffs = np.polyfit(u_points, y_points, deg)
            fitted = np.polyval(coeffs, u_points)
            if np.max(np.abs(fitted - y_points)) < 1e-10:
                return list(coeffs[::-1])
        coeffs = np.polyfit(u_points, y_points, 5)
        return list(coeffs[::-1])

    raise ValueError(f"Cannot extract coefficients from polynomial: {type(poly)}")


def _build_series_from_taylor(
    taylor_coeffs: np.ndarray,
    var: str,
    max_dx: int,
    max_dy: int,
) -> BivariateSeries:
    """
    Build BivariateSeries from Taylor coefficients in factorial basis.

    taylor_coeffs[j] = f^{(j)}(0) (not divided by j!)

    We want: f(delta) = sum_j f^{(j)}(0)/j! * delta^j

    Args:
        taylor_coeffs: Array of f^{(j)}(0) values (factorial basis)
        var: "x" or "y"
        max_dx, max_dy: Maximum degrees

    Returns:
        BivariateSeries representing the Taylor expansion
    """
    coeffs: Dict = {}

    if var == "x":
        for j, c in enumerate(taylor_coeffs):
            if j > max_dx:
                break
            if c != 0.0:
                # Divide by j! to convert from factorial to standard basis
                coeffs[(j, 0)] = c / math.factorial(j)
    else:  # var == "y"
        for j, c in enumerate(taylor_coeffs):
            if j > max_dy:
                break
            if c != 0.0:
                coeffs[(0, j)] = c / math.factorial(j)

    return BivariateSeries(max_dx=max_dx, max_dy=max_dy, coeffs=coeffs)


def compute_I1_unified_paper(
    R: float,
    theta: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    n_quad_u: int = 60,
    n_quad_t: int = 60,
    n_quad_a: int = 40,
    include_Q: bool = True,
    apply_factorial_norm: bool = True,
) -> UnifiedI1PaperResult:
    """
    Compute I1 for pair (ell1, ell2) using PAPER regime kernels.

    This uses:
    - Case B (raw polynomial) for P1 (omega=0)
    - Case C (kernel attenuation) for P2, P3 (omega=1,2)

    Args:
        R: PRZZ R parameter
        theta: PRZZ theta parameter (typically 4/7)
        ell1: First piece index (1, 2, or 3)
        ell2: Second piece index (1, 2, or 3)
        polynomials: Dict with keys "P1", "P2", "P3", "Q"
        n_quad_u: Quadrature points for u integral
        n_quad_t: Quadrature points for t integral
        n_quad_a: Quadrature points for Case C a-integral
        include_Q: Whether to include Q factors (default True)
        apply_factorial_norm: Whether to multiply by ell1!*ell2! (default True)

    Returns:
        UnifiedI1PaperResult with I1 value
    """
    max_dx = ell1
    max_dy = ell2

    omega1 = omega_for_ell(ell1)
    omega2 = omega_for_ell(ell2)

    u_nodes, u_weights = gauss_legendre_01(n_quad_u)
    t_nodes, t_weights = gauss_legendre_01(n_quad_t)

    # Get polynomial objects
    P_ell1 = polynomials.get(f"P{ell1}")
    P_ell2 = polynomials.get(f"P{ell2}")
    Q = polynomials.get("Q")

    if P_ell1 is None or P_ell2 is None:
        raise ValueError(f"Missing polynomial P{ell1} or P{ell2}")

    # Get Q coefficients for composition
    Q_coeffs = _extract_poly_coeffs(Q) if Q is not None and include_Q else None

    # PRZZ (1-u) power
    one_minus_u_power = ell1 + ell2

    total = 0.0
    for u, u_w in zip(u_nodes, u_weights):
        one_minus_u_factor = (1.0 - u) ** one_minus_u_power

        for t, t_w in zip(t_nodes, t_weights):
            # 1. Exp factor: exp(2Rt + R*theta*(2t-1)*(x+y))
            a0 = 2 * R * t
            a_xy = R * theta * (2 * t - 1)
            exp_factor = build_exp_bracket(a0, a_xy, a_xy, max_dx, max_dy)

            # 2. Log factor: 1/theta + x + y
            log_factor = build_log_factor(theta, max_dx, max_dy)

            # 3. P factors with PAPER REGIME kernels
            P_x = build_P_factor_paper(
                P_ell1, u, "x", omega1, R, theta, max_dx, max_dy, n_quad_a
            )
            P_y = build_P_factor_paper(
                P_ell2, u, "y", omega2, R, theta, max_dx, max_dy, n_quad_a
            )

            # Build bracket
            bracket = exp_factor * log_factor * P_x * P_y

            # 4. Q factors (if enabled)
            if include_Q and Q_coeffs is not None:
                Q_alpha = build_Q_factor(
                    Q_coeffs,
                    a0=t,
                    ax=theta * (t - 1),
                    ay=theta * t,
                    max_dx=max_dx,
                    max_dy=max_dy,
                )
                Q_beta = build_Q_factor(
                    Q_coeffs,
                    a0=t,
                    ax=theta * t,
                    ay=theta * (t - 1),
                    max_dx=max_dx,
                    max_dy=max_dy,
                )
                bracket = bracket * Q_alpha * Q_beta

            # 5. Extract x^ell1 y^ell2 coefficient
            coeff = bracket.extract(ell1, ell2)

            # 6. Add to integral
            total += coeff * one_minus_u_factor * u_w * t_w

    # 7. Apply factorial normalization if requested
    if apply_factorial_norm:
        total *= math.factorial(ell1) * math.factorial(ell2)

    # 8. Apply sign convention for off-diagonal pairs
    if ell1 != ell2:
        sign = (-1) ** (ell1 + ell2)
        total *= sign

    return UnifiedI1PaperResult(
        ell1=ell1,
        ell2=ell2,
        I1_value=total,
        omega1=omega1,
        omega2=omega2,
        n_quad_u=n_quad_u,
        n_quad_t=n_quad_t,
        n_quad_a=n_quad_a,
        include_Q=include_Q,
        factorial_norm_applied=apply_factorial_norm,
    )


# =============================================================================
# VALIDATION HELPERS
# =============================================================================


def compare_raw_vs_paper(
    R: float,
    theta: float,
    ell1: int,
    ell2: int,
    polynomials: Dict,
    n_quad: int = 60,
) -> Dict:
    """
    Compare raw regime (unified_general) vs paper regime for a given pair.

    Returns diagnostic dict with both values and ratio.
    """
    from src.unified_i1_general import compute_I1_unified_general

    raw_result = compute_I1_unified_general(
        R, theta, ell1, ell2, polynomials,
        n_quad_u=n_quad, n_quad_t=n_quad,
        include_Q=True, apply_factorial_norm=True,
    )

    paper_result = compute_I1_unified_paper(
        R, theta, ell1, ell2, polynomials,
        n_quad_u=n_quad, n_quad_t=n_quad, n_quad_a=40,
        include_Q=True, apply_factorial_norm=True,
    )

    raw_val = raw_result.I1_value
    paper_val = paper_result.I1_value

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

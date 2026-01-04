"""
src/composition.py
Polynomial composition with nilpotent perturbations.

Provides the core operation for PRZZ term evaluation:
    P(u + δ) where δ = Σ aᵢxᵢ (nilpotent variables)

Key identity (proved by composition bridge tests):
    Coefficient of x₁x₂...xₖ in P(u+δ) = (∏ᵢ aᵢ) · P⁽ᵏ⁾(u)

This module centralizes composition logic to prevent:
- Factorial errors in Taylor expansion
- Inconsistent var_names between series
- Redundant reimplementation across term types
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Protocol, Tuple, Union, runtime_checkable, Optional

from src.series import TruncatedSeries


@runtime_checkable
class PolyLike(Protocol):
    """Protocol for polynomial-like objects usable in composition.

    Any object with eval_deriv(x, k) -> ndarray is acceptable.
    Objects may optionally have:
    - .degree property (int)
    - .to_monomial() method returning an object with .degree
    """
    def eval_deriv(self, x: np.ndarray, k: int) -> np.ndarray:
        ...


def _get_poly_degree(poly: PolyLike, fallback: int) -> int:
    """Get polynomial degree, with robust fallback for wrapper classes.

    Tries in order:
    1. poly.degree (direct attribute)
    2. poly.to_monomial().degree (for PRZZ wrapper classes)
    3. fallback value (typically n_vars)

    Args:
        poly: Polynomial-like object
        fallback: Value to return if degree cannot be determined

    Returns:
        Polynomial degree, or fallback if unavailable
    """
    # Try direct .degree attribute
    degree = getattr(poly, "degree", None)
    if degree is not None:
        return degree

    # Try .to_monomial().degree for PRZZ wrapper classes
    to_mono = getattr(poly, "to_monomial", None)
    if to_mono is not None:
        mono = to_mono()
        degree = getattr(mono, "degree", None)
        if degree is not None:
            return degree

    # Fallback - safe because eval_deriv returns 0 for k > actual degree
    return fallback


def compose_polynomial_on_affine(
    poly: PolyLike,
    u0: Union[float, np.ndarray],
    lin: Dict[str, Union[float, np.ndarray]],
    var_names: Tuple[str, ...]
) -> TruncatedSeries:
    """
    Compute P(u0 + δ) as a TruncatedSeries.

    This is the core composition operation for PRZZ term evaluation.
    It expands a polynomial at a base point plus a nilpotent perturbation.

    Args:
        poly: Polynomial-like object to compose. Must have eval_deriv(x, k).
              Works with base Polynomial and PRZZ wrappers (P1Polynomial,
              PellPolynomial, QPolynomial).
        u0: Base point - constant term (scalar or array for grid evaluation)
        lin: Linear coefficients mapping var_name -> coefficient
             e.g., {"x": a, "y": b} means δ = a*x + b*y
             Coefficients can be scalars or arrays (must broadcast with u0)
        var_names: Tuple of variable names defining the series context.
                  All keys in `lin` must be present in var_names.

    Returns:
        TruncatedSeries representing P(u0 + δ) with the given var_names

    Raises:
        ValueError: If any key in lin is not in var_names

    Mathematical basis:
        Taylor expansion: P(u0 + δ) = Σₖ P⁽ᵏ⁾(u0)/k! · δᵏ

        Under nilpotent rules (xᵢ² = 0):
        - δ² keeps only cross terms: δ² = 2 Σᵢ<ⱼ aᵢaⱼxᵢxⱼ
        - δᵏ has coefficient k! · (product of k distinct aᵢ) for each k-subset

        Result: coeff(x₁...xₖ) = (∏ aᵢ) · P⁽ᵏ⁾(u0)
        The k! from δᵏ expansion cancels with 1/k! from Taylor.

    Example:
        >>> poly = Polynomial([1.0, 2.0, 3.0])  # 1 + 2x + 3x²
        >>> result = compose_polynomial_on_affine(
        ...     poly, u0=0.5, lin={"x": 2.0, "y": 3.0}, var_names=("x", "y")
        ... )
        >>> result.extract(("x", "y"))  # = 2.0 * 3.0 * P''(0.5) = 6 * 6 = 36
    """
    # Validate that all lin keys are in var_names
    for name in lin.keys():
        if name not in var_names:
            raise ValueError(
                f"Linear coefficient key '{name}' not in var_names {var_names}"
            )

    # Ensure u0 is an array for consistent arithmetic
    u0 = np.asarray(u0)

    # Build δ as a TruncatedSeries (pure nilpotent, no constant term)
    delta = TruncatedSeries.from_scalar(0.0, var_names)
    for name, coeff in lin.items():
        delta = delta + TruncatedSeries.variable(name, var_names) * coeff

    # P(u0 + δ) = Σₖ P⁽ᵏ⁾(u0)/k! · δᵏ
    # Series terminates at k = min(degree, n_active_vars)
    # Use active variable count (len(lin)) not total var_names for efficiency
    n_active = len(lin)
    poly_degree = _get_poly_degree(poly, fallback=n_active)
    max_k = min(poly_degree, n_active)

    result = TruncatedSeries.from_scalar(0.0, var_names)
    delta_power = TruncatedSeries.from_scalar(1.0, var_names)  # δ^0 = 1
    factorial = 1.0

    for k in range(max_k + 1):
        # Get P⁽ᵏ⁾(u0)
        deriv_k = poly.eval_deriv(u0, k)

        # Add P⁽ᵏ⁾(u0)/k! · δᵏ to result
        result = result + delta_power * (deriv_k / factorial)

        # Update for next iteration
        delta_power = delta_power * delta
        factorial *= (k + 1)

        # Early termination if δ^(k+1) is all zeros (fully truncated)
        if all(np.all(c == 0) for c in delta_power.coeffs.values()):
            break

    return result


def compose_exp_on_affine(
    R: float,
    u0: Union[float, np.ndarray],
    lin: Dict[str, Union[float, np.ndarray]],
    var_names: Tuple[str, ...]
) -> TruncatedSeries:
    """
    Compute exp(R * (u0 + δ)) as a TruncatedSeries.

    This is a convenience wrapper for the common PRZZ pattern of
    exponentiating a scaled affine expression.

    Args:
        R: Scaling constant (typically the PRZZ R parameter)
        u0: Base point (scalar or array)
        lin: Linear coefficients {var_name: coeff}
        var_names: Variable names tuple

    Returns:
        TruncatedSeries representing exp(R * (u0 + δ))

    Mathematical basis:
        exp(R*(u0 + δ)) = exp(R*u0) * exp(R*δ)

        Since δ is nilpotent:
        exp(R*δ) = 1 + R*δ + (R*δ)²/2! + ...

        The series terminates naturally.

    Raises:
        ValueError: If any key in lin is not in var_names
    """
    # Validate that all lin keys are in var_names
    for name in lin.keys():
        if name not in var_names:
            raise ValueError(
                f"Linear coefficient key '{name}' not in var_names {var_names}"
            )

    # Build the affine series: u0 + δ
    affine = TruncatedSeries.from_scalar(u0, var_names)
    for name, coeff in lin.items():
        affine = affine + TruncatedSeries.variable(name, var_names) * coeff

    # Scale by R and exponentiate
    scaled = affine * R
    return scaled.exp()


def compose_profile_on_affine(
    taylor_coeffs: np.ndarray,
    lin: Dict[str, Union[float, np.ndarray]],
    var_names: Tuple[str, ...]
) -> TruncatedSeries:
    """
    Compose a profile (given by its Taylor coefficients) with a nilpotent perturbation.

    This is the generalization of compose_polynomial_on_affine that works with
    any profile F(u + δ), not just polynomials. The profile is specified by
    its Taylor coefficients F^{(j)}(u) in the factorial basis.

    For Case B (ω=0): taylor_coeffs[j] = P^{(j)}(u)
    For Case C (ω>0): taylor_coeffs[j] = K_ω^{(j)}(u;R) from auxiliary a-integral

    Mathematical basis:
        F(u + δ) = Σⱼ F^{(j)}(u)/j! · δʲ

        Under nilpotent rules (xᵢ² = 0):
        coeff(x₁...xⱼ) = (∏ aᵢ) · F^{(j)}(u)

        The j! from δʲ expansion cancels with 1/j! from Taylor.

    Args:
        taylor_coeffs: Array of shape (max_order+1,) with coeffs[j] = F^{(j)}(u)
                       in the factorial basis (NOT divided by j!)
        lin: Linear coefficients mapping var_name -> coefficient
             e.g., {"x": a, "y": b} means δ = a*x + b*y
        var_names: Tuple of variable names defining the series context

    Returns:
        TruncatedSeries representing F(u + δ)

    Raises:
        ValueError: If any key in lin is not in var_names
    """
    from math import factorial

    # Validate that all lin keys are in var_names
    for name in lin.keys():
        if name not in var_names:
            raise ValueError(
                f"Linear coefficient key '{name}' not in var_names {var_names}"
            )

    # Build δ as a TruncatedSeries (pure nilpotent, no constant term)
    delta = TruncatedSeries.from_scalar(0.0, var_names)
    for name, coeff in lin.items():
        delta = delta + TruncatedSeries.variable(name, var_names) * coeff

    # F(u + δ) = Σⱼ F^{(j)}(u)/j! · δʲ
    # Series terminates at j = min(len(taylor_coeffs)-1, n_active_vars)
    n_active = len(lin)
    max_j = min(len(taylor_coeffs) - 1, n_active)

    result = TruncatedSeries.from_scalar(0.0, var_names)
    delta_power = TruncatedSeries.from_scalar(1.0, var_names)  # δ^0 = 1

    for j in range(max_j + 1):
        # Get F^{(j)}(u) from the pre-computed Taylor coefficients
        deriv_j = taylor_coeffs[j]

        # Add F^{(j)}(u)/j! · δʲ to result
        result = result + delta_power * (deriv_j / factorial(j))

        # Update for next iteration
        delta_power = delta_power * delta

        # Early termination if δ^(j+1) is all zeros (fully truncated)
        if all(np.all(c == 0) for c in delta_power.coeffs.values()):
            break

    return result


def compose_profile_on_affine_grid(
    taylor_coeffs_grid: np.ndarray,
    lin: Dict[str, Union[float, np.ndarray]],
    var_names: Tuple[str, ...]
) -> TruncatedSeries:
    """
    Compose a profile with Taylor coefficients on a grid.

    This is the grid-aware version where taylor_coeffs_grid has shape
    (n_grid_u, n_grid_t, max_order+1) or similar.

    For each grid point (i, j), taylor_coeffs_grid[i, j, k] = F^{(k)}(u_ij).

    Args:
        taylor_coeffs_grid: Array with last dimension = max_order+1
                            Each [..., k] slice is F^{(k)} on the grid
        lin: Linear coefficients {var_name: coeff_array}
        var_names: Variable names tuple

    Returns:
        TruncatedSeries with coefficients on the grid
    """
    from math import factorial

    # Validate
    for name in lin.keys():
        if name not in var_names:
            raise ValueError(
                f"Linear coefficient key '{name}' not in var_names {var_names}"
            )

    # Build δ
    delta = TruncatedSeries.from_scalar(0.0, var_names)
    for name, coeff in lin.items():
        delta = delta + TruncatedSeries.variable(name, var_names) * coeff

    # Get max order from last dimension
    max_order = taylor_coeffs_grid.shape[-1] - 1
    n_active = len(lin)
    max_j = min(max_order, n_active)

    result = TruncatedSeries.from_scalar(0.0, var_names)
    delta_power = TruncatedSeries.from_scalar(1.0, var_names)

    for j in range(max_j + 1):
        # Get F^{(j)}(u) grid from the last dimension
        deriv_j = taylor_coeffs_grid[..., j]

        # Add F^{(j)}(u)/j! · δʲ
        result = result + delta_power * (deriv_j / factorial(j))

        # Update for next iteration
        delta_power = delta_power * delta

        if all(np.all(c == 0) for c in delta_power.coeffs.values()):
            break

    return result

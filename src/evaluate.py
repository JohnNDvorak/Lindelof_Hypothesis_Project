"""
src/evaluate.py
Evaluation pipeline for computing c from PRZZ terms.

This module implements the staged evaluation pipeline:
1. Build quadrature grid
2. Create series context from term
3. Multiply all formal-variable-dependent factors (poly_factors, exp_factors)
4. If algebraic_prefactor exists, multiply it in
5. Extract derivative coefficient via term.deriv_tuple()
6. Multiply by grid-only prefactors (numeric_prefactor, poly_prefactors)
7. Integrate with quadrature weights
8. Sum contributions from all terms

Key design principles:
- Always return per-term breakdown for debugging
- Shape coercion: extracted coefficients coerced to grid shape before integration
- Staged pipeline prevents silent errors
- Reference-based validation tests

PRZZ Reference: Section 6.2.1
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np

from src.term_dsl import Term, SeriesContext, PolyFactor, ExpFactor
from src.series import TruncatedSeries
from src.quadrature import tensor_grid_2d
from src.composition import PolyLike
from src.kernel_registry import KernelRegime


# =============================================================================
# RESULT TYPES AND SPEC LOCKS - Extracted to src/evaluator/result_types.py
# =============================================================================
#
# The following types are imported from src.evaluator for backwards compatibility:
# - TermResult: Result of evaluating a single term
# - EvaluationResult: Result of evaluating multiple terms
# - S34OrderedPairsError: Raised when using wrong pair convention
# - I34MirrorForbiddenError: Raised when applying mirror to I3/I4
#
# Helper functions (also in src.evaluator):
# - get_s34_triangle_pairs(): S34 triangle x 2 convention
# - get_s34_factorial_normalization(): Factorial normalization factors
# - assert_s34_triangle_convention(): Guard against ordered pairs
# - assert_i34_no_mirror(): Guard against I3/I4 mirror
#
# SPEC LOCK: I3/I4 Mirror is FORBIDDEN
# Per TRUTH_SPEC.md Section 10: I1/I2 have mirror, I3/I4 do NOT
#
# SPEC LOCK: S34 uses TRIANGLE x 2 convention, NOT 9 ordered pairs
# Using 9 ordered pairs causes +11% OVERSHOOT.
# Reference: TRUTH_SPEC.md Section 13
# =============================================================================

from src.evaluator.result_types import (
    TermResult,
    EvaluationResult,
    S34OrderedPairsError,
    I34MirrorForbiddenError,
    get_s34_triangle_pairs,
    get_s34_factorial_normalization,
    assert_s34_triangle_convention as _assert_s34_triangle_convention,
    assert_i34_no_mirror as _assert_i34_no_mirror,
)


def _coerce_to_grid_shape(
    coeff: Any,
    W: np.ndarray,
    term_name: str
) -> np.ndarray:
    """
    Coerce extracted coefficient to grid shape.

    This prevents silent broadcasting errors and makes shape mismatches
    fail immediately rather than producing wrong results.

    Args:
        coeff: Extracted coefficient (may be scalar or array)
        W: Weight grid (defines expected shape)
        term_name: For error messages

    Returns:
        Array with shape W.shape

    Raises:
        ValueError: If coeff has wrong non-scalar shape
    """
    arr = np.asarray(coeff)

    # Scalar -> broadcast to grid shape
    if arr.shape == ():
        return np.full_like(W, arr)

    # Already grid-shaped
    if arr.shape == W.shape:
        return arr

    # Wrong shape - fail immediately
    raise ValueError(
        f"Term '{term_name}': extracted coefficient has shape {arr.shape}, "
        f"expected scalar or {W.shape}"
    )


def evaluate_term(
    term: Term,
    polynomials: Dict[str, PolyLike],
    n: int,
    return_debug: bool = False,
    R: Optional[float] = None,
    theta: Optional[float] = None,
    n_quad_a: int = 40,
) -> TermResult:
    """
    Evaluate a single term's contribution to c.

    Implements the staged evaluation pipeline:
    1. Build quadrature grid
    2. Create series context
    3. Multiply all series factors (poly + exp)
    4. Multiply by algebraic prefactor if present
    5. Extract coefficient via deriv_tuple()
    6. Multiply by numeric_prefactor and poly_prefactors
    7. Integrate

    Args:
        term: Term to evaluate
        polynomials: Dict mapping poly_name to polynomial object
            e.g., {"P1": p1_poly, "Q": q_poly}
        n: Number of quadrature points per dimension
        return_debug: If True, include debug info in result
        R: R parameter (for Case C omega handling)
        theta: θ parameter (for Case C omega handling)
        n_quad_a: Quadrature points for Case C a-integral

    Returns:
        TermResult with the term's contribution
    """
    # Stage 1: Build quadrature grid
    U, T, W = tensor_grid_2d(n)

    # Stage 2: Create series context
    ctx = term.create_context()

    # Stage 3: Build the series integrand from formal-var-dependent factors
    # Start with 1 (multiplicative identity)
    if len(term.vars) > 0:
        integrand = ctx.scalar_series(np.ones_like(U))
    else:
        # For terms with no formal variables, series is just scalar
        integrand = ctx.scalar_series(np.ones_like(U))

    # Multiply in all poly factors
    for factor in term.poly_factors:
        poly = polynomials.get(factor.poly_name)
        if poly is None:
            raise ValueError(
                f"Term '{term.name}': polynomial '{factor.poly_name}' not found"
            )
        # Pass R and theta for Case C omega handling
        factor_series = factor.evaluate(poly, U, T, ctx, R=R, theta=theta, n_quad_a=n_quad_a)
        integrand = integrand * factor_series

    # Multiply in all exp factors
    for factor in term.exp_factors:
        factor_series = factor.evaluate(U, T, ctx)
        integrand = integrand * factor_series

    # Stage 4: Multiply by algebraic prefactor if present
    if term.algebraic_prefactor is not None:
        prefactor_series = term.algebraic_prefactor.to_series(U, T, ctx)
        integrand = integrand * prefactor_series

    # Stage 5: Extract the derivative coefficient
    deriv_vars = term.deriv_tuple()
    coeff = integrand.extract(deriv_vars)

    # Coerce to grid shape (catches scalar 0 for missing terms)
    coeff = _coerce_to_grid_shape(coeff, W, term.name)

    # Stage 6: Multiply by grid-only prefactors
    # First: numeric prefactor (scalar)
    coeff = coeff * term.numeric_prefactor

    # Then: poly_prefactors (grid functions)
    for prefactor_func in term.poly_prefactors:
        prefactor_vals = prefactor_func(U, T)
        coeff = coeff * prefactor_vals

    # Stage 7: Integrate
    value = float(np.sum(W * coeff))

    # Build result
    result = TermResult(name=term.name, value=value)

    if return_debug:
        # Sample coefficient at grid center for debugging
        mid = n // 2
        result.extracted_coeff_sample = float(coeff[mid, mid])
        result.series_term_count = len(integrand.coeffs)

    return result


def evaluate_terms(
    terms: List[Term],
    polynomials: Dict[str, PolyLike],
    n: int,
    return_breakdown: bool = True,
    R: Optional[float] = None,
    theta: Optional[float] = None,
    n_quad_a: int = 40,
) -> EvaluationResult:
    """
    Evaluate multiple terms and sum their contributions.

    Args:
        terms: List of Term objects
        polynomials: Dict mapping poly_name to polynomial object
        n: Number of quadrature points per dimension
        return_breakdown: If True, include per-term breakdown
        R: R parameter (for Case C omega handling)
        theta: θ parameter (for Case C omega handling)
        n_quad_a: Quadrature points for Case C a-integral

    Returns:
        EvaluationResult with total and optional breakdown
    """
    term_results = []
    per_term = {}
    total = 0.0

    for term in terms:
        result = evaluate_term(
            term,
            polynomials,
            n,
            return_debug=return_breakdown,
            R=R,
            theta=theta,
            n_quad_a=n_quad_a,
        )
        term_results.append(result)
        per_term[term.name] = result.value
        total += result.value

    return EvaluationResult(
        total=total,
        per_term=per_term,
        n=n,
        term_results=term_results if return_breakdown else None
    )


def evaluate_c11(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict[str, PolyLike],
    return_breakdown: bool = True,
    n_quad_a: int = 40,
) -> EvaluationResult:
    """
    Evaluate c₁₁ from all (1,1) terms.

    Convenience function that builds all (1,1) terms and evaluates them.

    Args:
        theta: θ parameter (typically 4/7)
        R: R parameter (typically 1.3036)
        n: Number of quadrature points per dimension
        polynomials: Dict with keys "P1", "Q" mapping to polynomial objects
        return_breakdown: If True, include per-term breakdown
        n_quad_a: Quadrature points for Case C a-integral

    Returns:
        EvaluationResult with c₁₁ and optional breakdown
    """
    from src.terms_k3_d1 import make_all_terms_11

    terms = make_all_terms_11(theta, R)
    return evaluate_terms(
        terms,
        polynomials,
        n,
        return_breakdown,
        R=R,
        theta=theta,
        n_quad_a=n_quad_a,
    )


def convergence_sweep(
    theta: float,
    R: float,
    polynomials: Dict[str, PolyLike],
    ns: List[int] = None,
    n_ref: int = 200
) -> Dict[str, Any]:
    """
    Run convergence sweep at multiple quadrature resolutions.

    Args:
        theta: θ parameter
        R: R parameter
        polynomials: Dict with polynomial objects
        ns: List of n values to test (default [40, 60, 80, 100])
        n_ref: Reference n for computing errors

    Returns:
        Dict with:
        - 'reference': c₁₁ at n_ref
        - 'values': {n: c₁₁} for each n
        - 'errors': {n: |c₁₁(n) - reference|} for each n
        - 'per_term_ref': per-term breakdown at n_ref
    """
    if ns is None:
        ns = [40, 60, 80, 100]

    # Compute reference
    ref_result = evaluate_c11(theta, R, n_ref, polynomials, return_breakdown=True)
    reference = ref_result.total

    # Compute at each n
    values = {}
    errors = {}
    for n in ns:
        result = evaluate_c11(theta, R, n, polynomials, return_breakdown=False)
        values[n] = result.total
        errors[n] = abs(result.total - reference)

    return {
        'reference': reference,
        'n_ref': n_ref,
        'values': values,
        'errors': errors,
        'per_term_ref': ref_result.per_term
    }


# =============================================================================
# Specialized evaluation for analytic tests
# =============================================================================

def evaluate_I1_with_P1_Q1(
    theta: float,
    R: float,
    n: int
) -> float:
    """
    Evaluate I₁ with P≡1, Q≡1 for analytic comparison.

    When P=Q=1, all polynomial factors become 1, and the only
    formal-variable dependence comes from:
    - algebraic prefactor: (1/θ + x + y)
    - exp factors: exp(R·Arg_α) * exp(R·Arg_β)

    The xy coefficient can be computed analytically (see test).

    Args:
        theta: θ parameter
        R: R parameter
        n: Number of quadrature points

    Returns:
        I₁ contribution with P=Q=1
    """
    from src.terms_k3_d1 import make_I1_11
    from src.polynomials import Polynomial

    # Create constant polynomial P(x) = 1
    one_poly = Polynomial([1.0])

    term = make_I1_11(theta, R)
    polynomials = {"P1": one_poly, "Q": one_poly}

    result = evaluate_term(term, polynomials, n)
    return result.value


def evaluate_c_full(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict[str, PolyLike],
    return_breakdown: bool = True,
    use_factorial_normalization: bool = True,
    mode: str = "main",
    kernel_regime: KernelRegime = "raw",
    n_quad_a: int = 40,
    terms_version: str = "old",  # Run 10A: "old" or "v2"
) -> EvaluationResult:
    """
    Evaluate full c from all K=3 pair contributions.

    c = Σ_{ℓ₁ ≤ ℓ₂} multiplier(ℓ₁, ℓ₂) × c_{ℓ₁,ℓ₂}

    where:
    - Off-diagonal pairs (ℓ₁ < ℓ₂) have symmetry factor 2
    - If use_factorial_normalization=True, each pair has factor 1/(ℓ₁! × ℓ₂!)
      from the bracket combinatorics (residue extraction normalization)

    MODES:
    - "main": PRZZ asymptotic main term constant (I₅ FORBIDDEN)
              This is what PRZZ uses to compute published κ.
              PRZZ TeX lines 1626-1628: I₅ ≪ T/L (error term)
    - "with_error_terms": Includes I₅ as diagnostic. PRINTS WARNING.
              Do NOT use this mode for golden target matching.

    KERNEL REGIMES:
    - "paper": TeX-driven regime where ω drives Case B/C selection (for d=1:
              ω = ℓ - 1) and P₂/P₃ are evaluated via Case C kernels.
              This is the project’s “paper-truth” evaluation target.
    - "raw": Legacy/diagnostic regime where all P-factors are evaluated as raw
              polynomials P(u ± x) (no Case C transform). Kept for regression
              and debugging; not expected to match PRZZ κ/κ* benchmarks.

    Args:
        theta: θ parameter (typically 4/7)
        R: R parameter (typically 1.3036)
        n: Number of quadrature points per dimension
        polynomials: Dict with keys "P1", "P2", "P3", "Q"
        return_breakdown: If True, include per-term breakdown
        use_factorial_normalization: If True, apply 1/(ℓ₁!×ℓ₂!) normalization
        mode: "main" (default) or "with_error_terms"
        kernel_regime: "raw" (default) or "paper" for Case C kernel handling
        n_quad_a: Quadrature points for Case C a-integral

    Returns:
        EvaluationResult with full c and optional breakdown

    Raises:
        ValueError: If mode is not "main" or "with_error_terms"
    """
    # Validate mode
    if mode not in ("main", "with_error_terms"):
        raise ValueError(f"mode must be 'main' or 'with_error_terms', got '{mode}'")
    # Run 10A: Dispatch based on terms_version
    if terms_version == "v2":
        from src.terms_k3_d1 import make_all_terms_k3_v2 as make_all_terms_k3
    else:
        from src.terms_k3_d1 import make_all_terms_k3
    import math

    all_terms = make_all_terms_k3(theta, R, kernel_regime=kernel_regime)

    # Normalization factors: 1/(ℓ₁! × ℓ₂!)
    # This comes from the bracket combinatorics for residue extraction
    factorial_norm = {
        "11": 1.0 / (math.factorial(1) * math.factorial(1)),  # 1/1 = 1
        "22": 1.0 / (math.factorial(2) * math.factorial(2)),  # 1/4
        "33": 1.0 / (math.factorial(3) * math.factorial(3)),  # 1/36
        "12": 1.0 / (math.factorial(1) * math.factorial(2)),  # 1/2
        "13": 1.0 / (math.factorial(1) * math.factorial(3)),  # 1/6
        "23": 1.0 / (math.factorial(2) * math.factorial(3)),  # 1/12
    }

    # Evaluate each pair
    pair_results = {}
    total = 0.0
    all_per_term = {}

    for pair_key, terms in all_terms.items():
        pair_result = evaluate_terms(
            terms,
            polynomials,
            n,
            return_breakdown,
            R=R,
            theta=theta,
            n_quad_a=n_quad_a,
        )
        pair_results[pair_key] = pair_result.total

        # Off-diagonal pairs get factor of 2 for symmetry
        if pair_key in ("12", "13", "23"):
            symmetry_factor = 2.0
        else:
            symmetry_factor = 1.0

        # Apply factorial normalization if requested
        if use_factorial_normalization:
            norm = factorial_norm[pair_key]
        else:
            norm = 1.0

        total += symmetry_factor * norm * pair_result.total

        # Add per-term breakdown with pair prefix
        if return_breakdown:
            for term_name, term_val in pair_result.per_term.items():
                all_per_term[term_name] = term_val

    # I₅ arithmetic correction (ERROR TERM - diagnostic only)
    # PRZZ TeX lines 1626-1628: I₅ ≪ T/L (lower order than main term)
    #
    # WARNING: I₅ is NOT part of the asymptotic main constant that feeds κ.
    # Using it to "hit κ" means we're not computing PRZZ's object.
    #
    # Formula derived empirically: I₅ ≈ -S(0) × θ²/12 × I₂_total
    # This is included ONLY for diagnostic purposes in mode="with_error_terms".
    if mode == "with_error_terms":
        import warnings
        warnings.warn(
            "\n" + "=" * 60 + "\n"
            "WARNING: mode='with_error_terms' includes I₅.\n"
            "I₅ is O(T/L), NOT part of PRZZ's main constant.\n"
            "PRZZ TeX lines 1626-1628: 'I₅ ≪ T/L'\n"
            "Do NOT use this mode for golden target matching!\n"
            + "=" * 60,
            UserWarning,
            stacklevel=2
        )

        from src.arithmetic_constants import S_AT_ZERO

        # Compute I₂ sum (already computed as part of main evaluation)
        i2_total = 0.0
        for pair_key, terms in all_terms.items():
            # I₂ is the second term (index 1) in each pair's term list
            i2_term = terms[1]
            i2_result = evaluate_term(i2_term, polynomials, n, R=R, theta=theta, n_quad_a=n_quad_a)

            # Apply normalization
            if pair_key in ("12", "13", "23"):
                sym_factor = 2.0
            else:
                sym_factor = 1.0

            if use_factorial_normalization:
                norm = factorial_norm[pair_key]
            else:
                norm = 1.0

            i2_total += sym_factor * norm * i2_result.value

        # I₅ = -S(0) × θ²/12 × I₂_total
        i5_correction = -S_AT_ZERO * (theta ** 2 / 12.0) * i2_total
        total += i5_correction

        if return_breakdown:
            all_per_term["_I5_total"] = i5_correction
            all_per_term["_I2_sum_for_I5"] = i2_total
            all_per_term["_mode"] = "with_error_terms (I₅ INCLUDED - DIAGNOSTIC ONLY)"
    elif return_breakdown:
        all_per_term["_mode"] = "main (I₅ excluded per PRZZ)"

    # Add pair-level contributions to breakdown
    if return_breakdown:
        all_per_term["_c11_raw"] = pair_results["11"]
        all_per_term["_c22_raw"] = pair_results["22"]
        all_per_term["_c33_raw"] = pair_results["33"]
        all_per_term["_c12_raw"] = pair_results["12"]
        all_per_term["_c13_raw"] = pair_results["13"]
        all_per_term["_c23_raw"] = pair_results["23"]

        if use_factorial_normalization:
            all_per_term["_c11_norm"] = factorial_norm["11"] * pair_results["11"]
            all_per_term["_c22_norm"] = factorial_norm["22"] * pair_results["22"]
            all_per_term["_c33_norm"] = factorial_norm["33"] * pair_results["33"]
            all_per_term["_c12_norm"] = 2 * factorial_norm["12"] * pair_results["12"]
            all_per_term["_c13_norm"] = 2 * factorial_norm["13"] * pair_results["13"]
            all_per_term["_c23_norm"] = 2 * factorial_norm["23"] * pair_results["23"]

    return EvaluationResult(
        total=total,
        per_term=all_per_term,
        n=n,
        term_results=None  # Too many terms for detailed results
    )


def compute_kappa(c: float, R: float) -> float:
    """
    Compute κ from c using the Levinson bound.

    κ = 1 - log(c)/R

    Args:
        c: Main term constant
        R: R parameter

    Returns:
        κ value
    """
    import math
    return 1.0 - math.log(c) / R


def compute_c_paper(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict[str, PolyLike],
    *,
    pair_mode: str = "ordered",
    return_breakdown: bool = True,
    use_factorial_normalization: bool = True,
    mode: str = "main",
    n_quad_a: int = 40,
) -> EvaluationResult:
    """Paper-truth entrypoint for the DSL evaluator.

    Pair assembly modes:
    - pair_mode="ordered": sum all 9 ordered pairs with factorial norm only
      (NO symmetry factor). This is the safest default.
    - pair_mode="hybrid": use triangle×2 only for S12=(I1+I2) (empirically swap-symmetric)
      and ordered-sum for S34=(I3+I4) (empirically NOT swap-symmetric).
    - pair_mode="triangle": legacy upper-triangle with symmetry factor 2 on all
      off-diagonals. This is NOT valid for S34 under the current DSL object.
    """
    if pair_mode == "triangle":
        import warnings

        warnings.warn(
            "compute_c_paper(pair_mode='triangle') uses upper-triangle ×2 folding in the paper regime. "
            "This is diagnostic/legacy only and is not PRZZ-faithful once I₃/I₄ are present; "
            "prefer pair_mode='ordered' (default) or 'hybrid'.",
            UserWarning,
            stacklevel=2,
        )

        result = evaluate_c_full(
            theta=theta,
            R=R,
            n=n,
            polynomials=polynomials,
            return_breakdown=return_breakdown,
            use_factorial_normalization=use_factorial_normalization,
            mode=mode,
            kernel_regime="paper",
            n_quad_a=n_quad_a,
        )
        if return_breakdown and result.per_term is not None:
            result.per_term["_pair_mode"] = "triangle"
        return result

    if pair_mode == "ordered":
        return evaluate_c_ordered(
            theta=theta,
            R=R,
            n=n,
            polynomials=polynomials,
            return_breakdown=return_breakdown,
            use_factorial_normalization=use_factorial_normalization,
            kernel_regime="paper",
            mode=mode,
            n_quad_a=n_quad_a,
        )

    if pair_mode == "hybrid":
        return evaluate_c_hybrid(
            theta=theta,
            R=R,
            n=n,
            polynomials=polynomials,
            return_breakdown=return_breakdown,
            use_factorial_normalization=use_factorial_normalization,
            kernel_regime="paper",
            mode=mode,
            n_quad_a=n_quad_a,
        )

    raise ValueError(f"pair_mode must be 'ordered', 'hybrid', or 'triangle', got '{pair_mode}'")


def evaluate_c_hybrid(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict[str, PolyLike],
    *,
    return_breakdown: bool = True,
    use_factorial_normalization: bool = True,
    kernel_regime: KernelRegime = "paper",
    mode: str = "main",
    n_quad_a: int = 40,
    terms_version: str = "old",  # Run 10A: "old" or "v2"
) -> EvaluationResult:
    """Evaluate c using a hybrid pair assembly.

    - S12 = (I1+I2): upper-triangle pairs with symmetry factor 2 (swap-symmetric).
    - S34 = (I3+I4): all 9 ordered pairs (NOT swap-symmetric under the current DSL object).
    """
    if mode not in ("main", "with_error_terms"):
        raise ValueError(f"mode must be 'main' or 'with_error_terms', got '{mode}'")

    # Run 10A: Dispatch based on terms_version
    if terms_version == "v2":
        from src.terms_k3_d1 import make_all_terms_k3_v2 as make_all_terms_k3
        from src.terms_k3_d1 import make_all_terms_k3_ordered_v2 as make_all_terms_k3_ordered
    else:
        from src.terms_k3_d1 import make_all_terms_k3, make_all_terms_k3_ordered
    import math

    triangle_terms = make_all_terms_k3(theta, R, kernel_regime=kernel_regime)
    ordered_terms = make_all_terms_k3_ordered(theta, R, kernel_regime=kernel_regime)

    f_triangle = {
        "11": 1.0 / (math.factorial(1) * math.factorial(1)),  # 1
        "22": 1.0 / (math.factorial(2) * math.factorial(2)),  # 1/4
        "33": 1.0 / (math.factorial(3) * math.factorial(3)),  # 1/36
        "12": 1.0 / (math.factorial(1) * math.factorial(2)),  # 1/2
        "13": 1.0 / (math.factorial(1) * math.factorial(3)),  # 1/6
        "23": 1.0 / (math.factorial(2) * math.factorial(3)),  # 1/12
    }
    f_ordered = {
        "11": 1.0 / (math.factorial(1) * math.factorial(1)),
        "22": 1.0 / (math.factorial(2) * math.factorial(2)),
        "33": 1.0 / (math.factorial(3) * math.factorial(3)),
        "12": 1.0 / (math.factorial(1) * math.factorial(2)),
        "21": 1.0 / (math.factorial(2) * math.factorial(1)),
        "13": 1.0 / (math.factorial(1) * math.factorial(3)),
        "31": 1.0 / (math.factorial(3) * math.factorial(1)),
        "23": 1.0 / (math.factorial(2) * math.factorial(3)),
        "32": 1.0 / (math.factorial(3) * math.factorial(2)),
    }
    sym_triangle = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}

    total = 0.0
    s12_total = 0.0
    s34_total = 0.0
    i2_total = 0.0
    per_term: Dict[str, float] = {}

    # S12 = I1 + I2 via triangle×2
    for pair_key in ["11", "22", "33", "12", "13", "23"]:
        norm = f_triangle[pair_key] if use_factorial_normalization else 1.0
        w = sym_triangle[pair_key] * norm

        terms = triangle_terms[pair_key]
        for idx in [0, 1]:
            term = terms[idx]
            val = evaluate_term(term, polynomials, n, R=R, theta=theta, n_quad_a=n_quad_a).value
            total += w * val
            s12_total += w * val
            if idx == 1:
                i2_total += w * val
            if return_breakdown:
                per_term[term.name] = val

    # S34 = I3 + I4 via ordered-sum (no symmetry factor)
    for pair_key in ["11", "22", "33", "12", "21", "13", "31", "23", "32"]:
        norm = f_ordered[pair_key] if use_factorial_normalization else 1.0
        terms = ordered_terms[pair_key]
        for idx in [2, 3]:
            term = terms[idx]
            val = evaluate_term(term, polynomials, n, R=R, theta=theta, n_quad_a=n_quad_a).value
            total += norm * val
            s34_total += norm * val
            if return_breakdown:
                per_term[f"{term.name}_ordered"] = val

    if mode == "with_error_terms":
        import warnings
        from src.arithmetic_constants import S_AT_ZERO

        warnings.warn(
            "\n" + "=" * 60 + "\n"
            "WARNING: mode='with_error_terms' includes I₅.\n"
            "I₅ is O(T/L), NOT part of PRZZ's main constant.\n"
            "PRZZ TeX lines 1626-1628: 'I₅ ≪ T/L'\n"
            "Do NOT use this mode for golden target matching!\n"
            + "=" * 60,
            UserWarning,
            stacklevel=2,
        )

        i5_correction = -S_AT_ZERO * (theta ** 2 / 12.0) * i2_total
        total += i5_correction
        if return_breakdown:
            per_term["_I5_total"] = i5_correction
            per_term["_I2_sum_for_I5"] = i2_total

    if return_breakdown:
        per_term["_S12_total"] = s12_total
        per_term["_S34_ordered_total"] = s34_total
        per_term["_pair_mode"] = "hybrid"
        per_term["_kernel_regime"] = kernel_regime
        per_term["_use_factorial_normalization"] = float(bool(use_factorial_normalization))

    return EvaluationResult(total=total, per_term=per_term, n=n, term_results=None)


def compute_c_paper_mirror_experiment(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict[str, PolyLike],
    *,
    use_factorial_normalization: bool = True,
    mode: str = "main",
    n_quad_a: int = 40,
    mirror_multiplier: Optional[float] = None,
) -> Dict[str, float]:
    """Diagnostic helper for testing a naive “mirror recombination” hypothesis.

    This computes the paper-regime constant at +R and -R and returns the
    recombination:

        c_recombined = c(+R) + mirror_multiplier * c(-R)

    By default, `mirror_multiplier = exp(2R)` (common in PRZZ-style mirror terms),
    but callers can override it.
    """

    import math

    if mirror_multiplier is None:
        mirror_multiplier = math.exp(2.0 * R)

    c_plus = evaluate_c_full(
        theta=theta,
        R=R,
        n=n,
        polynomials=polynomials,
        return_breakdown=False,
        use_factorial_normalization=use_factorial_normalization,
        mode=mode,
        kernel_regime="paper",
        n_quad_a=n_quad_a,
    ).total

    c_minus = evaluate_c_full(
        theta=theta,
        R=-R,
        n=n,
        polynomials=polynomials,
        return_breakdown=False,
        use_factorial_normalization=use_factorial_normalization,
        mode=mode,
        kernel_regime="paper",
        n_quad_a=n_quad_a,
    ).total

    return {
        "c_plus": float(c_plus),
        "c_minus": float(c_minus),
        "mirror_multiplier": float(mirror_multiplier),
        "c_recombined": float(c_plus + mirror_multiplier * c_minus),
    }


def evaluate_c_full_with_exp_transform(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict[str, PolyLike],
    *,
    kernel_regime: KernelRegime = "paper",
    exp_scale_multiplier: float = 1.0,
    exp_t_flip: bool = False,
    q_a0_shift: float = 0.0,
    return_breakdown: bool = True,
    use_factorial_normalization: bool = True,
    mode: str = "main",
    n_quad_a: int = 40,
) -> EvaluationResult:
    """Evaluate c with an exp-factor-only transform (diagnostic).

    This exists to support controlled “mirror” experiments without globally
    flipping R, which would also flip Case C kernel internal exponents.

    The transform applies to ExpFactor objects:
    - scales multiplied by `exp_scale_multiplier` (e.g. -1 for sign flip)
    - optionally T -> 1-T inside ExpFactor arguments (`exp_t_flip=True`)

    Optionally, it can also shift Q(...) PolyFactor arguments by a constant:
    - `q_a0_shift` shifts the AffineExpr constant term `a0(u,t)` for Q factors.

    All non-Q PolyFactors (including Case C kernels) still see the original `R`.
    """

    if mode not in ("main", "with_error_terms"):
        raise ValueError(f"mode must be 'main' or 'with_error_terms', got '{mode}'")

    from src.terms_k3_d1 import make_all_terms_k3
    from src.mirror_transform import transform_terms_exp_factors, transform_terms_q_factors
    import math

    t_map = (lambda T: 1.0 - T) if exp_t_flip else None

    all_terms = make_all_terms_k3(theta, R, kernel_regime=kernel_regime)

    factorial_norm = {
        "11": 1.0 / (math.factorial(1) * math.factorial(1)),  # 1/1 = 1
        "22": 1.0 / (math.factorial(2) * math.factorial(2)),  # 1/4
        "33": 1.0 / (math.factorial(3) * math.factorial(3)),  # 1/36
        "12": 1.0 / (math.factorial(1) * math.factorial(2)),  # 1/2
        "13": 1.0 / (math.factorial(1) * math.factorial(3)),  # 1/6
        "23": 1.0 / (math.factorial(2) * math.factorial(3)),  # 1/12
    }

    pair_results = {}
    total = 0.0
    all_per_term: Dict[str, float] = {}

    for pair_key, terms in all_terms.items():
        terms_x = transform_terms_exp_factors(
            terms, scale_multiplier=exp_scale_multiplier, t_map=t_map
        )
        if q_a0_shift != 0.0:
            terms_x = transform_terms_q_factors(terms_x, q_a0_shift=q_a0_shift)

        pair_result = evaluate_terms(
            terms_x,
            polynomials,
            n,
            return_breakdown,
            R=R,
            theta=theta,
            n_quad_a=n_quad_a,
        )
        pair_results[pair_key] = pair_result.total

        symmetry_factor = 2.0 if pair_key in ("12", "13", "23") else 1.0
        norm = factorial_norm[pair_key] if use_factorial_normalization else 1.0
        total += symmetry_factor * norm * pair_result.total

        if return_breakdown:
            for term_name, term_val in pair_result.per_term.items():
                all_per_term[term_name] = term_val

    if return_breakdown:
        all_per_term["_c11_raw"] = pair_results["11"]
        all_per_term["_c22_raw"] = pair_results["22"]
        all_per_term["_c33_raw"] = pair_results["33"]
        all_per_term["_c12_raw"] = pair_results["12"]
        all_per_term["_c13_raw"] = pair_results["13"]
        all_per_term["_c23_raw"] = pair_results["23"]
        all_per_term["_kernel_regime"] = kernel_regime
        all_per_term["_exp_scale_multiplier"] = float(exp_scale_multiplier)
        all_per_term["_exp_t_flip"] = float(bool(exp_t_flip))
        all_per_term["_q_a0_shift"] = float(q_a0_shift)

    return EvaluationResult(total=total, per_term=all_per_term, n=n, term_results=None)


def compute_c_paper_with_mirror(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict[str, PolyLike],
    *,
    pair_mode: str = "hybrid",
    use_factorial_normalization: bool = True,
    mode: str = "main",
    n_quad_a: int = 40,
    K: int = 3,
    mirror_mode: str = "empirical_scalar",
    normalize_scalar_baseline: bool = True,
    normalization_mode: str = "auto",
    allow_diagnostic_correction: bool = False,
) -> EvaluationResult:
    """
    Compute c using paper regime with correct mirror term assembly.

    This implements the PRZZ mirror term structure from TRUTH_SPEC.md Section 10:
    - I₁ and I₂ require mirror: I(α,β) + m·I(-β,-α)
    - I₃ and I₄ do NOT require mirror

    Mirror multiplier (empirically validated within tested regime):
        m = exp(R) + (2K - 1)

    For K=3: m = exp(R) + 5.

    VALIDATION SCOPE (Phase 58, 2025-12-29):
    - Tested regime: R ∈ {1.1167, 1.3036}, K=3, PRZZ polynomial class
    - Baseline: Reproduces PRZZ κ = 0.4173 (m_needed = 8.814)
    - Transferability: Stable ratio ~1.015 across polynomial sets (0.12% drift)
    - Falsification: exp(2R) alternative gives κ < 0 (impossible)

    See: scripts/test_m_derivation.py, scripts/test_gfactor_transferability.py

    DERIVATION STATUS:
    - The formula behaves structurally (no fitted parameters), but the mathematical
      derivation from PRZZ's T^{-α-β} to this form is not complete.
    - The exponential weighting is incorporated internally in S12(±R) integrals;
      applying T^{-α-β} = exp(2R) externally would double-count.
    - See docs/APPENDIX_D_EXPONENTIAL_TRACE.md for explicit trace of where exp(2Rt)
      enters the integrand.

    Assembly formula:
        c = (I₁+I₂ at +R) + m×(I₁+I₂ at -R) + (I₃+I₄ at +R)

    Args:
        theta: θ parameter (typically 4/7)
        R: R parameter (typically 1.3036 for κ, 1.1167 for κ*)
        n: Number of quadrature points per dimension
        polynomials: Dict with keys "P1", "P2", "P3", "Q"
        pair_mode: "ordered" (no symmetry fold), "hybrid" (fold only S12), or "triangle" (legacy)
        mirror_mode:
            - "empirical_scalar" (default): legacy shim behavior (no Q operator shift)
            - "operator_q_shift": apply Q(1+D) effect in the mirror (-R) branch by
              replacing Q(x) with the lifted polynomial Q(x+1) (binomial-lifted
              coefficients), leaving the +R branch and S34 untouched.
            - "difference_quotient_v3": TRUE unified bracket approach (Phase 22)
              Builds bracket at each (u,t), does NOT compute at +R/-R separately.
              D=0 and B/A=5 emerge from the structure.
        use_factorial_normalization: Apply 1/(ℓ₁!×ℓ₂!) normalization
        mode: "main" (default) - I₅ forbidden per PRZZ
        n_quad_a: Quadrature points for Case C a-integral
        K: Mollifier piece count (for mirror multiplier, default 3)
        normalize_scalar_baseline: Deprecated, use normalization_mode (default True).
        normalization_mode: For difference_quotient_v3 mode only. One of:
            - "none": No normalization (raw unified bracket)
            - "scalar": Divide by F(R)/2 = (exp(2R)-1)/(4R) [Phase 22]
            - "diagnostic_corrected": QUARANTINED (Phase 24). Divide by F(R)/2 × correction(R).
              Requires allow_diagnostic_correction=True. [Phase 23 - empirically fitted]
            - "auto": Use "scalar" if normalize_scalar_baseline=True, else "none"
        allow_diagnostic_correction: Must be True to use "diagnostic_corrected" mode.
            This mode uses empirically-fitted correction and violates "derived > tuned" discipline.

    Returns:
        EvaluationResult with c and breakdown
    """
    if mode != "main":
        raise ValueError("compute_c_paper_with_mirror only supports mode='main' (I₅ forbidden per PRZZ)")

    if mirror_mode not in ("empirical_scalar", "operator_q_shift", "operator_q_shift_joint", "difference_quotient", "difference_quotient_v2", "difference_quotient_v3"):
        raise ValueError(
            "mirror_mode must be 'empirical_scalar', 'operator_q_shift', 'operator_q_shift_joint', "
            f"'difference_quotient', 'difference_quotient_v2', or 'difference_quotient_v3', got '{mirror_mode}'"
        )

    # Handle difference_quotient mode early - use unified bracket evaluator
    if mirror_mode == "difference_quotient":
        from src.unified_bracket_evaluator import compute_s12_with_difference_quotient
        from src.terms_k3_d1 import make_all_terms_k3

        # Compute S12 using difference quotient approach
        s12_result = compute_s12_with_difference_quotient(
            polynomials=polynomials,
            theta=theta,
            R=R,
            n=n,
            use_factorial_normalization=use_factorial_normalization,
            benchmark="difference_quotient",
        )

        # For S34, still use empirical approach (I3/I4 don't need mirror)
        all_terms_plus = make_all_terms_k3(theta, R, kernel_regime="paper")

        import math
        factorial_norm = {
            "11": 1.0 / (math.factorial(1) * math.factorial(1)),
            "22": 1.0 / (math.factorial(2) * math.factorial(2)),
            "33": 1.0 / (math.factorial(3) * math.factorial(3)),
            "12": 1.0 / (math.factorial(1) * math.factorial(2)),
            "13": 1.0 / (math.factorial(1) * math.factorial(3)),
            "23": 1.0 / (math.factorial(2) * math.factorial(3)),
        }
        symmetry_factor = {
            "11": 1.0, "22": 1.0, "33": 1.0,
            "12": 2.0, "13": 2.0, "23": 2.0
        }

        i3_i4_plus_total = 0.0
        per_term = {}

        for pair_key in ["11", "22", "33", "12", "13", "23"]:
            terms_plus = all_terms_plus[pair_key]
            norm = factorial_norm[pair_key] if use_factorial_normalization else 1.0
            sym = symmetry_factor[pair_key]
            full_norm = sym * norm

            # I₃ and I₄ (indices 2, 3) - NO mirror
            for term_plus in terms_plus[2:4]:  # I₃, I₄
                result_plus = evaluate_term(
                    term_plus, polynomials, n, R=R, theta=theta, n_quad_a=n_quad_a
                )
                contrib = full_norm * result_plus.value
                i3_i4_plus_total += contrib
                per_term[term_plus.name] = result_plus.value

        # Combine S12 (from difference quotient) with S34 (empirical)
        # In difference quotient mode: S12_plus = 0, S12_minus = A
        mirror_mult = math.exp(R) + (2 * K - 1)
        total = s12_result.S12_minus * mirror_mult + i3_i4_plus_total

        # Store diagnostic breakdown
        per_term["_S12_plus_total"] = s12_result.S12_plus
        per_term["_S12_minus_total"] = s12_result.S12_minus
        per_term["_S34_total"] = i3_i4_plus_total
        per_term["_I3_I4_plus_total"] = i3_i4_plus_total
        per_term["_mirror_multiplier"] = mirror_mult
        per_term["_mirror_mode"] = mirror_mode
        per_term["_s12_per_pair"] = s12_result.per_pair
        per_term["_abd_D"] = s12_result.abd.D
        per_term["_abd_B_over_A"] = s12_result.abd.B_over_A
        per_term["_assembly"] = (
            f"c = {mirror_mult:.4f}×S12_minus + S34 (difference quotient: S12_plus=0)"
        )

        return EvaluationResult(
            total=total,
            per_term=per_term,
            n=n,
            term_results=None
        )

    # Handle difference_quotient_v2 mode - uses unified S12 evaluator with symmetry-based D=0
    if mirror_mode == "difference_quotient_v2":
        from src.unified_s12_evaluator import compute_S12_unified_v2, run_dual_benchmark_v2
        from src.terms_k3_d1 import make_all_terms_k3
        import math

        # Compute S12 using the unified difference quotient structure
        # Key insight: I1(+R) = exp(2R) × I1(-R), so S12_combined = 0
        s12_result = compute_S12_unified_v2(
            R=R,
            theta=theta,
            n_quad=n,
            I34_plus=0.0,  # Micro-case: I34 = 0
            benchmark="difference_quotient_v2",
        )

        # For S34, still use empirical approach (I3/I4 don't need mirror)
        all_terms_plus = make_all_terms_k3(theta, R, kernel_regime="paper")

        factorial_norm = {
            "11": 1.0 / (math.factorial(1) * math.factorial(1)),
            "22": 1.0 / (math.factorial(2) * math.factorial(2)),
            "33": 1.0 / (math.factorial(3) * math.factorial(3)),
            "12": 1.0 / (math.factorial(1) * math.factorial(2)),
            "13": 1.0 / (math.factorial(1) * math.factorial(3)),
            "23": 1.0 / (math.factorial(2) * math.factorial(3)),
        }
        symmetry_factor = {
            "11": 1.0, "22": 1.0, "33": 1.0,
            "12": 2.0, "13": 2.0, "23": 2.0
        }

        i3_i4_plus_total = 0.0
        per_term = {}

        for pair_key in ["11", "22", "33", "12", "13", "23"]:
            terms_plus = all_terms_plus[pair_key]
            norm = factorial_norm[pair_key] if use_factorial_normalization else 1.0
            sym = symmetry_factor[pair_key]
            full_norm = sym * norm

            # I₃ and I₄ (indices 2, 3) - NO mirror
            for term_plus in terms_plus[2:4]:  # I₃, I₄
                result_plus = evaluate_term(
                    term_plus, polynomials, n, R=R, theta=theta, n_quad_a=n_quad_a
                )
                contrib = full_norm * result_plus.value
                i3_i4_plus_total += contrib
                per_term[term_plus.name] = result_plus.value

        # In the unified structure:
        # - S12_combined = I1_plus - exp(2R)*I1_minus = 0 (by symmetry)
        # - The baseline A = I1_minus
        # - D = 0 by construction
        # - c = A × exp(R) + B where B = 5A + D = 5A (since D=0)
        # - Therefore c = A × (exp(R) + 5) + I34

        mirror_mult = math.exp(R) + (2 * K - 1)  # exp(R) + 5 for K=3
        A = s12_result.I1_minus  # The baseline value
        total = A * mirror_mult + i3_i4_plus_total

        # Store diagnostic breakdown
        per_term["_S12_combined"] = s12_result.S12_combined
        per_term["_I1_plus"] = s12_result.I1_plus
        per_term["_I1_minus"] = s12_result.I1_minus
        per_term["_ratio"] = s12_result.ratio
        per_term["_expected_ratio"] = s12_result.expected_ratio
        per_term["_S34_total"] = i3_i4_plus_total
        per_term["_I3_I4_plus_total"] = i3_i4_plus_total
        per_term["_mirror_multiplier"] = mirror_mult
        per_term["_mirror_mode"] = mirror_mode
        per_term["_abd_D"] = s12_result.abd.D
        per_term["_abd_B_over_A"] = s12_result.abd.B_over_A
        per_term["_abd_A"] = s12_result.abd.A
        per_term["_abd_B"] = s12_result.abd.B
        per_term["_symmetry_holds"] = abs(s12_result.ratio - s12_result.expected_ratio) < 1e-10
        per_term["_assembly"] = (
            f"c = A×(exp(R)+5) + S34 where A=I1_minus (unified: D={s12_result.abd.D:.2e})"
        )

        return EvaluationResult(
            total=total,
            per_term=per_term,
            n=n,
            term_results=None
        )

    # Handle difference_quotient_v3 mode - TRUE integrand-level unified bracket
    # This builds the unified bracket at each (u,t), does NOT compute at +R/-R separately
    if mirror_mode == "difference_quotient_v3":
        from src.unified_s12_evaluator_v3 import compute_S12_unified_v3
        from src.terms_k3_d1 import make_all_terms_k3
        import math

        # Compute S12 using the TRUE unified bracket structure
        # Key insight: The bracket exp(2Rt + Rθ(2t-1)(x+y)) already combines direct+mirror
        # We do NOT compute at +R and -R separately
        #
        # Phase 22: normalize_scalar_baseline divides by F(R) = (exp(2R)-1)/(2R)
        # to remove the t-integral scalar inflation factor
        s12_result = compute_S12_unified_v3(
            R=R,
            theta=theta,
            polynomials=polynomials,
            n_quad_u=n,
            n_quad_t=n,
            include_Q=True,
            benchmark="difference_quotient_v3",
            normalize_scalar_baseline=normalize_scalar_baseline,
            normalization_mode=normalization_mode,
            allow_diagnostic_correction=allow_diagnostic_correction,
        )

        # For S34, still use empirical approach (I3/I4 don't need mirror)
        all_terms_plus = make_all_terms_k3(theta, R, kernel_regime="paper")

        factorial_norm = {
            "11": 1.0 / (math.factorial(1) * math.factorial(1)),
            "22": 1.0 / (math.factorial(2) * math.factorial(2)),
            "33": 1.0 / (math.factorial(3) * math.factorial(3)),
            "12": 1.0 / (math.factorial(1) * math.factorial(2)),
            "13": 1.0 / (math.factorial(1) * math.factorial(3)),
            "23": 1.0 / (math.factorial(2) * math.factorial(3)),
        }
        symmetry_factor = {
            "11": 1.0, "22": 1.0, "33": 1.0,
            "12": 2.0, "13": 2.0, "23": 2.0
        }

        i3_i4_plus_total = 0.0
        per_term = {}

        for pair_key in ["11", "22", "33", "12", "13", "23"]:
            terms_plus = all_terms_plus[pair_key]
            norm = factorial_norm[pair_key] if use_factorial_normalization else 1.0
            sym = symmetry_factor[pair_key]
            full_norm = sym * norm

            # I₃ and I₄ (indices 2, 3) - NO mirror
            for term_plus in terms_plus[2:4]:  # I₃, I₄
                result_plus = evaluate_term(
                    term_plus, polynomials, n, R=R, theta=theta, n_quad_a=n_quad_a
                )
                contrib = full_norm * result_plus.value
                i3_i4_plus_total += contrib
                per_term[term_plus.name] = result_plus.value

        # In the unified v3 structure:
        # - S12_total = V = A × (exp(R) + 5) where A is the baseline
        # - D = 0 emerges from the bracket structure (not forced)
        # - c = V + I34 = S12_total + S34

        # The unified value already incorporates the mirror assembly
        total = s12_result.S12_total + i3_i4_plus_total

        # Extract A for diagnostics
        mirror_mult = math.exp(R) + (2 * K - 1)  # exp(R) + 5 for K=3
        A = s12_result.S12_total / mirror_mult
        B = 5 * A
        D = s12_result.S12_total - A * math.exp(R) - B

        # Store diagnostic breakdown
        per_term["_S12_unified_total"] = s12_result.S12_total
        per_term["_S12_unnormalized"] = s12_result.S12_unnormalized
        per_term["_S34_total"] = i3_i4_plus_total
        per_term["_I3_I4_plus_total"] = i3_i4_plus_total
        per_term["_mirror_multiplier"] = mirror_mult
        per_term["_mirror_mode"] = mirror_mode
        per_term["_normalize_scalar_baseline"] = normalize_scalar_baseline
        per_term["_normalization_mode"] = s12_result.normalization_mode
        per_term["_scalar_baseline_factor"] = s12_result.scalar_baseline_factor
        per_term["_normalization_factor"] = s12_result.normalization_factor
        per_term["_abd_A"] = A
        per_term["_abd_B"] = B
        per_term["_abd_D"] = D
        per_term["_abd_B_over_A"] = B / A if A != 0 else float('inf')
        per_term["_per_pair_contributions"] = s12_result.pair_contributions
        per_term["_assembly"] = (
            f"c = S12_unified + S34 (true bracket: D={D:.2e}, B/A={B/A:.4f}, "
            f"mode={s12_result.normalization_mode}, factor={s12_result.normalization_factor:.4f})"
        )

        return EvaluationResult(
            total=total,
            per_term=per_term,
            n=n,
            term_results=None
        )

    # Q(D)[T^{-α-β} F] = T^{-α-β} Q(1+D)F  ⇒  mirror branch uses Q(x+1).
    #
    # Modes:
    # - operator_q_shift: apply Q(x) -> Q(x+1) to BOTH I₁ and I₂ mirror channels
    # - operator_q_shift_joint: apply Q(x) -> Q(x+1) to I₁ mirror channel only
    q_shift_i1 = 0.0
    q_shift_i2 = 0.0
    if mirror_mode == "operator_q_shift":
        q_shift_i1 = 1.0
        q_shift_i2 = 1.0
    elif mirror_mode == "operator_q_shift_joint":
        q_shift_i1 = 1.0
        q_shift_i2 = 0.0

    # Backwards-compatible aggregate (used by existing diagnostics/tests).
    q_poly_shift_mirror = q_shift_i1

    if pair_mode == "hybrid":
        # Fast-but-correct: S12 mirrored using triangle×2 (swap-symmetric), S34 summed over ordered pairs.
        res = compute_c_paper_ordered(
            theta=theta,
            R=R,
            n=n,
            polynomials=polynomials,
            use_factorial_normalization=use_factorial_normalization,
            n_quad_a=n_quad_a,
            K=K,
            s12_pair_mode="triangle",
            q_poly_shift_mirror=q_poly_shift_mirror,
            q_poly_shift_mirror_i2=q_shift_i2,
        )
        if res.per_term is not None:
            res.per_term["_mirror_mode"] = mirror_mode
        return res

    if pair_mode == "ordered":
        # Full ordered baseline: S12 mirrored over all 9 ordered pairs (no symmetry fold),
        # and S34 summed over all 9 ordered pairs.
        res = compute_c_paper_ordered(
            theta=theta,
            R=R,
            n=n,
            polynomials=polynomials,
            use_factorial_normalization=use_factorial_normalization,
            n_quad_a=n_quad_a,
            K=K,
            s12_pair_mode="ordered",
            q_poly_shift_mirror=q_poly_shift_mirror,
            q_poly_shift_mirror_i2=q_shift_i2,
        )
        if res.per_term is not None:
            res.per_term["_mirror_mode"] = mirror_mode
        return res

    if pair_mode != "triangle":
        raise ValueError("pair_mode must be 'ordered', 'hybrid', or 'triangle'")

    import math
    from src.terms_k3_d1 import make_all_terms_k3
    from src.q_operator import lift_poly_by_shift

    # Build terms with paper regime
    all_terms_plus = make_all_terms_k3(theta, R, kernel_regime="paper")
    all_terms_minus = make_all_terms_k3(theta, -R, kernel_regime="paper")

    polynomials_mirror_i1 = polynomials
    if q_shift_i1 != 0.0:
        polynomials_mirror_i1 = dict(polynomials)
        polynomials_mirror_i1["Q"] = lift_poly_by_shift(polynomials["Q"], shift=q_shift_i1)

    polynomials_mirror_i2 = polynomials
    if q_shift_i2 != 0.0:
        polynomials_mirror_i2 = dict(polynomials)
        polynomials_mirror_i2["Q"] = lift_poly_by_shift(polynomials["Q"], shift=q_shift_i2)

    # Mirror multiplier: m = exp(R) + (2K - 1)
    mirror_mult = math.exp(R) + (2 * K - 1)

    # Factorial normalization factors
    factorial_norm = {
        "11": 1.0 / (math.factorial(1) * math.factorial(1)),
        "22": 1.0 / (math.factorial(2) * math.factorial(2)),
        "33": 1.0 / (math.factorial(3) * math.factorial(3)),
        "12": 1.0 / (math.factorial(1) * math.factorial(2)),
        "13": 1.0 / (math.factorial(1) * math.factorial(3)),
        "23": 1.0 / (math.factorial(2) * math.factorial(3)),
    }

    symmetry_factor = {
        "11": 1.0, "22": 1.0, "33": 1.0,
        "12": 2.0, "13": 2.0, "23": 2.0
    }

    total = 0.0
    per_term = {}

    # Track I₁+I₂ and I₃+I₄ separately for diagnostics
    i1_i2_plus_total = 0.0
    i1_i2_minus_total = 0.0
    i3_i4_plus_total = 0.0
    i1_plus_total = 0.0
    i1_minus_total = 0.0
    i2_plus_total = 0.0
    i2_minus_total = 0.0

    for pair_key in ["11", "22", "33", "12", "13", "23"]:
        terms_plus = all_terms_plus[pair_key]
        terms_minus = all_terms_minus[pair_key]

        # Get normalization
        if use_factorial_normalization:
            norm = factorial_norm[pair_key]
        else:
            norm = 1.0
        sym = symmetry_factor[pair_key]
        full_norm = sym * norm

        # I₁ and I₂ (indices 0, 1) - need mirror
        for i, term_plus in enumerate(terms_plus[:2]):  # I₁, I₂
            term_minus = terms_minus[i]

            result_plus = evaluate_term(
                term_plus, polynomials, n, R=R, theta=theta, n_quad_a=n_quad_a
            )
            polynomials_mirror = polynomials_mirror_i1 if i == 0 else polynomials_mirror_i2
            result_minus = evaluate_term(
                term_minus, polynomials_mirror, n, R=-R, theta=theta, n_quad_a=n_quad_a
            )

            # Mirror assembly: val_plus + m * val_minus
            val = result_plus.value + mirror_mult * result_minus.value
            contrib = full_norm * val

            i1_i2_plus_total += full_norm * result_plus.value
            i1_i2_minus_total += full_norm * result_minus.value
            if i == 0:
                i1_plus_total += full_norm * result_plus.value
                i1_minus_total += full_norm * result_minus.value
            else:
                i2_plus_total += full_norm * result_plus.value
                i2_minus_total += full_norm * result_minus.value
            total += contrib

            per_term[f"{term_plus.name}_plus"] = result_plus.value
            per_term[f"{term_plus.name}_minus"] = result_minus.value
            per_term[f"{term_plus.name}_mirror"] = val

        # I₃ and I₄ (indices 2, 3) - NO mirror
        for term_plus in terms_plus[2:4]:  # I₃, I₄
            result_plus = evaluate_term(
                term_plus, polynomials, n, R=R, theta=theta, n_quad_a=n_quad_a
            )

            contrib = full_norm * result_plus.value
            i3_i4_plus_total += contrib
            total += contrib

            per_term[term_plus.name] = result_plus.value

    # Store diagnostic breakdown
    per_term["_I1_I2_plus_total"] = i1_i2_plus_total
    per_term["_I1_I2_minus_total"] = i1_i2_minus_total
    per_term["_I3_I4_plus_total"] = i3_i4_plus_total
    per_term["_S34_triangle_total"] = i3_i4_plus_total
    per_term["_I1_plus_total"] = i1_plus_total
    per_term["_I1_minus_total"] = i1_minus_total
    per_term["_I2_plus_total"] = i2_plus_total
    per_term["_I2_minus_total"] = i2_minus_total
    per_term["_S12_plus_total"] = i1_i2_plus_total
    per_term["_S12_minus_total"] = i1_i2_minus_total
    per_term["_mirror_multiplier"] = mirror_mult
    per_term["_mirror_q_poly_shift"] = float(q_poly_shift_mirror)
    per_term["_mirror_mode"] = mirror_mode

    # Additional diagnostics for m_needed calculation (GPT recommendation 2025-12-19)
    # _direct_c: total without mirror contribution
    # _mirror_I12: the I1+I2 at -R (what gets multiplied by m)
    per_term["_direct_c"] = i1_i2_plus_total + i3_i4_plus_total
    per_term["_mirror_I12"] = i1_i2_minus_total

    per_term["_assembly"] = (
        f"c = I1I2(+R) + {mirror_mult:.4f}×I1I2(-R) + I3I4(+R)"
    )
    # NOTE: m = exp(R) + (2K-1) is EMPIRICAL, not derived from first principles.
    # GPT analysis suggests this approximates the true Q-operator shift:
    # Q(D)[T^{-α-β} F] = T^{-α-β} Q(1+D)F, not Q(D)F × T^{-α-β}
    per_term["_formula"] = f"m = exp({R}) + {2*K-1} = {mirror_mult:.6f} (EMPIRICAL)"

    return EvaluationResult(
        total=total,
        per_term=per_term,
        n=n,
        term_results=None
    )


def compute_c_paper_ordered(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict[str, PolyLike],
    *,
    use_factorial_normalization: bool = True,
    n_quad_a: int = 40,
    K: int = 3,
    s12_pair_mode: str = "triangle",
    q_poly_shift_mirror: float = 0.0,
    q_poly_shift_mirror_i1: Optional[float] = None,
    q_poly_shift_mirror_i2: Optional[float] = None,
) -> EvaluationResult:
    """
    Compute c using paper regime with ORDERED S34 and mirror assembly.

    CRITICAL UPDATE (2025-12-19):
    Measurements proved that S12 (I1+I2) is symmetric under pair swap, but
    S34 (I3+I4) is NOT. Therefore:
    - S12: use triangle×2 (safe, empirically verified)
    - S34: evaluate all 9 ordered pairs (no symmetry factor)

    Assembly formula:
        c = S12(+R) + m×S12(-R) + S34_ordered(+R)

    where m = exp(R) + (2K-1) is the empirical mirror multiplier.

    Args:
        theta: θ parameter (typically 4/7)
        R: R parameter (typically 1.3036 for κ, 1.1167 for κ*)
        n: Number of quadrature points per dimension
        polynomials: Dict with keys "P1", "P2", "P3", "Q"
        use_factorial_normalization: Apply 1/(ℓ₁!×ℓ₂!) normalization
        n_quad_a: Quadrature points for Case C a-integral
        K: Mollifier piece count (for mirror multiplier, default 3)
        s12_pair_mode: "triangle" (default, folds only S12) or "ordered" (no folding anywhere)

    Returns:
        EvaluationResult with c and breakdown
    """
    import math
    from src.q_operator import lift_poly_by_shift
    from src.terms_k3_d1 import make_all_terms_k3, make_all_terms_k3_ordered

    if s12_pair_mode not in ("triangle", "ordered"):
        raise ValueError(f"s12_pair_mode must be 'triangle' or 'ordered', got '{s12_pair_mode}'")

    # Build terms
    # For S12 (I1+I2):
    # - triangle mode: use upper-triangle terms (safe to fold because S12 is swap-symmetric)
    # - ordered mode: use all 9 ordered pairs directly (no symmetry fold)
    triangle_plus = None
    triangle_minus = None
    ordered_minus = None
    if s12_pair_mode == "triangle":
        triangle_plus = make_all_terms_k3(theta, R, kernel_regime="paper")
        triangle_minus = make_all_terms_k3(theta, -R, kernel_regime="paper")
    else:
        ordered_minus = make_all_terms_k3_ordered(theta, -R, kernel_regime="paper")

    # For S34 (I3+I4): use ordered terms (NOT symmetric, must not fold)
    ordered_plus = make_all_terms_k3_ordered(theta, R, kernel_regime="paper")

    # Mirror multiplier: m = exp(R) + (2K - 1)
    mirror_mult = math.exp(R) + (2 * K - 1)

    # Mirror-side Q-shift knobs:
    # - q_poly_shift_mirror (legacy): apply to BOTH I₁ and I₂ mirror channels
    # - q_poly_shift_mirror_i1 / _i2: override per-channel
    shift_i1 = q_poly_shift_mirror if q_poly_shift_mirror_i1 is None else q_poly_shift_mirror_i1
    shift_i2 = q_poly_shift_mirror if q_poly_shift_mirror_i2 is None else q_poly_shift_mirror_i2

    polynomials_mirror_i1 = polynomials
    if shift_i1 != 0.0:
        polynomials_mirror_i1 = dict(polynomials)
        polynomials_mirror_i1["Q"] = lift_poly_by_shift(polynomials["Q"], shift=shift_i1)

    polynomials_mirror_i2 = polynomials
    if shift_i2 != 0.0:
        polynomials_mirror_i2 = dict(polynomials)
        polynomials_mirror_i2["Q"] = lift_poly_by_shift(polynomials["Q"], shift=shift_i2)

    # Factorial normalization factors
    f_triangle = {
        "11": 1.0, "22": 0.25, "33": 1/36,
        "12": 0.5, "13": 1/6, "23": 1/12,
    }
    f_ordered = {
        "11": 1.0, "22": 0.25, "33": 1/36,
        "12": 0.5, "21": 0.5,
        "13": 1/6, "31": 1/6,
        "23": 1/12, "32": 1/12,
    }

    # Symmetry factor for S12 (triangle mode)
    sym_s12 = {"11": 1.0, "22": 1.0, "33": 1.0, "12": 2.0, "13": 2.0, "23": 2.0}

    total = 0.0
    per_term = {}

    # Track components for diagnostics
    s12_plus_total = 0.0
    s12_minus_total = 0.0
    s34_ordered_total = 0.0
    i1_plus_total = 0.0
    i1_minus_total = 0.0
    i2_plus_total = 0.0
    i2_minus_total = 0.0

    # =====================================================================
    # PART 1: S12 (I1+I2) with mirror assembly
    # =====================================================================
    if s12_pair_mode == "triangle":
        assert triangle_plus is not None
        assert triangle_minus is not None
        s12_pair_keys = ["11", "22", "33", "12", "13", "23"]
        for pair_key in s12_pair_keys:
            terms_plus = triangle_plus[pair_key]
            terms_minus = triangle_minus[pair_key]

            norm = f_triangle[pair_key] if use_factorial_normalization else 1.0
            sym = sym_s12[pair_key]
            full_norm = sym * norm

            # I₁ and I₂ (indices 0, 1)
            for i in range(2):  # I₁, I₂
                term_plus = terms_plus[i]
                term_minus = terms_minus[i]

                result_plus = evaluate_term(
                    term_plus, polynomials, n, R=R, theta=theta, n_quad_a=n_quad_a
                )
                polynomials_mirror = polynomials_mirror_i1 if i == 0 else polynomials_mirror_i2
                result_minus = evaluate_term(
                    term_minus, polynomials_mirror, n, R=-R, theta=theta, n_quad_a=n_quad_a
                )

                # Mirror assembly: val_plus + m * val_minus
                val = result_plus.value + mirror_mult * result_minus.value
                contrib = full_norm * val

                s12_plus_total += full_norm * result_plus.value
                s12_minus_total += full_norm * result_minus.value
                if i == 0:
                    i1_plus_total += full_norm * result_plus.value
                    i1_minus_total += full_norm * result_minus.value
                else:
                    i2_plus_total += full_norm * result_plus.value
                    i2_minus_total += full_norm * result_minus.value
                total += contrib

                per_term[f"{term_plus.name}_plus"] = result_plus.value
                per_term[f"{term_plus.name}_minus"] = result_minus.value
    else:
        assert ordered_minus is not None
        for pair_key in ["11", "22", "33", "12", "21", "13", "31", "23", "32"]:
            terms_plus = ordered_plus[pair_key]
            terms_minus = ordered_minus[pair_key]

            full_norm = f_ordered[pair_key] if use_factorial_normalization else 1.0

            for i in range(2):  # I₁, I₂
                term_plus = terms_plus[i]
                term_minus = terms_minus[i]

                result_plus = evaluate_term(
                    term_plus, polynomials, n, R=R, theta=theta, n_quad_a=n_quad_a
                )
                polynomials_mirror = polynomials_mirror_i1 if i == 0 else polynomials_mirror_i2
                result_minus = evaluate_term(
                    term_minus, polynomials_mirror, n, R=-R, theta=theta, n_quad_a=n_quad_a
                )

                val = result_plus.value + mirror_mult * result_minus.value
                contrib = full_norm * val

                s12_plus_total += full_norm * result_plus.value
                s12_minus_total += full_norm * result_minus.value
                if i == 0:
                    i1_plus_total += full_norm * result_plus.value
                    i1_minus_total += full_norm * result_minus.value
                else:
                    i2_plus_total += full_norm * result_plus.value
                    i2_minus_total += full_norm * result_minus.value
                total += contrib

                per_term[f"{term_plus.name}_plus"] = result_plus.value
                per_term[f"{term_plus.name}_minus"] = result_minus.value

    # =====================================================================
    # PART 2: S34 (I3+I4) with TRIANGLE pairs ×2, NO mirror
    # =====================================================================
    # NOTE (2025-12-22): Despite S34 being term-asymmetric (I₃(1,2) ≠ I₃(2,1)),
    # PRZZ sums over ℓ₁ ≤ ℓ₂ with symmetry factor 2 for off-diagonal.
    # The asymmetry test (Δ_S34 = 0.54) proved terms differ, but PRZZ never
    # evaluates both (1,2) and (2,1) — they use triangle convention throughout.
    # Numerical verification: triangle×2 gives -1.3% error, ordered gives +11%.
    for pair_key in ["11", "22", "33", "12", "13", "23"]:
        terms = ordered_plus[pair_key]

        norm = f_triangle[pair_key] if use_factorial_normalization else 1.0
        sym = sym_s12[pair_key]  # 1 for diagonal, 2 for off-diagonal
        full_norm = sym * norm

        # I₃ and I₄ (indices 2, 3)
        for i in [2, 3]:  # I₃, I₄
            term = terms[i]
            result = evaluate_term(
                term, polynomials, n, R=R, theta=theta, n_quad_a=n_quad_a
            )

            contrib = full_norm * result.value
            s34_ordered_total += contrib
            total += contrib

            per_term[f"{term.name}_triangle"] = result.value

    # Store diagnostic breakdown
    per_term["_I1_plus_total"] = i1_plus_total
    per_term["_I1_minus_total"] = i1_minus_total
    per_term["_I2_plus_total"] = i2_plus_total
    per_term["_I2_minus_total"] = i2_minus_total
    per_term["_S12_plus_total"] = s12_plus_total
    per_term["_S12_minus_total"] = s12_minus_total
    per_term["_S34_ordered_total"] = s34_ordered_total
    per_term["_S34_plus_total"] = s34_ordered_total
    per_term["_mirror_multiplier"] = mirror_mult
    per_term["_mirror_q_poly_shift"] = float(q_poly_shift_mirror)
    per_term["_mirror_q_poly_shift_i1"] = float(shift_i1)
    per_term["_mirror_q_poly_shift_i2"] = float(shift_i2)
    # Compatibility with compute_c_paper_with_mirror diagnostics/tests
    per_term["_direct_c"] = s12_plus_total + s34_ordered_total
    per_term["_mirror_I12"] = s12_minus_total
    per_term["_pair_mode"] = "ordered" if s12_pair_mode == "ordered" else "hybrid"
    per_term["_s12_pair_mode"] = s12_pair_mode

    per_term["_assembly"] = (
        f"c = S12(+R) + {mirror_mult:.4f}×S12(-R) + S34_triangle(+R)"
    )
    per_term["_note"] = (
        "S12 uses triangle×2, S34 uses triangle×2 (PRZZ convention, not ordered)"
    )

    return EvaluationResult(
        total=total,
        per_term=per_term,
        n=n,
        term_results=None
    )


def compute_c_paper_two_weight(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict[str, PolyLike],
    *,
    m1: float,
    m2: float,
    use_factorial_normalization: bool = True,
    n_quad_a: int = 40,
) -> EvaluationResult:
    """
    Compute c using paper regime with TWO-WEIGHT mirror assembly (Stage C2/C3).

    This implements the empirically-validated two-weight model:
        c = I₁(+R) + m₁×I₁(-R) + I₂(+R) + m₂×I₂(-R) + S34(+R)

    where m₁ and m₂ are separate mirror coefficients for I₁ and I₂.

    THEORETICAL BASIS (from TeX analysis):
    - I₁ has (1-u)² weight, Q(·)×Q(·) structure, full x,y dependence
    - I₂ has no weight, Q(t)² structure, no x,y dependence
    - The different structures lead to different effective mirror coefficients.
      These coefficients are diagnostic outputs of a two-benchmark solve; do not
      hard-code any particular fitted values (they can move as structural fixes land).

    IMPORTANT: This function takes m₁ and m₂ as parameters. To find the
    values that match benchmark targets, use the solve_two_weight_coefficients()
    function, which solves a 2×2 system using both κ and κ* benchmarks.

    Args:
        theta: θ parameter (typically 4/7)
        R: R parameter
        n: Number of quadrature points per dimension
        polynomials: Dict with keys "P1", "P2", "P3", "Q"
        m1: Mirror coefficient for I₁
        m2: Mirror coefficient for I₂
        use_factorial_normalization: Apply 1/(ℓ₁!×ℓ₂!) normalization
        n_quad_a: Quadrature points for Case C a-integral

    Returns:
        EvaluationResult with c and detailed breakdown
    """
    import math
    from src.terms_k3_d1 import make_all_terms_k3_ordered

    # Build terms for all 9 ordered pairs
    ordered_plus = make_all_terms_k3_ordered(theta, R, kernel_regime="paper")
    ordered_minus = make_all_terms_k3_ordered(theta, -R, kernel_regime="paper")

    # Factorial normalization factors
    f_ordered = {
        "11": 1.0, "22": 0.25, "33": 1/36,
        "12": 0.5, "21": 0.5,
        "13": 1/6, "31": 1/6,
        "23": 1/12, "32": 1/12,
    }

    total = 0.0
    per_term = {}

    # Track components for diagnostics
    i1_plus_total = 0.0
    i1_minus_total = 0.0
    i2_plus_total = 0.0
    i2_minus_total = 0.0
    s34_plus_total = 0.0

    # =====================================================================
    # PART 1: I₁ with mirror coefficient m₁
    # =====================================================================
    for pair_key in ["11", "22", "33", "12", "21", "13", "31", "23", "32"]:
        terms_plus = ordered_plus[pair_key]
        terms_minus = ordered_minus[pair_key]
        norm = f_ordered[pair_key] if use_factorial_normalization else 1.0

        # I₁ is index 0
        term_plus = terms_plus[0]
        term_minus = terms_minus[0]

        result_plus = evaluate_term(
            term_plus, polynomials, n, R=R, theta=theta, n_quad_a=n_quad_a
        )
        result_minus = evaluate_term(
            term_minus, polynomials, n, R=-R, theta=theta, n_quad_a=n_quad_a
        )

        val = result_plus.value + m1 * result_minus.value
        contrib = norm * val

        i1_plus_total += norm * result_plus.value
        i1_minus_total += norm * result_minus.value
        total += contrib

        per_term[f"{pair_key}_I1_plus"] = result_plus.value
        per_term[f"{pair_key}_I1_minus"] = result_minus.value

    # =====================================================================
    # PART 2: I₂ with mirror coefficient m₂
    # =====================================================================
    for pair_key in ["11", "22", "33", "12", "21", "13", "31", "23", "32"]:
        terms_plus = ordered_plus[pair_key]
        terms_minus = ordered_minus[pair_key]
        norm = f_ordered[pair_key] if use_factorial_normalization else 1.0

        # I₂ is index 1
        term_plus = terms_plus[1]
        term_minus = terms_minus[1]

        result_plus = evaluate_term(
            term_plus, polynomials, n, R=R, theta=theta, n_quad_a=n_quad_a
        )
        result_minus = evaluate_term(
            term_minus, polynomials, n, R=-R, theta=theta, n_quad_a=n_quad_a
        )

        val = result_plus.value + m2 * result_minus.value
        contrib = norm * val

        i2_plus_total += norm * result_plus.value
        i2_minus_total += norm * result_minus.value
        total += contrib

        per_term[f"{pair_key}_I2_plus"] = result_plus.value
        per_term[f"{pair_key}_I2_minus"] = result_minus.value

    # =====================================================================
    # PART 3: S34 (I₃+I₄) - NO mirror
    # =====================================================================
    for pair_key in ["11", "22", "33", "12", "21", "13", "31", "23", "32"]:
        terms = ordered_plus[pair_key]
        norm = f_ordered[pair_key] if use_factorial_normalization else 1.0

        # I₃ and I₄ are indices 2, 3
        for i, i_label in [(2, "I3"), (3, "I4")]:
            term = terms[i]
            result = evaluate_term(
                term, polynomials, n, R=R, theta=theta, n_quad_a=n_quad_a
            )

            contrib = norm * result.value
            s34_plus_total += contrib
            total += contrib

            per_term[f"{pair_key}_{i_label}_plus"] = result.value

    # Store diagnostic breakdown
    per_term["_I1_plus"] = i1_plus_total
    per_term["_I1_minus"] = i1_minus_total
    per_term["_I2_plus"] = i2_plus_total
    per_term["_I2_minus"] = i2_minus_total
    per_term["_S12_plus"] = i1_plus_total + i2_plus_total
    per_term["_S12_minus"] = i1_minus_total + i2_minus_total
    per_term["_S34_plus"] = s34_plus_total
    per_term["_m1"] = m1
    per_term["_m2"] = m2
    per_term["_m_ratio"] = m1 / m2 if m2 != 0 else float('inf')
    per_term["_pair_mode"] = "ordered"

    per_term["_assembly"] = (
        f"c = I1(+R) + {m1:.4f}×I1(-R) + I2(+R) + {m2:.4f}×I2(-R) + S34(+R)"
    )
    per_term["_model"] = "two_weight"

    return EvaluationResult(
        total=total,
        per_term=per_term,
        n=n,
        term_results=None
    )


def solve_two_weight_coefficients(
    channels_kappa: Dict[str, float],
    channels_kappa_star: Dict[str, float],
    c_target_kappa: float = 2.137,
    c_target_kappa_star: float = 1.938,
) -> Tuple[float, float, float]:
    """
    Solve for (m₁, m₂) that satisfy both κ and κ* benchmark targets.

    Given the split channel values from both benchmarks, solves the 2×2 system:
        I1⁺_κ  + m₁×I1⁻_κ  + I2⁺_κ  + m₂×I2⁻_κ  + S34⁺_κ  = c_κ
        I1⁺_κ* + m₁×I1⁻_κ* + I2⁺_κ* + m₂×I2⁻_κ* + S34⁺_κ* = c_κ*

    Args:
        channels_kappa: Dict with keys "_I1_plus", "_I1_minus", "_I2_plus",
                        "_I2_minus", "_S34_plus" for κ benchmark
        channels_kappa_star: Same keys for κ* benchmark
        c_target_kappa: Target c for κ benchmark (default 2.137)
        c_target_kappa_star: Target c for κ* benchmark (default 1.938)

    Returns:
        Tuple (m₁, m₂, det) where det is the determinant of the coefficient matrix
    """
    import numpy as np

    # Extract channels
    I1m_k = channels_kappa["_I1_minus"]
    I2m_k = channels_kappa["_I2_minus"]
    I1m_s = channels_kappa_star["_I1_minus"]
    I2m_s = channels_kappa_star["_I2_minus"]

    # Coefficient matrix
    A = np.array([
        [I1m_k, I2m_k],
        [I1m_s, I2m_s],
    ])

    # RHS: c_target - I1_plus - I2_plus - S34_plus
    rhs_k = (c_target_kappa
             - channels_kappa["_I1_plus"]
             - channels_kappa["_I2_plus"]
             - channels_kappa["_S34_plus"])
    rhs_s = (c_target_kappa_star
             - channels_kappa_star["_I1_plus"]
             - channels_kappa_star["_I2_plus"]
             - channels_kappa_star["_S34_plus"])

    b = np.array([rhs_k, rhs_s])

    det = np.linalg.det(A)
    if abs(det) < 1e-15:
        return float('inf'), float('inf'), det

    m = np.linalg.solve(A, b)
    return float(m[0]), float(m[1]), float(det)


def compute_c_paper_operator_q_shift(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict[str, PolyLike],
    *,
    use_factorial_normalization: bool = True,
    n_quad_a: int = 40,
    verbose: bool = False,
) -> EvaluationResult:
    """
    Compute c using operator mode: Q → Q_lift in mirror branch only.

    This implements the GPT-guided Stage C3 operator mode:
    - Direct (+R) branch: Uses original polynomials (bitwise identical)
    - Mirror (-R) branch: Transforms Q → Q(1+x) via binomial lift

    The theory: PRZZ mirror assembly involves Q(D) → Q(1+D) where D is the
    differential operator. The binomial lift Q(x) → Q(1+x) captures this.

    If the theory is correct, the implied weights (I_minus_op / I_minus_base)
    should be comparable to the solved two-weight coefficients from the
    κ/κ* benchmark system (see `run_operator_mode_diagnostic.py`), without
    hard-coding any fitted values here.

    Args:
        theta: θ parameter (typically 4/7)
        R: R parameter
        n: Number of quadrature points per dimension
        polynomials: Dict with keys "P1", "P2", "P3", "Q"
        use_factorial_normalization: Apply 1/(ℓ₁!×ℓ₂!) normalization
        n_quad_a: Quadrature points for Case C a-integral
        verbose: Print diagnostic information

    Returns:
        EvaluationResult with c and breakdown including implied weights
    """
    import math
    from src.terms_k3_d1 import make_all_terms_k3_ordered
    from src.q_operator import binomial_lift_coeffs
    from src.polynomials import Polynomial
    import numpy as np

    # =========================================================================
    # STEP 1: Create Q_lift polynomial
    # =========================================================================
    Q = polynomials["Q"]

    # Q may be a QPolynomial (PRZZ basis) or regular Polynomial
    # Convert to monomial form if needed
    if hasattr(Q, "to_monomial"):
        Q_mono = Q.to_monomial()
        Q_coeffs = list(np.asarray(Q_mono.coeffs, dtype=float))
    else:
        Q_coeffs = list(np.asarray(Q.coeffs, dtype=float))

    Q_lift_coeffs = binomial_lift_coeffs(Q_coeffs)
    Q_lift = Polynomial(np.asarray(Q_lift_coeffs, dtype=float))

    # Create lifted polynomials dict for mirror branch
    polys_lift = {
        "P1": polynomials["P1"],
        "P2": polynomials["P2"],
        "P3": polynomials["P3"],
        "Q": Q_lift,
    }

    if verbose:
        print(f"\n{'='*70}")
        print("OPERATOR MODE: Q → Q_lift in mirror branch")
        print(f"{'='*70}")
        print(f"Q coefficients (base):  {Q_coeffs[:6]}...")
        print(f"Q coefficients (lift):  {Q_lift_coeffs[:6]}...")
        print(f"Q(0) base: {sum(c * 0**j for j, c in enumerate(Q_coeffs)):.6f}")
        print(f"Q(1) base: {sum(c * 1**j for j, c in enumerate(Q_coeffs)):.6f}")
        print(f"Q_lift(0) = Q(1): {sum(c * 0**j for j, c in enumerate(Q_lift_coeffs)):.6f}")

    # Build terms for all 9 ordered pairs
    ordered_plus = make_all_terms_k3_ordered(theta, R, kernel_regime="paper")
    ordered_minus = make_all_terms_k3_ordered(theta, -R, kernel_regime="paper")

    # Factorial normalization factors
    f_ordered = {
        "11": 1.0, "22": 0.25, "33": 1/36,
        "12": 0.5, "21": 0.5,
        "13": 1/6, "31": 1/6,
        "23": 1/12, "32": 1/12,
    }

    total = 0.0
    per_term = {}

    # Track components for base vs operator comparison
    i1_plus_total = 0.0
    i1_minus_base = 0.0
    i1_minus_op = 0.0
    i2_plus_total = 0.0
    i2_minus_base = 0.0
    i2_minus_op = 0.0
    s34_plus_total = 0.0

    # =========================================================================
    # PART 1: I₁ with operator Q_lift in mirror branch
    # =========================================================================
    for pair_key in ["11", "22", "33", "12", "21", "13", "31", "23", "32"]:
        terms_plus = ordered_plus[pair_key]
        terms_minus = ordered_minus[pair_key]
        norm = f_ordered[pair_key] if use_factorial_normalization else 1.0

        # I₁ is index 0
        term_plus = terms_plus[0]
        term_minus = terms_minus[0]

        # Direct branch: original polynomials (unchanged)
        result_plus = evaluate_term(
            term_plus, polynomials, n, R=R, theta=theta, n_quad_a=n_quad_a
        )

        # Mirror branch: BASE (original Q)
        result_minus_base = evaluate_term(
            term_minus, polynomials, n, R=-R, theta=theta, n_quad_a=n_quad_a
        )

        # Mirror branch: OPERATOR (lifted Q)
        result_minus_op = evaluate_term(
            term_minus, polys_lift, n, R=-R, theta=theta, n_quad_a=n_quad_a
        )

        # Use operator result for total
        val = result_plus.value + result_minus_op.value
        contrib = norm * val

        i1_plus_total += norm * result_plus.value
        i1_minus_base += norm * result_minus_base.value
        i1_minus_op += norm * result_minus_op.value
        total += contrib

        per_term[f"{pair_key}_I1_plus"] = result_plus.value
        per_term[f"{pair_key}_I1_minus_base"] = result_minus_base.value
        per_term[f"{pair_key}_I1_minus_op"] = result_minus_op.value

    # =========================================================================
    # PART 2: I₂ with operator Q_lift in mirror branch
    # =========================================================================
    for pair_key in ["11", "22", "33", "12", "21", "13", "31", "23", "32"]:
        terms_plus = ordered_plus[pair_key]
        terms_minus = ordered_minus[pair_key]
        norm = f_ordered[pair_key] if use_factorial_normalization else 1.0

        # I₂ is index 1
        term_plus = terms_plus[1]
        term_minus = terms_minus[1]

        # Direct branch: original polynomials (unchanged)
        result_plus = evaluate_term(
            term_plus, polynomials, n, R=R, theta=theta, n_quad_a=n_quad_a
        )

        # Mirror branch: BASE (original Q)
        result_minus_base = evaluate_term(
            term_minus, polynomials, n, R=-R, theta=theta, n_quad_a=n_quad_a
        )

        # Mirror branch: OPERATOR (lifted Q)
        result_minus_op = evaluate_term(
            term_minus, polys_lift, n, R=-R, theta=theta, n_quad_a=n_quad_a
        )

        # Use operator result for total
        val = result_plus.value + result_minus_op.value
        contrib = norm * val

        i2_plus_total += norm * result_plus.value
        i2_minus_base += norm * result_minus_base.value
        i2_minus_op += norm * result_minus_op.value
        total += contrib

        per_term[f"{pair_key}_I2_plus"] = result_plus.value
        per_term[f"{pair_key}_I2_minus_base"] = result_minus_base.value
        per_term[f"{pair_key}_I2_minus_op"] = result_minus_op.value

    # =========================================================================
    # PART 3: S34 (I₃+I₄) - NO mirror (direct branch only)
    # =========================================================================
    for pair_key in ["11", "22", "33", "12", "21", "13", "31", "23", "32"]:
        terms = ordered_plus[pair_key]
        norm = f_ordered[pair_key] if use_factorial_normalization else 1.0

        # I₃ and I₄ are indices 2, 3
        for i, i_label in [(2, "I3"), (3, "I4")]:
            term = terms[i]
            result = evaluate_term(
                term, polynomials, n, R=R, theta=theta, n_quad_a=n_quad_a
            )

            contrib = norm * result.value
            s34_plus_total += contrib
            total += contrib

            per_term[f"{pair_key}_{i_label}_plus"] = result.value

    # =========================================================================
    # Compute implied weights: m_implied = I_minus_op / I_minus_base
    # =========================================================================
    m1_implied = i1_minus_op / i1_minus_base if abs(i1_minus_base) > 1e-15 else float('inf')
    m2_implied = i2_minus_op / i2_minus_base if abs(i2_minus_base) > 1e-15 else float('inf')

    if verbose:
        print(f"\n{'='*70}")
        print("IMPLIED WEIGHTS (operator / base)")
        print(f"{'='*70}")
        print(f"  I1_minus_base:  {i1_minus_base:+.8f}")
        print(f"  I1_minus_op:    {i1_minus_op:+.8f}")
        print(f"  m1_implied:     {m1_implied:.6f}")
        print()
        print(f"  I2_minus_base:  {i2_minus_base:+.8f}")
        print(f"  I2_minus_op:    {i2_minus_op:+.8f}")
        print(f"  m2_implied:     {m2_implied:.6f}")
        print()
        ratio_implied = (m1_implied / m2_implied) if m2_implied != 0 else float("inf")
        print(f"  Ratio:          {ratio_implied:.4f}")

    # Store diagnostic breakdown
    per_term["_I1_plus"] = i1_plus_total
    per_term["_I1_minus_base"] = i1_minus_base
    per_term["_I1_minus_op"] = i1_minus_op
    per_term["_I2_plus"] = i2_plus_total
    per_term["_I2_minus_base"] = i2_minus_base
    per_term["_I2_minus_op"] = i2_minus_op
    per_term["_S12_plus"] = i1_plus_total + i2_plus_total
    per_term["_S12_minus_base"] = i1_minus_base + i2_minus_base
    per_term["_S12_minus_op"] = i1_minus_op + i2_minus_op
    per_term["_S34_plus"] = s34_plus_total
    per_term["_m1_implied"] = m1_implied
    per_term["_m2_implied"] = m2_implied
    per_term["_m_ratio_implied"] = m1_implied / m2_implied if m2_implied != 0 else float('inf')
    per_term["_pair_mode"] = "ordered"
    per_term["_mirror_mode"] = "operator_q_shift"
    per_term["_model"] = "operator_q_shift"

    per_term["_assembly"] = (
        f"c = I1(+R) + I1_op(-R) + I2(+R) + I2_op(-R) + S34(+R)"
    )

    return EvaluationResult(
        total=total,
        per_term=per_term,
        n=n,
        term_results=None
    )


def compute_c_paper_operator_q_shift_joint(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict[str, PolyLike],
    *,
    use_factorial_normalization: bool = True,
    n_quad_a: int = 40,
    verbose: bool = False,
) -> EvaluationResult:
    """
    Compute c using a QUARANTINED operator mode: Q_lift normalized by Q(0)/Q(1).

    This normalization is numerically unstable when Q(1) is near 0 (as in PRZZ's
    benchmark polynomials), and it is NOT TeX-justified. Keep it only as a
    negative control and do not use it for inference.

    This variant addresses the issue that Q(1) differs significantly between
    κ and κ* polynomials (-0.019 vs -0.032), causing the standard Q-lift to
    have opposite effects on the two benchmarks.

    The joint mode normalizes Q_lift so that Q_lift_norm(0) = Q(0) = 1,
    preserving the normalization constraint while still applying the
    structural shift Q(x) → Q(1+x).

    In effect: Q_lift_norm(x) = Q_lift(x) * Q(0) / Q(1)
                              = Q(1+x) * Q(0) / Q(1)

    This ensures that at x=0, Q_lift_norm(0) = Q(1) * Q(0)/Q(1) = Q(0) = 1.

    Args:
        theta: θ parameter (typically 4/7)
        R: R parameter
        n: Number of quadrature points per dimension
        polynomials: Dict with keys "P1", "P2", "P3", "Q"
        use_factorial_normalization: Apply 1/(ℓ₁!×ℓ₂!) normalization
        n_quad_a: Quadrature points for Case C a-integral
        verbose: Print diagnostic information

    Returns:
        EvaluationResult with c and breakdown including implied weights
    """
    import warnings

    warnings.warn(
        "compute_c_paper_operator_q_shift_joint is a negative-control mode (unstable when Q(1)≈0); "
        "prefer compute_c_paper_operator_unified(normalization='l2norm'|'gridnorm', ...) for diagnostics.",
        RuntimeWarning,
        stacklevel=2,
    )

    # Kept for backwards compatibility: delegate to the unified operator evaluator.
    res = compute_c_paper_operator_unified(
        theta=theta,
        R=R,
        n=n,
        polynomials=polynomials,
        use_factorial_normalization=use_factorial_normalization,
        n_quad_a=n_quad_a,
        verbose=verbose,
        normalization="q1_ratio",
        lift_scope="both",
        allow_unstable=True,
    )
    if res.per_term is not None:
        res.per_term["_mirror_mode"] = "operator_q_shift_norm_q1"
        res.per_term["_model"] = "operator_q_shift_norm_q1"
    return res


def compute_c_paper_operator_q_shift_norm_q1(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict[str, PolyLike],
    *,
    use_factorial_normalization: bool = True,
    n_quad_a: int = 40,
    verbose: bool = False,
) -> EvaluationResult:
    """Alias for the quarantined Q(0)/Q(1) normalization (negative control only)."""
    res = compute_c_paper_operator_unified(
        theta=theta,
        R=R,
        n=n,
        polynomials=polynomials,
        use_factorial_normalization=use_factorial_normalization,
        n_quad_a=n_quad_a,
        verbose=verbose,
        normalization="q1_ratio",
        lift_scope="both",
        allow_unstable=True,
    )
    if res.per_term is not None:
        res.per_term["_mirror_mode"] = "operator_q_shift_norm_q1"
        res.per_term["_model"] = "operator_q_shift_norm_q1"
    return res


def compute_c_paper_operator_unified(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict[str, PolyLike],
    *,
    use_factorial_normalization: bool = True,
    n_quad_a: int = 40,
    verbose: bool = False,
    normalization: str = "none",
    lift_scope: str = "both",
    operator_shift_sigma: float = 1.0,
    allow_unstable: bool = False,
) -> EvaluationResult:
    """
    Unified operator mode with configurable normalization and selective lifting.

    This is the main operator mode evaluator supporting all variants (Codex B+C).

    Args:
        theta: θ parameter (typically 4/7)
        R: R parameter
        n: Number of quadrature points per dimension
        polynomials: Dict with keys "P1", "P2", "P3", "Q"
        use_factorial_normalization: Apply 1/(ℓ₁!×ℓ₂!) normalization
        n_quad_a: Quadrature points for Case C a-integral
        verbose: Print diagnostic information
        normalization: How to normalize Q_lift:
            - "none": No normalization (standard Q → Q(1+x))
            - "l2": Scale so ∫Q²dt = ∫Q_lift²dt (RECOMMENDED)
            - "grid": Scale so mean(Q²) = mean(Q_lift²) on quadrature grid
            - "q1_ratio": Scale by Q(0)/Q(1) (UNSTABLE - negative control only)
        lift_scope: Which terms to apply Q-lift to:
            - "both": Apply to both I₁ and I₂ (default)
            - "i1_only": Apply only to I₁
            - "i2_only": Apply only to I₂

    Returns:
        EvaluationResult with c and breakdown
    """
    from dataclasses import replace
    from src.term_dsl import PolyFactor, Term
    from src.terms_k3_d1 import make_all_terms_k3_ordered
    from src.q_operator import binomial_shift_coeffs
    from src.polynomials import Polynomial
    from src.quadrature import gauss_legendre_01
    import numpy as np
    import warnings

    # =========================================================================
    # STEP 1: Create Q_lift polynomial with appropriate normalization
    # =========================================================================
    normalization_key = str(normalization).strip().lower()
    normalization_aliases = {
        "none": "none",
        "l2": "l2",
        "l2norm": "l2",
        "grid": "grid",
        "gridnorm": "grid",
        # Quarantined negative-control normalization.
        "q1_ratio": "q1_ratio",
        "norm_q1": "q1_ratio",
        "q1ratio": "q1_ratio",
        "operator_q_shift_norm_q1": "q1_ratio",
    }
    try:
        normalization_key = normalization_aliases[normalization_key]
    except KeyError as exc:
        raise ValueError(f"Unknown normalization: {normalization}") from exc

    Q = polynomials["Q"]

    # Convert to monomial coefficients
    if hasattr(Q, "to_monomial"):
        Q_mono = Q.to_monomial()
        Q_coeffs = list(np.asarray(Q_mono.coeffs, dtype=float))
    else:
        Q_coeffs = list(np.asarray(Q.coeffs, dtype=float))

    sigma = float(operator_shift_sigma)
    Q_lift_coeffs = binomial_shift_coeffs(Q_coeffs, shift=sigma)

    # Compute key Q values
    Q_0 = Q_coeffs[0] if Q_coeffs else 1.0
    Q_1 = sum(Q_coeffs) if Q_coeffs else 1.0

    nodes, weights = gauss_legendre_01(n)

    # Evaluate Q and Q_lift on grid
    Q_vals = np.array([sum(c * (t ** j) for j, c in enumerate(Q_coeffs)) for t in nodes])
    Q_lift_vals = np.array([sum(c * (t ** j) for j, c in enumerate(Q_lift_coeffs)) for t in nodes])

    # Compute normalization factor based on mode
    if normalization_key == "none":
        norm_factor = 1.0
        norm_desc = "none"
    elif normalization_key == "l2":
        # L2 normalization: sqrt(∫Q²dt / ∫Q_lift²dt)
        q_l2_sq = np.sum(weights * Q_vals**2)
        q_lift_l2_sq = np.sum(weights * Q_lift_vals**2)
        if q_lift_l2_sq > 1e-15:
            norm_factor = np.sqrt(q_l2_sq / q_lift_l2_sq)
        else:
            norm_factor = 1.0
        norm_desc = f"L2 (factor={norm_factor:.4f})"
    elif normalization_key == "grid":
        # Grid normalization: sqrt(mean(Q²) / mean(Q_lift²))
        q_mean_sq = np.mean(Q_vals**2)
        q_lift_mean_sq = np.mean(Q_lift_vals**2)
        if q_lift_mean_sq > 1e-15:
            norm_factor = np.sqrt(q_mean_sq / q_lift_mean_sq)
        else:
            norm_factor = 1.0
        norm_desc = f"grid (factor={norm_factor:.4f})"
    elif normalization_key == "q1_ratio":
        # Q(0)/Q(1) - UNSTABLE, negative control only
        if not allow_unstable:
            raise ValueError(
                "q1_ratio normalization is quarantined (negative control); "
                "pass allow_unstable=True to use it explicitly."
            )
        if abs(Q_1) < 1e-2:
            raise ValueError(
                f"q1_ratio normalization is unstable because |Q(1)|={abs(Q_1):.6f} < 1e-2; "
                "refusing unless you change the threshold."
            )
        warnings.warn(
            "q1_ratio normalization uses Q(0)/Q(1) scaling; this is unstable when Q(1)≈0 "
            "and is intended as a negative control only (not TeX-justified).",
            RuntimeWarning,
            stacklevel=2,
        )
        if abs(Q_1) > 1e-15:
            norm_factor = Q_0 / Q_1
        else:
            norm_factor = 1.0
        norm_desc = f"Q(0)/Q(1) [UNSTABLE] (factor={norm_factor:.4f})"
        if verbose:
            print("WARNING: q1_ratio normalization is UNSTABLE when Q(1)≈0 (negative control).")

    # Apply normalization
    Q_lift_norm_coeffs = [c * norm_factor for c in Q_lift_coeffs]
    Q_lift_norm = Polynomial(np.asarray(Q_lift_norm_coeffs, dtype=float))

    # Create polynomial dicts
    polys_base = polynomials
    polys_lift_full = {
        "P1": polynomials["P1"],
        "P2": polynomials["P2"],
        "P3": polynomials["P3"],
        "Q": Q_lift_norm,
    }
    polys_i1_partial = dict(polys_base)
    polys_i1_partial["Q_lift"] = Q_lift_norm

    if verbose:
        print(f"\n{'='*70}")
        print(f"UNIFIED OPERATOR MODE: norm={normalization_key}, scope={lift_scope}, sigma={sigma}")
        print(f"{'='*70}")
        print(f"Q(0): {Q_0:.6f}, Q(1): {Q_1:.6f}")
        print(f"Normalization: {norm_desc}")
        print(f"Q_shift(0) raw: {Q_lift_coeffs[0]:.6f}, normalized: {Q_lift_norm_coeffs[0]:.6f}")

    # Build terms
    ordered_plus = make_all_terms_k3_ordered(theta, R, kernel_regime="paper")
    ordered_minus = make_all_terms_k3_ordered(theta, -R, kernel_regime="paper")

    f_ordered = {
        "11": 1.0, "22": 0.25, "33": 1/36,
        "12": 0.5, "21": 0.5,
        "13": 1/6, "31": 1/6,
        "23": 1/12, "32": 1/12,
    }

    total = 0.0
    per_term = {}

    i1_plus_total = 0.0
    i1_minus_base = 0.0
    i1_minus_op = 0.0
    i2_plus_total = 0.0
    i2_minus_base = 0.0
    i2_minus_op = 0.0
    s34_plus_total = 0.0

    # Determine lifting scope
    apply_lift_to_i1 = lift_scope in ["both", "i1_only", "i1_left_only", "i1_right_only"]
    apply_lift_to_i2 = lift_scope in ["both", "i2_only"]
    i1_partial_mode = lift_scope in ["i1_left_only", "i1_right_only"]

    def _rename_i1_q_factors(term: Term, *, left_name: str, right_name: str) -> Term:
        q_indices = [i for i, pf in enumerate(term.poly_factors) if pf.poly_name == "Q"]
        if len(q_indices) != 2:
            raise ValueError(
                f"Expected exactly 2 Q factors in I1 term '{term.name}', found {len(q_indices)}"
            )
        poly_factors = list(term.poly_factors)
        poly_factors[q_indices[0]] = replace(poly_factors[q_indices[0]], poly_name=left_name)
        poly_factors[q_indices[1]] = replace(poly_factors[q_indices[1]], poly_name=right_name)
        return replace(term, poly_factors=poly_factors)

    # =========================================================================
    # PART 1: I₁
    # =========================================================================
    for pair_key in ["11", "22", "33", "12", "21", "13", "31", "23", "32"]:
        terms_plus = ordered_plus[pair_key]
        terms_minus = ordered_minus[pair_key]
        norm = f_ordered[pair_key] if use_factorial_normalization else 1.0

        term_plus = terms_plus[0]
        term_minus = terms_minus[0]

        result_plus = evaluate_term(
            term_plus, polys_base, n, R=R, theta=theta, n_quad_a=n_quad_a
        )
        result_minus_base = evaluate_term(
            term_minus, polys_base, n, R=-R, theta=theta, n_quad_a=n_quad_a
        )

        if apply_lift_to_i1:
            if i1_partial_mode:
                if lift_scope == "i1_left_only":
                    term_minus_op = _rename_i1_q_factors(term_minus, left_name="Q_lift", right_name="Q")
                else:
                    term_minus_op = _rename_i1_q_factors(term_minus, left_name="Q", right_name="Q_lift")
                result_minus_op = evaluate_term(
                    term_minus_op, polys_i1_partial, n, R=-R, theta=theta, n_quad_a=n_quad_a
                )
            else:
                result_minus_op = evaluate_term(
                    term_minus, polys_lift_full, n, R=-R, theta=theta, n_quad_a=n_quad_a
                )
        else:
            result_minus_op = result_minus_base

        val = result_plus.value + result_minus_op.value
        contrib = norm * val

        i1_plus_total += norm * result_plus.value
        i1_minus_base += norm * result_minus_base.value
        i1_minus_op += norm * result_minus_op.value
        total += contrib

    # =========================================================================
    # PART 2: I₂
    # =========================================================================
    for pair_key in ["11", "22", "33", "12", "21", "13", "31", "23", "32"]:
        terms_plus = ordered_plus[pair_key]
        terms_minus = ordered_minus[pair_key]
        norm = f_ordered[pair_key] if use_factorial_normalization else 1.0

        term_plus = terms_plus[1]
        term_minus = terms_minus[1]

        result_plus = evaluate_term(
            term_plus, polys_base, n, R=R, theta=theta, n_quad_a=n_quad_a
        )
        result_minus_base = evaluate_term(
            term_minus, polys_base, n, R=-R, theta=theta, n_quad_a=n_quad_a
        )

        if apply_lift_to_i2:
            result_minus_op = evaluate_term(
                term_minus, polys_lift_full, n, R=-R, theta=theta, n_quad_a=n_quad_a
            )
        else:
            result_minus_op = result_minus_base

        val = result_plus.value + result_minus_op.value
        contrib = norm * val

        i2_plus_total += norm * result_plus.value
        i2_minus_base += norm * result_minus_base.value
        i2_minus_op += norm * result_minus_op.value
        total += contrib

    # =========================================================================
    # PART 3: S34 (direct branch only)
    # =========================================================================
    for pair_key in ["11", "22", "33", "12", "21", "13", "31", "23", "32"]:
        terms = ordered_plus[pair_key]
        norm = f_ordered[pair_key] if use_factorial_normalization else 1.0

        for i in [2, 3]:
            term = terms[i]
            result = evaluate_term(
                term, polys_base, n, R=R, theta=theta, n_quad_a=n_quad_a
            )
            contrib = norm * result.value
            s34_plus_total += contrib
            total += contrib

    # Compute implied weights
    m1_implied = i1_minus_op / i1_minus_base if abs(i1_minus_base) > 1e-15 else float('inf')
    m2_implied = i2_minus_op / i2_minus_base if abs(i2_minus_base) > 1e-15 else float('inf')

    if verbose:
        print(f"\nChannels: I1+={i1_plus_total:.6f}, I1-base={i1_minus_base:.6f}, I1-op={i1_minus_op:.6f}")
        print(f"          I2+={i2_plus_total:.6f}, I2-base={i2_minus_base:.6f}, I2-op={i2_minus_op:.6f}")
        print(f"          S34+={s34_plus_total:.6f}")
        print(f"Implied:  m1={m1_implied:.4f}, m2={m2_implied:.4f}")

    # Store results
    per_term["_I1_plus"] = i1_plus_total
    per_term["_I1_minus_base"] = i1_minus_base
    per_term["_I1_minus_op"] = i1_minus_op
    per_term["_I2_plus"] = i2_plus_total
    per_term["_I2_minus_base"] = i2_minus_base
    per_term["_I2_minus_op"] = i2_minus_op
    per_term["_S34_plus"] = s34_plus_total
    per_term["_m1_implied"] = m1_implied
    per_term["_m2_implied"] = m2_implied
    per_term["_norm_factor"] = norm_factor
    per_term["_normalization"] = normalization_key
    per_term["_lift_scope"] = lift_scope
    per_term["_operator_shift_sigma"] = sigma
    per_term["_Q_0"] = Q_0
    per_term["_Q_1"] = Q_1
    per_term["_pair_mode"] = "ordered"
    per_term["_mirror_mode"] = f"unified_{normalization_key}_{lift_scope}"

    return EvaluationResult(
        total=total,
        per_term=per_term,
        n=n,
        term_results=None
    )


def solve_two_weight_operator(
    result_k: EvaluationResult,
    result_k_star: EvaluationResult,
    c_target_k: float,
    c_target_k_star: float,
    use_operator_channels: bool = True,
) -> Dict[str, float]:
    """
    Solve for (m₁, m₂) using both benchmarks together.

    This computes the operator-solved weights using the same 2×2 system
    approach as the baseline two-weight solver, but with operator-mode
    minus channels instead of base minus channels.

    The linear system is:
        c_target - (I1_plus + I2_plus + S34_plus) = m₁·I1_minus + m₂·I2_minus

    For each benchmark (κ and κ*), giving a 2×2 system.

    Args:
        result_k: EvaluationResult from κ benchmark (R=1.3036)
        result_k_star: EvaluationResult from κ* benchmark (R=1.1167)
        c_target_k: Target c for κ (2.137)
        c_target_k_star: Target c for κ* (1.938)
        use_operator_channels: If True, use I_minus_op; else use I_minus_base

    Returns:
        Dict with m1, m2, det, cond, and comparison ratios
    """
    import numpy as np

    # Extract channel values
    if use_operator_channels:
        I1_minus_k = result_k.per_term["_I1_minus_op"]
        I2_minus_k = result_k.per_term["_I2_minus_op"]
        I1_minus_ks = result_k_star.per_term["_I1_minus_op"]
        I2_minus_ks = result_k_star.per_term["_I2_minus_op"]
        mode = "operator"
    else:
        I1_minus_k = result_k.per_term["_I1_minus_base"]
        I2_minus_k = result_k.per_term["_I2_minus_base"]
        I1_minus_ks = result_k_star.per_term["_I1_minus_base"]
        I2_minus_ks = result_k_star.per_term["_I2_minus_base"]
        mode = "base"

    # Plus channels (same for both modes)
    I1_plus_k = result_k.per_term["_I1_plus"]
    I2_plus_k = result_k.per_term["_I2_plus"]
    S34_plus_k = result_k.per_term["_S34_plus"]
    I1_plus_ks = result_k_star.per_term["_I1_plus"]
    I2_plus_ks = result_k_star.per_term["_I2_plus"]
    S34_plus_ks = result_k_star.per_term["_S34_plus"]

    # RHS: what the minus channels need to provide
    rhs_k = c_target_k - (I1_plus_k + I2_plus_k + S34_plus_k)
    rhs_ks = c_target_k_star - (I1_plus_ks + I2_plus_ks + S34_plus_ks)

    # Build the 2×2 system: A @ [m1, m2]^T = b
    # Row 1: κ benchmark
    # Row 2: κ* benchmark
    A = np.array([
        [I1_minus_k, I2_minus_k],
        [I1_minus_ks, I2_minus_ks]
    ])
    b = np.array([rhs_k, rhs_ks])

    # Solve
    det = np.linalg.det(A)
    cond = np.linalg.cond(A)

    try:
        m = np.linalg.solve(A, b)
        m1, m2 = float(m[0]), float(m[1])
    except np.linalg.LinAlgError:
        m1, m2 = float('inf'), float('inf')

    return {
        "m1": m1,
        "m2": m2,
        "det": det,
        "cond": cond,
        "mode": mode,
        "rhs_k": rhs_k,
        "rhs_ks": rhs_ks,
    }


def evaluate_I2_separable(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict[str, PolyLike]
) -> Tuple[float, float, float]:
    """
    Evaluate I₂ and verify separability.

    I₂ is separable in u and t:
    - u-integrand: P₁(u)²
    - t-integrand: Q(t)² · exp(2R·t) · (1/θ)

    Returns both the 2D evaluation and the product of 1D integrals
    for comparison.

    Args:
        theta: θ parameter
        R: R parameter
        n: Number of quadrature points
        polynomials: Dict with "P1", "Q"

    Returns:
        Tuple of (2D_result, 1D_product, relative_error)
    """
    from src.terms_k3_d1 import make_I2_11
    from src.quadrature import gauss_legendre_01

    # 2D evaluation
    term = make_I2_11(theta, R)
    result_2d = evaluate_term(term, polynomials, n)
    val_2d = result_2d.value

    # 1D evaluation for comparison
    nodes, weights = gauss_legendre_01(n)

    P1 = polynomials["P1"]
    Q = polynomials["Q"]

    # u-integral: ∫ P₁(u)² du
    P1_vals = P1.eval(nodes)
    u_integral = np.sum(weights * P1_vals**2)

    # t-integral: ∫ Q(t)² · exp(2R·t) dt · (1/θ)
    Q_vals = Q.eval(nodes)
    exp_vals = np.exp(2 * R * nodes)
    t_integral = np.sum(weights * Q_vals**2 * exp_vals) / theta

    # Product should equal 2D result
    val_1d_product = u_integral * t_integral

    # Compute relative error
    if abs(val_1d_product) > 1e-15:
        rel_error = abs(val_2d - val_1d_product) / abs(val_1d_product)
    else:
        rel_error = abs(val_2d - val_1d_product)

    return val_2d, val_1d_product, rel_error


# =============================================================================
# Diagnostic report for discrepancy analysis
# =============================================================================

# PRZZ target values
PRZZ_C_TARGET = 2.13745440613217263636
PRZZ_KAPPA_TARGET = 0.417293962


@dataclass
class DiagnosticResult:
    """Complete diagnostic breakdown for debugging discrepancies."""
    # Summary
    c_computed: float
    c_target: float
    delta_c: float
    kappa_computed: float
    kappa_target: float
    delta_kappa: float

    # Per-pair raw values (before normalization)
    pair_raw: Dict[str, float]

    # Per-pair normalized values (after factorial and symmetry factors)
    pair_normalized: Dict[str, float]

    # Per-term breakdown (I₁, I₂, I₃, I₄ for each pair)
    per_term: Dict[str, float]

    # Normalization factors applied
    norm_factors: Dict[str, float]

    # Effective flags
    flags: Dict[str, Any]

    # Quadrature info
    n: int


def print_diagnostic_report(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict[str, PolyLike],
    use_factorial_normalization: bool = True,
    mode: str = "main",
    enforce_Q0: bool = True,
    kernel_regime: KernelRegime = "raw",
    n_quad_a: int = 40,
) -> DiagnosticResult:
    """
    Compute c and print detailed diagnostic report for discrepancy analysis.

    This prints:
    - Raw and normalized per-pair contributions
    - Per-pair per-term breakdown (I₁, I₂, I₃, I₄)
    - Normalization factor applied for each pair
    - Exact assembly formula used
    - Effective model flags
    - Totals and deltas vs target

    MODES:
    - "main": PRZZ asymptotic main term (I₅ FORBIDDEN) - DEFAULT
    - "with_error_terms": Includes I₅, prints warning

    KERNEL REGIMES:
    - "paper": TeX-driven regime where ω drives Case B/C selection and P₂/P₃
              are evaluated via Case C kernels.
    - "raw": Legacy/diagnostic regime using raw polynomials P(u ± x) everywhere.

    Args:
        theta: θ parameter (typically 4/7)
        R: R parameter (typically 1.3036)
        n: Number of quadrature points per dimension
        polynomials: Dict with keys "P1", "P2", "P3", "Q"
        use_factorial_normalization: If True, apply 1/(ℓ₁!×ℓ₂!) normalization
        mode: "main" (default) or "with_error_terms"
        enforce_Q0: The Q(0) normalization mode used for polynomials
        kernel_regime: "raw" (default) or "paper" for Case C kernel handling
        n_quad_a: Quadrature points for Case C a-integral

    Returns:
        DiagnosticResult with full breakdown
    """
    import math
    from src.terms_k3_d1 import make_all_terms_k3

    # Compute full c with breakdown
    all_terms = make_all_terms_k3(theta, R, kernel_regime=kernel_regime)

    # Factorial normalization factors
    factorial_norm = {
        "11": 1.0 / (math.factorial(1) * math.factorial(1)),  # 1
        "22": 1.0 / (math.factorial(2) * math.factorial(2)),  # 1/4
        "33": 1.0 / (math.factorial(3) * math.factorial(3)),  # 1/36
        "12": 1.0 / (math.factorial(1) * math.factorial(2)),  # 1/2
        "13": 1.0 / (math.factorial(1) * math.factorial(3)),  # 1/6
        "23": 1.0 / (math.factorial(2) * math.factorial(3)),  # 1/12
    }

    symmetry_factor = {
        "11": 1.0, "22": 1.0, "33": 1.0,
        "12": 2.0, "13": 2.0, "23": 2.0
    }

    # Evaluate each pair
    pair_raw = {}
    pair_normalized = {}
    per_term = {}
    norm_factors = {}
    total = 0.0

    for pair_key, terms in all_terms.items():
        pair_result = evaluate_terms(
            terms,
            polynomials,
            n,
            return_breakdown=True,
            R=R,
            theta=theta,
            n_quad_a=n_quad_a,
        )
        pair_raw[pair_key] = pair_result.total

        # Store per-term breakdown
        for term_name, term_val in pair_result.per_term.items():
            per_term[term_name] = term_val

        # Compute normalization factor
        if use_factorial_normalization:
            norm = factorial_norm[pair_key]
        else:
            norm = 1.0

        sym = symmetry_factor[pair_key]
        norm_factors[pair_key] = sym * norm

        # Normalized contribution
        pair_normalized[pair_key] = sym * norm * pair_result.total
        total += pair_normalized[pair_key]

    # I₅ arithmetic correction (ERROR TERM - diagnostic only)
    # PRZZ TeX lines 1626-1628: I₅ ≪ T/L
    # Formula: I₅ = -S(0) × θ²/12 × I₂_total
    i5_contrib_total = 0.0
    i2_sum_for_i5 = 0.0
    if mode == "with_error_terms":
        import warnings
        warnings.warn(
            "mode='with_error_terms' includes I₅ - DIAGNOSTIC ONLY",
            UserWarning,
            stacklevel=2
        )

        from src.arithmetic_constants import S_AT_ZERO

        # Compute I₂ sum
        for pair_key, terms in all_terms.items():
            i2_term = terms[1]
            i2_result = evaluate_term(i2_term, polynomials, n, R=R, theta=theta, n_quad_a=n_quad_a)

            sym = symmetry_factor[pair_key]
            if use_factorial_normalization:
                norm = factorial_norm[pair_key]
            else:
                norm = 1.0

            i2_sum_for_i5 += sym * norm * i2_result.value

        # I₅ = -S(0) × θ²/12 × I₂_total
        i5_contrib_total = -S_AT_ZERO * (theta ** 2 / 12.0) * i2_sum_for_i5
        total += i5_contrib_total
        per_term["_I5_total"] = i5_contrib_total

    # Compute κ
    kappa = compute_kappa(total, R)

    # Build result
    result = DiagnosticResult(
        c_computed=total,
        c_target=PRZZ_C_TARGET,
        delta_c=total - PRZZ_C_TARGET,
        kappa_computed=kappa,
        kappa_target=PRZZ_KAPPA_TARGET,
        delta_kappa=kappa - PRZZ_KAPPA_TARGET,
        pair_raw=pair_raw,
        pair_normalized=pair_normalized,
        per_term=per_term,
        norm_factors=norm_factors,
        flags={
            "use_factorial_normalization": use_factorial_normalization,
            "mode": mode,
            "enforce_Q0": enforce_Q0,
            "kernel_regime": kernel_regime,
            "theta": theta,
            "R": R,
        },
        n=n
    )

    # Print report
    print("\n" + "=" * 70)
    print("PRZZ DIAGNOSTIC REPORT")
    print("=" * 70)

    print("\n--- Effective Model Flags ---")
    print(f"  use_factorial_normalization: {use_factorial_normalization}")
    print(f"  mode:                        {mode}")
    print(f"  enforce_Q0:                  {enforce_Q0}")
    print(f"  kernel_regime:               {kernel_regime}")
    print(f"  theta:                       {theta:.10f}")
    print(f"  R:                           {R}")
    print(f"  n (quadrature):              {n}")

    print("\n--- Raw Per-Pair Values (before normalization) ---")
    for pair in ["11", "22", "33", "12", "13", "23"]:
        print(f"  c_{pair}_raw:  {pair_raw[pair]:+18.12f}")

    print("\n--- Normalization Factors Applied ---")
    for pair in ["11", "22", "33", "12", "13", "23"]:
        fact_str = f"1/{math.factorial(int(pair[0])) * math.factorial(int(pair[1]))}"
        sym_str = "×2" if pair in ("12", "13", "23") else "×1"
        print(f"  {pair}: {fact_str:>6s} {sym_str} = {norm_factors[pair]:.6f}")

    print("\n--- Normalized Per-Pair Values ---")
    for pair in ["11", "22", "33", "12", "13", "23"]:
        print(f"  c_{pair}_norm: {pair_normalized[pair]:+18.12f}")

    print("\n--- Assembly Formula ---")
    formula_parts = []
    for pair in ["11", "22", "33", "12", "13", "23"]:
        if pair in ("12", "13", "23"):
            formula_parts.append(f"2×c_{pair}/({math.factorial(int(pair[0]))}!×{math.factorial(int(pair[1]))}!)")
        else:
            formula_parts.append(f"c_{pair}/({math.factorial(int(pair[0]))}!×{math.factorial(int(pair[1]))}!)")
    print(f"  c = {' + '.join(formula_parts)}")

    print("\n--- Per-Term Breakdown (by integral type) ---")
    for pair in ["11", "22", "33", "12", "13", "23"]:
        print(f"  Pair ({pair[0]},{pair[1]}):")
        for i in range(1, 5):  # I₁-I₄
            term_name = f"I{i}_{pair}"
            if term_name in per_term:
                print(f"    I{i}: {per_term[term_name]:+18.12f}")
            else:
                print(f"    I{i}: (not computed)")
        # I₅ (arithmetic correction - only in with_error_terms mode)
        term_name = f"I5_{pair}"
        if term_name in per_term:
            print(f"    I₅: {per_term[term_name]:+18.12f} (S correction)")
        elif mode == "with_error_terms":
            print(f"    I₅: (not computed)")

    if mode == "with_error_terms":
        from src.arithmetic_constants import S_AT_ZERO as S0
        print(f"\n--- I₅ Arithmetic Correction (DIAGNOSTIC ONLY) ---")
        print(f"  WARNING: I₅ ≪ T/L per PRZZ lines 1626-1628")
        print(f"  Formula: I₅ = -S(0) × θ²/12 × I₂_total")
        print(f"  S(0) = {S0:.10f}")
        print(f"  θ² = {theta**2:.10f}")
        print(f"  θ²/12 = {theta**2/12:.10f}")
        print(f"  I₂_total (normalized) = {i2_sum_for_i5:+18.12f}")
        print(f"  I₅ correction = {i5_contrib_total:+18.12f}")
    else:
        print(f"\n--- I₅ Status ---")
        print(f"  mode='main': I₅ EXCLUDED (correct per PRZZ)")

    print("\n--- Summary ---")
    print(f"  c_computed:    {total:20.15f}")
    print(f"  c_target:      {PRZZ_C_TARGET:20.15f}")
    print(f"  Δc:            {total - PRZZ_C_TARGET:+20.15f}")
    print(f"  Δc relative:   {(total - PRZZ_C_TARGET) / PRZZ_C_TARGET * 100:+.6f}%")
    print()
    print(f"  κ_computed:    {kappa:20.15f}")
    print(f"  κ_target:      {PRZZ_KAPPA_TARGET:20.15f}")
    print(f"  Δκ:            {kappa - PRZZ_KAPPA_TARGET:+20.15f}")

    print("\n" + "=" * 70)

    return result


# =============================================================================
# ORDERED-PAIR EVALUATION (Priority A - GPT Guidance 2025-12-19)
# =============================================================================

def evaluate_c_ordered(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict[str, PolyLike],
    *,
    return_breakdown: bool = True,
    use_factorial_normalization: bool = True,
    kernel_regime: KernelRegime = "paper",
    mode: str = "main",
    n_quad_a: int = 40,
    terms_version: str = "old",  # Run 10A: "old" or "v2"
) -> EvaluationResult:
    """
    Evaluate c by summing ALL 9 ordered pairs directly.

    NO symmetry factor applied - each pair counted once with factorial norm only.

    This serves as ground truth for validating the triangle×2 assumption.

    c = sum_{ell1, ell2 in {1,2,3}} [1/(ell1! * ell2!)] * c_{ell1,ell2}

    Pairs: 11, 12, 13, 21, 22, 23, 31, 32, 33

    Args:
        theta: θ parameter (typically 4/7)
        R: R parameter (typically 1.3036)
        n: Number of quadrature points per dimension
        polynomials: Dict with keys "P1", "P2", "P3", "Q"
        use_factorial_normalization: If True, apply 1/(ℓ₁!×ℓ₂!) normalization
        kernel_regime: "raw" or "paper" for Case C kernel handling
        n_quad_a: Quadrature points for Case C a-integral

    Returns:
        EvaluationResult with full c from ordered-sum approach
    """
    if mode not in ("main", "with_error_terms"):
        raise ValueError(f"mode must be 'main' or 'with_error_terms', got '{mode}'")

    # Run 10A: Dispatch based on terms_version
    if terms_version == "v2":
        from src.terms_k3_d1 import make_all_terms_k3_ordered_v2 as make_all_terms_k3_ordered
    else:
        from src.terms_k3_d1 import make_all_terms_k3_ordered
    import math

    # Get all 9 ordered pairs
    all_ordered_terms = make_all_terms_k3_ordered(theta, R, kernel_regime=kernel_regime)

    # Factorial normalization for all 9 pairs (NO symmetry factor)
    factorial_norm = {
        "11": 1.0 / (math.factorial(1) * math.factorial(1)),  # 1/1 = 1.0
        "22": 1.0 / (math.factorial(2) * math.factorial(2)),  # 1/4 = 0.25
        "33": 1.0 / (math.factorial(3) * math.factorial(3)),  # 1/36
        "12": 1.0 / (math.factorial(1) * math.factorial(2)),  # 1/2 = 0.5
        "21": 1.0 / (math.factorial(2) * math.factorial(1)),  # 1/2 = 0.5
        "13": 1.0 / (math.factorial(1) * math.factorial(3)),  # 1/6
        "31": 1.0 / (math.factorial(3) * math.factorial(1)),  # 1/6
        "23": 1.0 / (math.factorial(2) * math.factorial(3)),  # 1/12
        "32": 1.0 / (math.factorial(3) * math.factorial(2)),  # 1/12
    }

    # Evaluate each pair
    pair_results = {}
    total = 0.0
    all_per_term: Dict[str, Any] = {}
    i2_total = 0.0

    for pair_key, terms in all_ordered_terms.items():
        pair_result = evaluate_terms(
            terms,
            polynomials,
            n,
            return_breakdown=return_breakdown,
            R=R,
            theta=theta,
            n_quad_a=n_quad_a,
        )
        pair_results[pair_key] = pair_result.total

        # Apply factorial normalization if requested (NO symmetry factor)
        if use_factorial_normalization:
            norm = factorial_norm[pair_key]
        else:
            norm = 1.0

        total += norm * pair_result.total

        if mode == "with_error_terms":
            # I2 is the second term (index 1) in each pair.
            i2_val = evaluate_term(
                terms[1], polynomials, n, R=R, theta=theta, n_quad_a=n_quad_a
            ).value
            i2_total += norm * i2_val

        # Add per-term breakdown with pair prefix
        if return_breakdown and pair_result.per_term is not None:
            for term_name, term_val in pair_result.per_term.items():
                all_per_term[f"{pair_key}_{term_name}"] = term_val

    # Store raw and normalized pair values
    all_per_term["_ordered_pair_raw"] = pair_results
    all_per_term["_ordered_pair_normalized"] = {
        pair_key: factorial_norm[pair_key] * val if use_factorial_normalization else val
        for pair_key, val in pair_results.items()
    }
    all_per_term["_pair_mode"] = "ordered"
    all_per_term["_kernel_regime"] = kernel_regime
    all_per_term["_use_factorial_normalization"] = float(bool(use_factorial_normalization))

    if mode == "with_error_terms":
        import warnings
        from src.arithmetic_constants import S_AT_ZERO

        warnings.warn(
            "\n" + "=" * 60 + "\n"
            "WARNING: mode='with_error_terms' includes I₅.\n"
            "I₅ is O(T/L), NOT part of PRZZ's main constant.\n"
            "PRZZ TeX lines 1626-1628: 'I₅ ≪ T/L'\n"
            "Do NOT use this mode for golden target matching!\n"
            + "=" * 60,
            UserWarning,
            stacklevel=2,
        )

        i5_correction = -S_AT_ZERO * (theta ** 2 / 12.0) * i2_total
        total += i5_correction
        all_per_term["_I5_total"] = i5_correction
        all_per_term["_I2_sum_for_I5"] = i2_total

    return EvaluationResult(
        total=total,
        per_term=all_per_term,
        n=n,
        term_results=None
    )


def compare_triangle_vs_ordered(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict[str, PolyLike],
    *,
    kernel_regime: KernelRegime = "paper",
    n_quad_a: int = 40,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Diagnostic: compare triangle×2 vs ordered-sum evaluation.

    The identity being tested:
        C_triangle - C_ordered = f(12)*(S_12-S_21) + f(13)*(S_13-S_31) + f(23)*(S_23-S_32)

    Where:
        S_pq = I1_pq + I2_pq + I3_pq + I4_pq  (total for ordered pair)
        S12_pq = I1_pq + I2_pq  (block 12)
        S34_pq = I3_pq + I4_pq  (block 34)
        f(pq) = 1/(ℓ_p! × ℓ_q!)  (factorial norm)

    If S_pq == S_qp for all off-diagonal pairs, then triangle×2 == ordered.

    Returns dict with:
    - 'C_triangle', 'C_ordered', 'delta', 'delta_rel'
    - 'off_diagonal': detailed breakdown for {12,21}, {13,31}, {23,32}
    """
    from src.terms_k3_d1 import make_all_terms_k3_ordered
    import math

    # Factorial norms
    f = {
        "11": 1.0, "22": 0.25, "33": 1/36,
        "12": 0.5, "21": 0.5,
        "13": 1/6, "31": 1/6,
        "23": 1/12, "32": 1/12,
    }

    # Get all ordered terms and evaluate each pair's I1,I2,I3,I4
    ordered_terms = make_all_terms_k3_ordered(theta, R, kernel_regime=kernel_regime)

    pair_values = {}  # {pair: {"I1": val, "I2": val, "I3": val, "I4": val, "S": total, "S12": block, "S34": block}}

    for pair_key, terms in ordered_terms.items():
        i1 = evaluate_term(terms[0], polynomials, n, R=R, theta=theta, n_quad_a=n_quad_a).value
        i2 = evaluate_term(terms[1], polynomials, n, R=R, theta=theta, n_quad_a=n_quad_a).value
        i3 = evaluate_term(terms[2], polynomials, n, R=R, theta=theta, n_quad_a=n_quad_a).value
        i4 = evaluate_term(terms[3], polynomials, n, R=R, theta=theta, n_quad_a=n_quad_a).value

        S = i1 + i2 + i3 + i4
        S12 = i1 + i2
        S34 = i3 + i4

        pair_values[pair_key] = {
            "I1": i1, "I2": i2, "I3": i3, "I4": i4,
            "S": S, "S12": S12, "S34": S34,
        }

    # Compute totals
    C_ordered = sum(f[p] * pair_values[p]["S"] for p in pair_values)

    # Triangle: diagonal + 2×upper-triangle off-diagonal
    C_triangle = (
        f["11"] * pair_values["11"]["S"] +
        f["22"] * pair_values["22"]["S"] +
        f["33"] * pair_values["33"]["S"] +
        2 * f["12"] * pair_values["12"]["S"] +
        2 * f["13"] * pair_values["13"]["S"] +
        2 * f["23"] * pair_values["23"]["S"]
    )

    delta = C_triangle - C_ordered
    delta_rel = delta / C_ordered if C_ordered != 0 else float('inf')

    # Verify identity: delta should equal sum of f(pq)*(S_pq - S_qp)
    identity_check = (
        f["12"] * (pair_values["12"]["S"] - pair_values["21"]["S"]) +
        f["13"] * (pair_values["13"]["S"] - pair_values["31"]["S"]) +
        f["23"] * (pair_values["23"]["S"] - pair_values["32"]["S"])
    )

    # Off-diagonal breakdown
    off_diagonal = {}
    for pq, qp in [("12", "21"), ("13", "31"), ("23", "32")]:
        pv_pq = pair_values[pq]
        pv_qp = pair_values[qp]

        off_diagonal[f"{pq}_vs_{qp}"] = {
            "S_pq": pv_pq["S"],
            "S_qp": pv_qp["S"],
            "S12_pq": pv_pq["S12"],
            "S12_qp": pv_qp["S12"],
            "S34_pq": pv_pq["S34"],
            "S34_qp": pv_qp["S34"],
            # Mismatches
            "delta_S": pv_pq["S"] - pv_qp["S"],
            "delta_S12": pv_pq["S12"] - pv_qp["S12"],
            "delta_S34": pv_pq["S34"] - pv_qp["S34"],
            # Factorial-weighted contribution to total delta
            "f_weighted_delta": f[pq] * (pv_pq["S"] - pv_qp["S"]),
        }

    result = {
        "C_triangle": C_triangle,
        "C_ordered": C_ordered,
        "delta": delta,
        "delta_rel": delta_rel,
        "identity_check": identity_check,  # Should equal delta
        "off_diagonal": off_diagonal,
        "pair_values": pair_values,
        "kernel_regime": kernel_regime,
    }

    if verbose:
        print("\n" + "=" * 80)
        print(f"TRIANGLE×2 vs ORDERED-SUM COMPARISON ({kernel_regime} regime)")
        print("=" * 80)
        print(f"  C_triangle:    {C_triangle:+.12f}")
        print(f"  C_ordered:     {C_ordered:+.12f}")
        print(f"  delta:         {delta:+.12e}")
        print(f"  delta_rel:     {delta_rel:+.6e}")
        print(f"  identity_chk:  {identity_check:+.12e}  (should == delta)")
        print()
        print("Off-diagonal breakdown (S = I1+I2+I3+I4):")
        print("-" * 80)
        for key, d in off_diagonal.items():
            pq, qp = key.split("_vs_")
            print(f"  {key}:")
            print(f"    S_{pq}={d['S_pq']:+.8e}  S_{qp}={d['S_qp']:+.8e}  Δ_S={d['delta_S']:+.6e}")
            print(f"    S12_{pq}={d['S12_pq']:+.8e}  S12_{qp}={d['S12_qp']:+.8e}  Δ_S12={d['delta_S12']:+.6e}")
            print(f"    S34_{pq}={d['S34_pq']:+.8e}  S34_{qp}={d['S34_qp']:+.8e}  Δ_S34={d['delta_S34']:+.6e}")
        print("=" * 80)

    return result


def evaluate_c_ordered_with_exp_transform(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict[str, PolyLike],
    *,
    kernel_regime: KernelRegime = "paper",
    exp_scale_multiplier: float = 1.0,
    exp_t_flip: bool = False,
    q_a0_shift: float = 0.0,
    use_factorial_normalization: bool = True,
    n_quad_a: int = 40,
) -> EvaluationResult:
    """
    Evaluate c using ordered-sum with exp-factor transform (for mirror diagnostics).

    Like evaluate_c_ordered() but applies transforms for mirror analysis:
    - exp_scale_multiplier: Scales ExpFactor coefficients (e.g., -1 for sign flip)
    - exp_t_flip: Maps t -> 1-t inside ExpFactor arguments
    - q_a0_shift: Shifts Q(...) AffineExpr constant term

    Returns all 9 ordered pairs without symmetry factor.
    """
    from src.terms_k3_d1 import make_all_terms_k3_ordered
    from src.mirror_transform import transform_terms_exp_factors, transform_terms_q_factors
    import math

    t_map = (lambda T: 1.0 - T) if exp_t_flip else None

    all_ordered_terms = make_all_terms_k3_ordered(theta, R, kernel_regime=kernel_regime)

    factorial_norm = {
        "11": 1.0 / (math.factorial(1) * math.factorial(1)),
        "22": 1.0 / (math.factorial(2) * math.factorial(2)),
        "33": 1.0 / (math.factorial(3) * math.factorial(3)),
        "12": 1.0 / (math.factorial(1) * math.factorial(2)),
        "21": 1.0 / (math.factorial(2) * math.factorial(1)),
        "13": 1.0 / (math.factorial(1) * math.factorial(3)),
        "31": 1.0 / (math.factorial(3) * math.factorial(1)),
        "23": 1.0 / (math.factorial(2) * math.factorial(3)),
        "32": 1.0 / (math.factorial(3) * math.factorial(2)),
    }

    pair_results = {}
    total = 0.0
    all_per_term: Dict[str, float] = {}

    for pair_key, terms in all_ordered_terms.items():
        # Apply exp transforms
        terms_x = transform_terms_exp_factors(
            terms, scale_multiplier=exp_scale_multiplier, t_map=t_map
        )
        if q_a0_shift != 0.0:
            terms_x = transform_terms_q_factors(terms_x, q_a0_shift=q_a0_shift)

        pair_result = evaluate_terms(
            terms_x,
            polynomials,
            n,
            return_breakdown=True,
            R=R,
            theta=theta,
            n_quad_a=n_quad_a,
        )
        pair_results[pair_key] = pair_result.total

        # NO symmetry factor - each pair counted once
        norm = factorial_norm[pair_key] if use_factorial_normalization else 1.0
        total += norm * pair_result.total

        # Add per-term breakdown with pair prefix
        for term_name, term_val in pair_result.per_term.items():
            all_per_term[f"{pair_key}_{term_name}"] = term_val

    # Store raw and normalized pair values
    all_per_term["_ordered_pair_raw"] = pair_results
    all_per_term["_ordered_pair_normalized"] = {
        pair_key: factorial_norm[pair_key] * val if use_factorial_normalization else val
        for pair_key, val in pair_results.items()
    }
    all_per_term["_exp_scale_multiplier"] = float(exp_scale_multiplier)
    all_per_term["_exp_t_flip"] = float(bool(exp_t_flip))
    all_per_term["_q_a0_shift"] = float(q_a0_shift)

    return EvaluationResult(
        total=total,
        per_term=all_per_term,
        n=n,
        term_results=None
    )


# =============================================================================
# OPERATOR MODE V2: Comprehensive Q-shift with factor localization
# =============================================================================
# Implements Codex Tasks 1-4 from GPT guidance 2025-12-20

def compute_c_paper_operator_v2(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict[str, PolyLike],
    *,
    use_factorial_normalization: bool = True,
    n_quad_a: int = 40,
    verbose: bool = False,
    normalization: str = "none",
    lift_scope: str = "i1_only",
    sigma: float = 1.0,
    allow_unstable: bool = False,
    i1_source: str = "dsl",  # GPT Phase 1: "dsl" (default) or "post_identity_operator"
    i2_source: str = "dsl",  # Run 8A: "dsl" (default) or "direct_case_c" (proven)
    terms_version: str = "old",  # Run 10A: "old" or "v2"
) -> EvaluationResult:
    """
    Operator mode V2 with factor localization and configurable shift magnitude.

    This is the comprehensive operator evaluator supporting:
    - Factor localization: Apply lift to specific Q factors in I₁
    - Configurable shift magnitude: Q(x+sigma) instead of hardcoded Q(1+x)
    - Safe normalizations with stability checks

    Args:
        theta: θ parameter (typically 4/7)
        R: R parameter
        n: Number of quadrature points per dimension
        polynomials: Dict with keys "P1", "P2", "P3", "Q"
        use_factorial_normalization: Apply 1/(ℓ₁!×ℓ₂!) normalization
        n_quad_a: Quadrature points for Case C a-integral
        verbose: Print diagnostic information
        normalization: How to normalize Q_lift:
            - "none": No normalization (Q → Q(x+sigma))
            - "l2": Scale so ∫Q²dt = ∫Q_lift²dt
            - "grid": Scale so mean(Q²) = mean(Q_lift²) on quadrature grid
            - "q1_ratio": Q(0)/Q(1) scaling [UNSTABLE - requires allow_unstable]
        lift_scope: Which Q factors to apply Q-lift to:
            - "both": Apply to both I₁ and I₂ (all Q factors)
            - "i1_only": Apply to both Q factors in I₁ only
            - "i2_only": Apply to Q² factor in I₂ only
            - "i1_left_only": Apply only to Q(Arg_α) in I₁
            - "i1_right_only": Apply only to Q(Arg_β) in I₁
        sigma: Shift magnitude (default 1.0 for Q(1+x))
        allow_unstable: Allow unstable normalizations like q1_ratio
        i1_source: Source for I1 computation:
            - "dsl" (default): Use DSL-based term evaluation
            - "post_identity_operator": Use post-identity operator approach (GPT Phase 1)
        i2_source: Source for I2 computation:
            - "dsl" (default): Use DSL-based term evaluation
            - "direct_case_c": Use proven Case C kernel evaluation (Run 7/8)

    Returns:
        EvaluationResult with c and detailed channel breakdown
    """
    from dataclasses import replace
    # Run 10A: Dispatch based on terms_version
    if terms_version == "v2":
        from src.terms_k3_d1 import make_all_terms_k3_ordered_v2 as make_all_terms_k3_ordered
    else:
        from src.terms_k3_d1 import make_all_terms_k3_ordered
    from src.q_operator import binomial_shift_coeffs
    from src.polynomials import Polynomial
    from src.quadrature import gauss_legendre_01
    import warnings

    # =========================================================================
    # STEP 0: Validate inputs
    # =========================================================================
    valid_scopes = ["both", "i1_only", "i2_only", "i1_left_only", "i1_right_only"]
    if lift_scope not in valid_scopes:
        raise ValueError(f"Invalid lift_scope '{lift_scope}'. Valid: {valid_scopes}")

    # sigma=0 should reproduce base exactly (identity check)
    is_identity_mode = abs(sigma) < 1e-15

    # =========================================================================
    # STEP 1: Create Q_lift polynomial with appropriate shift and normalization
    # =========================================================================
    normalization_key = str(normalization).strip().lower()
    normalization_aliases = {
        "none": "none",
        "l2": "l2",
        "l2norm": "l2",
        "grid": "grid",
        "gridnorm": "grid",
        "q1_ratio": "q1_ratio",
        "norm_q1": "q1_ratio",
        "operator_q_shift_norm_q1": "q1_ratio",
    }
    try:
        normalization_key = normalization_aliases[normalization_key]
    except KeyError as exc:
        raise ValueError(f"Unknown normalization: {normalization}") from exc

    Q = polynomials["Q"]

    # Convert to monomial coefficients
    if hasattr(Q, "to_monomial"):
        Q_mono = Q.to_monomial()
        Q_coeffs = list(np.asarray(Q_mono.coeffs, dtype=float))
    else:
        Q_coeffs = list(np.asarray(Q.coeffs, dtype=float))

    # Apply binomial shift with sigma
    if is_identity_mode:
        Q_lift_coeffs = Q_coeffs.copy()  # Identity: no shift
    else:
        Q_lift_coeffs = binomial_shift_coeffs(Q_coeffs, shift=sigma)

    # Compute key Q values for diagnostics and normalization
    Q_0 = Q_coeffs[0] if Q_coeffs else 1.0
    Q_at_sigma = sum(c * (sigma ** j) for j, c in enumerate(Q_coeffs))
    Q_1 = sum(Q_coeffs) if Q_coeffs else 1.0  # Q(1)

    nodes, weights = gauss_legendre_01(n)

    # Evaluate Q and Q_lift on grid
    Q_vals = np.array([sum(c * (t ** j) for j, c in enumerate(Q_coeffs)) for t in nodes])
    Q_lift_vals = np.array([sum(c * (t ** j) for j, c in enumerate(Q_lift_coeffs)) for t in nodes])

    # Compute normalization factor based on mode
    if normalization_key == "none":
        norm_factor = 1.0
        norm_desc = "none"
    elif normalization_key == "l2":
        # L2 normalization: sqrt(∫Q²dt / ∫Q_lift²dt)
        q_l2_sq = np.sum(weights * Q_vals**2)
        q_lift_l2_sq = np.sum(weights * Q_lift_vals**2)
        if q_lift_l2_sq > 1e-15:
            norm_factor = np.sqrt(q_l2_sq / q_lift_l2_sq)
        else:
            norm_factor = 1.0
        norm_desc = f"L2 (factor={norm_factor:.6f})"
    elif normalization_key == "grid":
        # Grid normalization: sqrt(mean(Q²) / mean(Q_lift²))
        q_mean_sq = np.mean(Q_vals**2)
        q_lift_mean_sq = np.mean(Q_lift_vals**2)
        if q_lift_mean_sq > 1e-15:
            norm_factor = np.sqrt(q_mean_sq / q_lift_mean_sq)
        else:
            norm_factor = 1.0
        norm_desc = f"grid (factor={norm_factor:.6f})"
    elif normalization_key == "q1_ratio":
        # Q(0)/Q(1) - QUARANTINED negative control (NOT TeX-justified).
        if not allow_unstable:
            raise ValueError(
                "q1_ratio normalization is quarantined (negative control; unstable when Q(1)≈0). "
                "Pass allow_unstable=True to use it explicitly."
            )
        warnings.warn(
            f"q1_ratio normalization uses Q(0)/Q(1) with Q(1)={Q_1:+.6f}; "
            "this can be extremely unstable and is intended as a negative control only.",
            RuntimeWarning,
            stacklevel=2,
        )
        if abs(Q_1) > 1e-15:
            norm_factor = Q_0 / Q_1
        else:
            norm_factor = 1.0
        norm_desc = f"q1_ratio [UNSTABLE] (factor={norm_factor:.6f})"

    # Apply normalization
    Q_lift_norm_coeffs = [c * norm_factor for c in Q_lift_coeffs]
    Q_lift_norm = Polynomial(np.asarray(Q_lift_norm_coeffs, dtype=float))

    # =========================================================================
    # STEP 2: Create polynomial dicts for different configurations
    # =========================================================================
    polys_base = polynomials

    # Full lift (both Q factors lifted)
    polys_lift_full = {
        "P1": polynomials["P1"],
        "P2": polynomials["P2"],
        "P3": polynomials["P3"],
        "Q": Q_lift_norm,
    }
    # Partial lift for I₁ left/right factor localization
    polys_lift_partial = dict(polys_base)
    polys_lift_partial["Q_lift"] = Q_lift_norm

    if verbose:
        print(f"\n{'='*70}")
        print(f"OPERATOR MODE V2: norm={normalization_key}, scope={lift_scope}, sigma={sigma:.4f}")
        print(f"{'='*70}")
        print(f"Q(0): {Q_0:.6f}, Q({sigma:.2f}): {Q_at_sigma:.6f}, Q(1): {Q_1:.6f}")
        print(f"Normalization: {norm_desc}")
        print(f"Q_lift(0) raw: {Q_lift_coeffs[0]:.6f}, normalized: {Q_lift_norm_coeffs[0]:.6f}")
        if is_identity_mode:
            print("IDENTITY MODE: sigma=0, expecting base reproduction")

    # Build terms
    ordered_plus = make_all_terms_k3_ordered(theta, R, kernel_regime="paper")
    ordered_minus = make_all_terms_k3_ordered(theta, -R, kernel_regime="paper")

    f_ordered = {
        "11": 1.0, "22": 0.25, "33": 1/36,
        "12": 0.5, "21": 0.5,
        "13": 1/6, "31": 1/6,
        "23": 1/12, "32": 1/12,
    }

    total = 0.0
    per_term = {}

    # Channel accumulators
    i1_plus_total = 0.0
    i1_minus_base = 0.0
    i1_minus_op = 0.0
    i2_plus_total = 0.0
    i2_minus_base = 0.0
    i2_minus_op = 0.0
    s34_plus_total = 0.0

    # Per-pair breakdown for localization (weighted in "c-units" via factorial norm)
    pair_breakdown: Dict[str, Dict[str, float]] = {}

    # Determine lifting scope
    apply_lift_to_i1 = lift_scope in ["both", "i1_only", "i1_left_only", "i1_right_only"]
    apply_lift_to_i2 = lift_scope in ["both", "i2_only"]
    i1_partial_mode = lift_scope in ["i1_left_only", "i1_right_only"]

    def _rename_i1_q_factors(term, *, left_name: str, right_name: str):
        q_indices = [i for i, pf in enumerate(term.poly_factors) if pf.poly_name == "Q"]
        if len(q_indices) != 2:
            raise ValueError(
                f"Expected exactly 2 Q factors in I1 term '{term.name}', found {len(q_indices)}"
            )
        poly_factors = list(term.poly_factors)
        poly_factors[q_indices[0]] = replace(poly_factors[q_indices[0]], poly_name=left_name)
        poly_factors[q_indices[1]] = replace(poly_factors[q_indices[1]], poly_name=right_name)
        return replace(term, poly_factors=poly_factors)

    # =========================================================================
    # PART 1: I₁ (terms[0] for each pair)
    # =========================================================================
    # GPT Phase 1: Support both DSL and post-identity operator sources
    if i1_source == "post_identity_operator":
        # Use post-identity operator evaluation (GPT Phase 1)
        from src.operator_post_identity import compute_I1_operator_post_identity_pair

        for pair_key in ["11", "22", "33", "12", "21", "13", "31", "23", "32"]:
            ell1, ell2 = int(pair_key[0]), int(pair_key[1])
            norm = f_ordered[pair_key] if use_factorial_normalization else 1.0

            # Post-identity gives I1 at +R and -R
            result_plus = compute_I1_operator_post_identity_pair(
                theta, R, ell1, ell2, n, polys_base
            )
            result_minus = compute_I1_operator_post_identity_pair(
                theta, -R, ell1, ell2, n, polys_base
            )

            i1_plus_raw = result_plus.I1_value
            i1_minus_base_raw = result_minus.I1_value

            # For post_identity_operator, skip Q-lift (like direct_case_c does for I2)
            # The post-identity operator doesn't support Q-lift
            i1_minus_op_raw = i1_minus_base_raw

            # Store weighted per-pair channels
            pair = pair_breakdown.setdefault(pair_key, {"weight": float(norm)})
            pair["I1_plus"] = float(norm * i1_plus_raw)
            pair["I1_minus_base"] = float(norm * i1_minus_base_raw)
            pair["I1_minus_op"] = float(norm * i1_minus_op_raw)
            pair["I1_plus_raw"] = float(i1_plus_raw)
            pair["I1_minus_base_raw"] = float(i1_minus_base_raw)
            pair["I1_minus_op_raw"] = float(i1_minus_op_raw)

            val = i1_plus_raw + i1_minus_op_raw
            contrib = norm * val

            i1_plus_total += norm * i1_plus_raw
            i1_minus_base += norm * i1_minus_base_raw
            i1_minus_op += norm * i1_minus_op_raw
            total += contrib

    else:
        # Default: Use DSL-based term evaluation
        for pair_key in ["11", "22", "33", "12", "21", "13", "31", "23", "32"]:
            terms_plus = ordered_plus[pair_key]
            terms_minus = ordered_minus[pair_key]
            norm = f_ordered[pair_key] if use_factorial_normalization else 1.0

            term_plus = terms_plus[0]
            term_minus = terms_minus[0]

            result_plus = evaluate_term(
                term_plus, polys_base, n, R=R, theta=theta, n_quad_a=n_quad_a
            )
            result_minus_base = evaluate_term(
                term_minus, polys_base, n, R=-R, theta=theta, n_quad_a=n_quad_a
            )

            if apply_lift_to_i1:
                if i1_partial_mode:
                    if lift_scope == "i1_left_only":
                        term_minus_op = _rename_i1_q_factors(term_minus, left_name="Q_lift", right_name="Q")
                    else:
                        term_minus_op = _rename_i1_q_factors(term_minus, left_name="Q", right_name="Q_lift")
                    result_minus_op = evaluate_term(
                        term_minus_op, polys_lift_partial, n, R=-R, theta=theta, n_quad_a=n_quad_a
                    )
                else:
                    result_minus_op = evaluate_term(
                        term_minus, polys_lift_full, n, R=-R, theta=theta, n_quad_a=n_quad_a
                    )
            else:
                result_minus_op = result_minus_base

            # Store weighted per-pair channels
            pair = pair_breakdown.setdefault(pair_key, {"weight": float(norm)})
            pair["I1_plus"] = float(norm * result_plus.value)
            pair["I1_minus_base"] = float(norm * result_minus_base.value)
            pair["I1_minus_op"] = float(norm * result_minus_op.value)
            pair["I1_plus_raw"] = float(result_plus.value)
            pair["I1_minus_base_raw"] = float(result_minus_base.value)
            pair["I1_minus_op_raw"] = float(result_minus_op.value)

            val = result_plus.value + result_minus_op.value
            contrib = norm * val

            i1_plus_total += norm * result_plus.value
            i1_minus_base += norm * result_minus_base.value
            i1_minus_op += norm * result_minus_op.value
            total += contrib

    # =========================================================================
    # PART 2: I₂ (terms[1] for each pair)
    # =========================================================================
    # Run 8A: Support both DSL and direct Case C sources
    if i2_source == "direct_case_c":
        # Use proven Case C kernel evaluation (Run 7)
        from src.case_c_kernel import compute_i2_all_pairs_case_c
        direct_i2 = compute_i2_all_pairs_case_c(
            theta=theta, R=R, polynomials=polys_base, n=n, n_a=n_quad_a
        )

        for pair_key in ["11", "22", "33", "12", "21", "13", "31", "23", "32"]:
            norm = f_ordered[pair_key] if use_factorial_normalization else 1.0

            # Get raw values from direct computation
            i2_plus_raw = direct_i2[pair_key]["i2_plus"]
            i2_minus_base_raw = direct_i2[pair_key]["i2_minus"]

            # For direct_case_c, we don't apply Q-lift to I2
            # (I2 is proven from first principles, not using lifted Q)
            i2_minus_op_raw = i2_minus_base_raw

            pair = pair_breakdown.setdefault(pair_key, {"weight": float(norm)})
            pair["I2_plus"] = float(norm * i2_plus_raw)
            pair["I2_minus_base"] = float(norm * i2_minus_base_raw)
            pair["I2_minus_op"] = float(norm * i2_minus_op_raw)
            pair["I2_plus_raw"] = float(i2_plus_raw)
            pair["I2_minus_base_raw"] = float(i2_minus_base_raw)
            pair["I2_minus_op_raw"] = float(i2_minus_op_raw)

            val = i2_plus_raw + i2_minus_op_raw
            contrib = norm * val

            i2_plus_total += norm * i2_plus_raw
            i2_minus_base += norm * i2_minus_base_raw
            i2_minus_op += norm * i2_minus_op_raw
            total += contrib

    else:
        # Default: Use DSL-based term evaluation
        for pair_key in ["11", "22", "33", "12", "21", "13", "31", "23", "32"]:
            terms_plus = ordered_plus[pair_key]
            terms_minus = ordered_minus[pair_key]
            norm = f_ordered[pair_key] if use_factorial_normalization else 1.0

            term_plus = terms_plus[1]
            term_minus = terms_minus[1]

            result_plus = evaluate_term(
                term_plus, polys_base, n, R=R, theta=theta, n_quad_a=n_quad_a
            )
            result_minus_base = evaluate_term(
                term_minus, polys_base, n, R=-R, theta=theta, n_quad_a=n_quad_a
            )

            if apply_lift_to_i2:
                result_minus_op = evaluate_term(
                    term_minus, polys_lift_full, n, R=-R, theta=theta, n_quad_a=n_quad_a
                )
            else:
                result_minus_op = result_minus_base

            pair = pair_breakdown.setdefault(pair_key, {"weight": float(norm)})
            pair["I2_plus"] = float(norm * result_plus.value)
            pair["I2_minus_base"] = float(norm * result_minus_base.value)
            pair["I2_minus_op"] = float(norm * result_minus_op.value)
            pair["I2_plus_raw"] = float(result_plus.value)
            pair["I2_minus_base_raw"] = float(result_minus_base.value)
            pair["I2_minus_op_raw"] = float(result_minus_op.value)

            val = result_plus.value + result_minus_op.value
            contrib = norm * val

            i2_plus_total += norm * result_plus.value
            i2_minus_base += norm * result_minus_base.value
            i2_minus_op += norm * result_minus_op.value
            total += contrib

    # =========================================================================
    # PART 3: S34 (direct branch only - terms[2] and terms[3])
    # =========================================================================
    for pair_key in ["11", "22", "33", "12", "21", "13", "31", "23", "32"]:
        terms = ordered_plus[pair_key]
        norm = f_ordered[pair_key] if use_factorial_normalization else 1.0

        for i in [2, 3]:
            term = terms[i]
            result = evaluate_term(
                term, polys_base, n, R=R, theta=theta, n_quad_a=n_quad_a
            )
            contrib = norm * result.value
            s34_plus_total += contrib
            total += contrib

            pair = pair_breakdown.setdefault(pair_key, {"weight": float(norm)})
            key = "I3_plus" if i == 2 else "I4_plus"
            key_raw = "I3_plus_raw" if i == 2 else "I4_plus_raw"
            pair[key] = float(contrib)
            pair[key_raw] = float(result.value)

        pair = pair_breakdown.setdefault(pair_key, {"weight": float(norm)})
        pair["S34_plus"] = float(pair.get("I3_plus", 0.0) + pair.get("I4_plus", 0.0))
        pair["S34_plus_raw"] = float(pair.get("I3_plus_raw", 0.0) + pair.get("I4_plus_raw", 0.0))

    # Compute implied weights (ratio of op/base for each channel)
    m1_implied = i1_minus_op / i1_minus_base if abs(i1_minus_base) > 1e-15 else float('inf')
    m2_implied = i2_minus_op / i2_minus_base if abs(i2_minus_base) > 1e-15 else float('inf')

    if verbose:
        print(f"\nChannels:")
        print(f"  I1+:      {i1_plus_total:+.8f}")
        print(f"  I1-base:  {i1_minus_base:+.8f}")
        print(f"  I1-op:    {i1_minus_op:+.8f}")
        print(f"  I2+:      {i2_plus_total:+.8f}")
        print(f"  I2-base:  {i2_minus_base:+.8f}")
        print(f"  I2-op:    {i2_minus_op:+.8f}")
        print(f"  S34+:     {s34_plus_total:+.8f}")
        print(f"Implied:  m1={m1_implied:.4f}, m2={m2_implied:.4f}")

    # Store results
    per_term["_I1_plus"] = i1_plus_total
    per_term["_I1_minus_base"] = i1_minus_base
    per_term["_I1_minus_op"] = i1_minus_op
    per_term["_I2_plus"] = i2_plus_total
    per_term["_I2_minus_base"] = i2_minus_base
    per_term["_I2_minus_op"] = i2_minus_op
    per_term["_S34_plus"] = s34_plus_total
    per_term["_m1_implied"] = m1_implied
    per_term["_m2_implied"] = m2_implied
    per_term["_norm_factor"] = norm_factor
    per_term["_normalization"] = normalization_key
    per_term["_lift_scope"] = lift_scope
    per_term["_sigma"] = sigma
    per_term["_i1_source"] = i1_source
    per_term["_i2_source"] = i2_source
    per_term["_Q_0"] = Q_0
    per_term["_Q_1"] = Q_1
    per_term["_Q_at_sigma"] = Q_at_sigma
    per_term["_is_identity_mode"] = is_identity_mode
    per_term["_pair_mode"] = "ordered"
    per_term["_mirror_mode"] = f"v2_{normalization_key}_{lift_scope}_s{sigma:.2f}"
    per_term["_pair_breakdown"] = pair_breakdown

    return EvaluationResult(
        total=total,
        per_term=per_term,
        n=n,
        term_results=None
    )


# =============================================================================
# CODEX TASK 1: Operator-Implied Weights (No Fitting)
# =============================================================================
# Per GPT guidance 2025-12-20: This function computes implied weights from
# the operator model WITHOUT any 2×2 solve. It's the bridge from
# "operator hypothesis" to "measured consequence".

@dataclass
class OperatorImpliedWeights:
    """Result of computing operator-implied weights (no fitting).

    This is the canonical output for operator mode diagnostics.
    The implied weights m1, m2 are computed as ratios I_minus_op / I_minus_base,
    NOT from solving a 2×2 system.
    """
    # Benchmark parameters
    theta: float
    R: float
    sigma: float
    normalization: str
    lift_scope: str

    # Base channel values
    I1_minus_base: float
    I2_minus_base: float

    # Operator channel values
    I1_minus_op: float
    I2_minus_op: float

    # IMPLIED weights (ratio, NOT solved)
    m1_implied: float
    m2_implied: float

    # Plus channels (for computing c)
    I1_plus: float
    I2_plus: float
    S34_plus: float

    # Full breakdown by pair (for localization)
    pair_breakdown: Dict[str, Dict[str, float]]

    # Operator-mode c value (using implied weights)
    c_operator: float

    # Metadata
    is_identity_mode: bool
    norm_factor: float


def compute_operator_implied_weights(
    theta: float,
    R: float,
    polynomials: Dict[str, PolyLike],
    *,
    sigma: float = 5/32,  # Default to discovered optimal
    normalization: str = "grid",
    lift_scope: str = "i1_only",
    n: int = 60,
    n_quad_a: int = 40,
    verbose: bool = False,
    i1_source: str = "dsl",  # GPT Phase 1: "dsl" or "post_identity_operator"
    i2_source: str = "dsl",  # Run 8A: "dsl" or "direct_case_c"
    terms_version: str = "old",  # Run 10A: "old" or "v2"
) -> OperatorImpliedWeights:
    """
    Compute operator-implied weights WITHOUT any 2×2 solve.

    This is the first-class API for operator mode (Codex Task 1).
    The implied weights are computed as ratios:
        m1_implied = I1_minus_op / I1_minus_base
        m2_implied = I2_minus_op / I2_minus_base

    This function does NOT call solve_two_weight_operator or any fitting routine.

    Args:
        theta: θ parameter (typically 4/7)
        R: R parameter (e.g., 1.3036 for κ, 1.1167 for κ*)
        polynomials: Dict with keys "P1", "P2", "P3", "Q"
        sigma: Shift magnitude (default 5/32 = 0.15625)
        normalization: How to normalize Q_lift ("none", "l2", "grid")
        lift_scope: Which Q factors to apply lift to ("i1_only", "i2_only", "both")
        n: Number of quadrature points per dimension
        n_quad_a: Quadrature points for Case C a-integral
        verbose: Print diagnostic information
        i1_source: "dsl" (DSL-based) or "post_identity_operator" (GPT Phase 1)
        i2_source: "dsl" (DSL-based) or "direct_case_c" (proven Run 7)

    Returns:
        OperatorImpliedWeights with implied weights and channel breakdown

    Note:
        This function is for computing implied weights ONLY.
        It does NOT solve for (m1, m2) using target c values.
        Use solve_two_weight_operator_diagnostic() for comparison purposes.
    """
    # Call the core operator evaluator
    result = compute_c_paper_operator_v2(
        theta=theta,
        R=R,
        n=n,
        polynomials=polynomials,
        n_quad_a=n_quad_a,
        verbose=verbose,
        normalization=normalization,
        lift_scope=lift_scope,
        sigma=sigma,
        i1_source=i1_source,  # GPT Phase 1
        i2_source=i2_source,
        terms_version=terms_version,  # Run 10A
    )

    # Extract channel values
    I1_minus_base = result.per_term["_I1_minus_base"]
    I2_minus_base = result.per_term["_I2_minus_base"]
    I1_minus_op = result.per_term["_I1_minus_op"]
    I2_minus_op = result.per_term["_I2_minus_op"]
    I1_plus = result.per_term["_I1_plus"]
    I2_plus = result.per_term["_I2_plus"]
    S34_plus = result.per_term["_S34_plus"]

    # Compute implied weights as ratios (NO SOLVE)
    m1_implied = I1_minus_op / I1_minus_base if abs(I1_minus_base) > 1e-15 else float('inf')
    m2_implied = I2_minus_op / I2_minus_base if abs(I2_minus_base) > 1e-15 else float('inf')

    # Compute operator-mode c using implied weights
    c_operator = I1_plus + I1_minus_op + I2_plus + I2_minus_op + S34_plus

    # Build pair breakdown (for term-level localization)
    pair_breakdown = result.per_term.get("_pair_breakdown", {})

    is_identity_mode = result.per_term.get("_is_identity_mode", False)
    norm_factor = result.per_term.get("_norm_factor", 1.0)

    return OperatorImpliedWeights(
        theta=theta,
        R=R,
        sigma=sigma,
        normalization=normalization,
        lift_scope=lift_scope,
        I1_minus_base=I1_minus_base,
        I2_minus_base=I2_minus_base,
        I1_minus_op=I1_minus_op,
        I2_minus_op=I2_minus_op,
        m1_implied=m1_implied,
        m2_implied=m2_implied,
        I1_plus=I1_plus,
        I2_plus=I2_plus,
        S34_plus=S34_plus,
        pair_breakdown=pair_breakdown,
        c_operator=c_operator,
        is_identity_mode=is_identity_mode,
        norm_factor=norm_factor,
    )


# =============================================================================
# CODEX TASK 2: Diagnostic Solve (Explicitly Labeled)
# =============================================================================
# This wrapper makes it explicit that solving is DIAGNOSTIC ONLY.

def solve_two_weight_operator_diagnostic(
    result_k: EvaluationResult,
    result_k_star: EvaluationResult,
    c_target_k: float,
    c_target_k_star: float,
    use_operator_channels: bool = True,
) -> Dict[str, float]:
    """
    Solve for (m₁, m₂) - DIAGNOSTIC ONLY.

    This function is explicitly labeled as diagnostic. It should NOT be used
    in truth evaluation pipelines. The solved weights are for comparison
    purposes only.

    See solve_two_weight_operator() for the underlying implementation.

    WARNING: This performs curve-fitting on (κ, κ*). The resulting weights
    may be artifacts of fitting, not structural features.
    """
    import warnings
    warnings.warn(
        "solve_two_weight_operator_diagnostic() is a DIAGNOSTIC function. "
        "The solved weights are NOT ground truth - they are curve-fit artifacts. "
        "For structural analysis, use compute_operator_implied_weights().",
        UserWarning,
        stacklevel=2,
    )
    return solve_two_weight_operator(
        result_k, result_k_star, c_target_k, c_target_k_star, use_operator_channels
    )


# =============================================================================
# CODEX TASK 3: Fixed-σ Operator Mirror Mode
# =============================================================================
# Compute c with a fixed σ shift - NO FITTING.

def compute_c_operator_sigma_shift(
    theta: float,
    R: float,
    polynomials: Dict[str, PolyLike],
    *,
    sigma: float = 5/32,  # Fixed default to discovered optimal
    normalization: str = "grid",
    lift_scope: str = "i1_only",
    n: int = 60,
    n_quad_a: int = 40,
    verbose: bool = False,
    i2_source: str = "dsl",  # Run 8A: "dsl" or "direct_case_c"
) -> Dict[str, Any]:
    """
    Compute c using fixed-σ operator mode (NO FITTING).

    This is the "operator_sigma_shift" mirror mode (Codex Task 3).
    It uses σ as a fixed constant and computes c directly.

    INVARIANT: Direct branch (+R) outputs are IDENTICAL to non-operator paper truth.

    Args:
        theta: θ parameter (typically 4/7)
        R: R parameter
        polynomials: Dict with keys "P1", "P2", "P3", "Q"
        sigma: Fixed shift magnitude (default 5/32)
        normalization: How to normalize Q_lift ("none", "l2", "grid")
        lift_scope: Which Q factors to apply lift to
        n: Number of quadrature points
        n_quad_a: Quadrature points for Case C a-integral
        verbose: Print diagnostic information

    Returns:
        Dict with:
        - c_operator: The computed c value using operator mode
        - implied_weights: OperatorImpliedWeights object
        - channel_breakdown: Detailed channel values
        - comparison: (DIAGNOSTIC ONLY) Comparison with base if requested

    Note:
        This function does NOT call any 2×2 solve internally.
        Any comparison with solved weights is for DIAGNOSTIC purposes only.
    """
    # Get implied weights (this does NOT solve)
    implied = compute_operator_implied_weights(
        theta=theta,
        R=R,
        polynomials=polynomials,
        sigma=sigma,
        normalization=normalization,
        lift_scope=lift_scope,
        n=n,
        n_quad_a=n_quad_a,
        verbose=verbose,
        i2_source=i2_source,
    )

    # Channel breakdown for transparency
    channel_breakdown = {
        "I1_plus": implied.I1_plus,
        "I1_minus_base": implied.I1_minus_base,
        "I1_minus_op": implied.I1_minus_op,
        "I2_plus": implied.I2_plus,
        "I2_minus_base": implied.I2_minus_base,
        "I2_minus_op": implied.I2_minus_op,
        "S34_plus": implied.S34_plus,
    }

    return {
        "c_operator": implied.c_operator,
        "implied_weights": implied,
        "channel_breakdown": channel_breakdown,
        "sigma": sigma,
        "normalization": normalization,
        "lift_scope": lift_scope,
        "mirror_mode": "operator_sigma_shift",
    }


# =============================================================================
# CODEX TASK 1: Normalize Nothing by Default
# =============================================================================
# Per GPT guidance 2025-12-20: Make normalization explicit and quarantined.
# Default should be "none" and any normalization should emit a warning.

def compute_operator_components_raw(
    theta: float,
    R: float,
    polynomials: Dict[str, PolyLike],
    *,
    sigma: float = 5/32,
    lift_scope: str = "i1_only",
    n: int = 60,
    n_quad_a: int = 40,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Compute operator components with NO normalization (raw mode).

    This is the canonical entry point for operator analysis.
    Normalization is DISABLED by default to expose true amplitude structure.

    Args:
        theta: θ parameter
        R: R parameter
        polynomials: Dict with keys "P1", "P2", "P3", "Q"
        sigma: Shift magnitude (default 5/32)
        lift_scope: Which Q factors to apply lift to
        n: Quadrature points per dimension
        n_quad_a: Quadrature points for Case C a-integral
        verbose: Print diagnostic information

    Returns:
        Dict with raw operator components (no normalization applied)
    """
    result = compute_c_paper_operator_v2(
        theta=theta,
        R=R,
        n=n,
        polynomials=polynomials,
        n_quad_a=n_quad_a,
        verbose=verbose,
        normalization="none",  # ALWAYS none for raw mode
        lift_scope=lift_scope,
        sigma=sigma,
    )

    return {
        "I1_plus": result.per_term["_I1_plus"],
        "I1_minus_base": result.per_term["_I1_minus_base"],
        "I1_minus_op": result.per_term["_I1_minus_op"],
        "I2_plus": result.per_term["_I2_plus"],
        "I2_minus_base": result.per_term["_I2_minus_base"],
        "I2_minus_op": result.per_term["_I2_minus_op"],
        "S34_plus": result.per_term["_S34_plus"],
        "m1_implied": result.per_term["_m1_implied"],
        "m2_implied": result.per_term["_m2_implied"],
        "sigma": sigma,
        "lift_scope": lift_scope,
        "normalization": "none",
        "R": R,
        "theta": theta,
    }


# =============================================================================
# CODEX TASK 2: Global Mirror Amplitude Parameter
# =============================================================================
# Factor out the global amplitude from the operator deformation.

@dataclass
class OperatorFactorization:
    """Factorized operator result separating shape and amplitude."""
    # Base channels
    I1_minus_base: float
    I2_minus_base: float

    # Operator-deformed channels (shape effect)
    I1_minus_op: float
    I2_minus_op: float

    # Implied shape factors (should be O(1))
    m1_shape: float  # = I1_minus_op / I1_minus_base
    m2_shape: float  # = I2_minus_op / I2_minus_base

    # Plus channels (unchanged by operator)
    I1_plus: float
    I2_plus: float
    S34_plus: float

    # Parameters
    theta: float
    R: float
    sigma: float


def compute_operator_factorization(
    theta: float,
    R: float,
    polynomials: Dict[str, PolyLike],
    *,
    sigma: float = 5/32,
    lift_scope: str = "i1_only",
    n: int = 60,
    n_quad_a: int = 40,
) -> OperatorFactorization:
    """
    Compute operator factorization separating shape from amplitude.

    The operator effect is factored as:
        I_minus_op = m_shape × I_minus_base

    where m_shape captures the "shape" deformation (σ-shift effect),
    and the remaining amplitude is to be derived from TeX.

    Args:
        theta: θ parameter
        R: R parameter
        polynomials: Dict with keys "P1", "P2", "P3", "Q"
        sigma: Shift magnitude
        lift_scope: Which Q factors to apply lift to
        n: Quadrature points
        n_quad_a: Quadrature points for Case C

    Returns:
        OperatorFactorization with separated shape and base components
    """
    raw = compute_operator_components_raw(
        theta=theta, R=R, polynomials=polynomials,
        sigma=sigma, lift_scope=lift_scope, n=n, n_quad_a=n_quad_a,
    )

    m1_shape = raw["I1_minus_op"] / raw["I1_minus_base"] if abs(raw["I1_minus_base"]) > 1e-15 else float('inf')
    m2_shape = raw["I2_minus_op"] / raw["I2_minus_base"] if abs(raw["I2_minus_base"]) > 1e-15 else float('inf')

    return OperatorFactorization(
        I1_minus_base=raw["I1_minus_base"],
        I2_minus_base=raw["I2_minus_base"],
        I1_minus_op=raw["I1_minus_op"],
        I2_minus_op=raw["I2_minus_op"],
        m1_shape=m1_shape,
        m2_shape=m2_shape,
        I1_plus=raw["I1_plus"],
        I2_plus=raw["I2_plus"],
        S34_plus=raw["S34_plus"],
        theta=theta,
        R=R,
        sigma=sigma,
    )


# =============================================================================
# CODEX TASK 3: Residual Factorization Report
# =============================================================================
# Report what amplitude is still missing after operator deformation.

@dataclass
class ResidualFactorizationReport:
    """Report of residual amplitude after operator deformation."""
    # Benchmark identifiers
    benchmark: str  # "kappa" or "kappa_star"
    R: float

    # Solved weights (from 2×2 fit - DIAGNOSTIC ONLY)
    m1_solved: float
    m2_solved: float

    # Implied shape factors (from operator mode)
    m1_shape: float
    m2_shape: float

    # Residual amplitudes (what's still missing)
    A1_resid: float  # = m1_solved / m1_shape
    A2_resid: float  # = m2_solved / m2_shape

    # Consistency check
    A_ratio: float  # = A1_resid / A2_resid (should be ~1 if global)


def report_residual_amplitude(
    theta: float,
    R_kappa: float,
    R_kappa_star: float,
    polys_kappa: Dict[str, PolyLike],
    polys_kappa_star: Dict[str, PolyLike],
    c_target_kappa: float,
    c_target_kappa_star: float,
    *,
    sigma: float = 5/32,
    lift_scope: str = "i1_only",
    n: int = 60,
    n_quad_a: int = 40,
) -> Tuple["ResidualFactorizationReport", "ResidualFactorizationReport"]:
    """
    Report residual amplitude after operator deformation.

    For each benchmark (κ and κ*), computes:
    - m1_solved, m2_solved: From 2×2 fit (DIAGNOSTIC ONLY)
    - m1_shape, m2_shape: From operator mode (σ-shift effect)
    - A1_resid, A2_resid: Residual amplitudes (m_solved / m_shape)

    Success criterion:
    - If A1_resid ≈ A2_resid → global missing prefactor
    - If A_resid similar across κ and κ* → consistent structure

    Args:
        theta: θ parameter
        R_kappa: R for κ benchmark
        R_kappa_star: R for κ* benchmark
        polys_kappa: Polynomials for κ
        polys_kappa_star: Polynomials for κ*
        c_target_kappa: Target c for κ
        c_target_kappa_star: Target c for κ*
        sigma: Shift magnitude
        lift_scope: Which Q factors to apply lift to
        n: Quadrature points
        n_quad_a: Quadrature points for Case C

    Returns:
        Tuple of (report_kappa, report_kappa_star)
    """
    import numpy as np

    # Get operator factorizations
    fact_k = compute_operator_factorization(
        theta=theta, R=R_kappa, polynomials=polys_kappa,
        sigma=sigma, lift_scope=lift_scope, n=n, n_quad_a=n_quad_a,
    )
    fact_ks = compute_operator_factorization(
        theta=theta, R=R_kappa_star, polynomials=polys_kappa_star,
        sigma=sigma, lift_scope=lift_scope, n=n, n_quad_a=n_quad_a,
    )

    # Get solved weights (DIAGNOSTIC ONLY - from 2×2 fit)
    result_k = compute_c_paper_operator_v2(
        theta=theta, R=R_kappa, n=n, polynomials=polys_kappa,
        n_quad_a=n_quad_a, verbose=False,
        normalization="none", lift_scope=lift_scope, sigma=sigma,
    )
    result_ks = compute_c_paper_operator_v2(
        theta=theta, R=R_kappa_star, n=n, polynomials=polys_kappa_star,
        n_quad_a=n_quad_a, verbose=False,
        normalization="none", lift_scope=lift_scope, sigma=sigma,
    )

    solved = solve_two_weight_operator(
        result_k, result_ks,
        c_target_k=c_target_kappa, c_target_k_star=c_target_kappa_star,
        use_operator_channels=False,  # Use BASE channels for solved weights
    )

    m1_solved, m2_solved = solved["m1"], solved["m2"]

    # Compute residual amplitudes for each benchmark
    def make_report(fact, benchmark, R):
        A1 = m1_solved / fact.m1_shape if abs(fact.m1_shape) > 1e-15 else float('inf')
        A2 = m2_solved / fact.m2_shape if abs(fact.m2_shape) > 1e-15 else float('inf')
        A_ratio = A1 / A2 if abs(A2) > 1e-15 else float('inf')

        return ResidualFactorizationReport(
            benchmark=benchmark,
            R=R,
            m1_solved=m1_solved,
            m2_solved=m2_solved,
            m1_shape=fact.m1_shape,
            m2_shape=fact.m2_shape,
            A1_resid=A1,
            A2_resid=A2,
            A_ratio=A_ratio,
        )

    report_k = make_report(fact_k, "kappa", R_kappa)
    report_ks = make_report(fact_ks, "kappa_star", R_kappa_star)

    return report_k, report_ks


# =============================================================================
# CODEX TASK 4: Candidate Amplitude Library
# =============================================================================
# TeX-motivated amplitude candidates for fitting residuals.

def candidate_amplitude_functions() -> Dict[str, callable]:
    """
    Return a library of TeX-motivated amplitude candidates.

    These are potential forms for the missing global mirror amplitude A(R, θ).
    """
    import numpy as np

    return {
        "exp(R)": lambda R, theta: np.exp(R),
        "exp(2R)": lambda R, theta: np.exp(2*R),
        "(exp(2R)-1)/(2R)": lambda R, theta: (np.exp(2*R) - 1) / (2*R) if R != 0 else 1.0,
        "(1/theta)*(exp(2R)-1)/(2R)": lambda R, theta: (np.exp(2*R) - 1) / (2*R*theta) if R != 0 else 1.0/theta,
        "exp(R)+5": lambda R, theta: np.exp(R) + 5,
        "exp(2R/theta)": lambda R, theta: np.exp(2*R/theta),
    }


def fit_amplitude_candidates(
    R_values: List[float],
    A_resid_values: List[float],
    theta: float,
) -> Dict[str, Dict[str, float]]:
    """
    Fit candidate amplitude functions to residual data.

    For each candidate A(R, θ), compute the best-fit scaling factor k
    such that k × A(R, θ) ≈ A_resid(R), and report the fit error.

    Args:
        R_values: List of R values
        A_resid_values: Corresponding residual amplitudes
        theta: θ parameter

    Returns:
        Dict mapping candidate name to {"scale": k, "rmse": error, "max_error": max_error}
    """
    import numpy as np

    candidates = candidate_amplitude_functions()
    results = {}

    for name, func in candidates.items():
        # Compute candidate values
        A_candidate = np.array([func(R, theta) for R in R_values])

        # Find best-fit scale (least squares)
        if np.sum(A_candidate**2) > 1e-15:
            k = np.sum(np.array(A_resid_values) * A_candidate) / np.sum(A_candidate**2)
        else:
            k = 1.0

        # Compute errors
        A_fitted = k * A_candidate
        errors = np.array(A_resid_values) - A_fitted
        rmse = np.sqrt(np.mean(errors**2))
        max_error = np.max(np.abs(errors))
        rel_error = rmse / np.mean(np.abs(A_resid_values)) * 100 if np.mean(np.abs(A_resid_values)) > 1e-15 else float('inf')

        results[name] = {
            "scale": k,
            "rmse": rmse,
            "max_error": max_error,
            "rel_error_pct": rel_error,
        }

    return results


def rank_amplitude_fits(
    R_values: List[float],
    A1_resid_values: List[float],
    A2_resid_values: List[float],
    theta: float,
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Rank amplitude candidates by fit quality.

    Fits both A1_resid and A2_resid, then ranks by combined error.

    Args:
        R_values: List of R values
        A1_resid_values: Residual amplitudes for channel 1
        A2_resid_values: Residual amplitudes for channel 2
        theta: θ parameter

    Returns:
        List of (candidate_name, fit_info) sorted by combined error (best first)
    """
    fits_A1 = fit_amplitude_candidates(R_values, A1_resid_values, theta)
    fits_A2 = fit_amplitude_candidates(R_values, A2_resid_values, theta)

    combined = []
    for name in fits_A1.keys():
        combined_rmse = (fits_A1[name]["rmse"] + fits_A2[name]["rmse"]) / 2
        combined_rel = (fits_A1[name]["rel_error_pct"] + fits_A2[name]["rel_error_pct"]) / 2
        combined.append((name, {
            "A1_scale": fits_A1[name]["scale"],
            "A2_scale": fits_A2[name]["scale"],
            "A1_rmse": fits_A1[name]["rmse"],
            "A2_rmse": fits_A2[name]["rmse"],
            "combined_rmse": combined_rmse,
            "combined_rel_pct": combined_rel,
            "scale_ratio": fits_A1[name]["scale"] / fits_A2[name]["scale"] if fits_A2[name]["scale"] != 0 else float('inf'),
        }))

    # Sort by combined relative error
    combined.sort(key=lambda x: x[1]["combined_rel_pct"])
    return combined


# =============================================================================
# CODEX TASK 4b: Affine (a·f(R,θ)+b) Amplitude Fits
# =============================================================================
# Some TeX-motivated amplitude mechanisms may include an additive constant.
# These helpers fit a two-parameter affine family to residual amplitude data.


def candidate_affine_amplitude_families() -> Dict[str, callable]:
    """Return base functions for affine amplitude families a·f(R,θ)+b."""
    base = candidate_amplitude_functions()
    return {
        "a*exp(R)+b": base["exp(R)"],
        "a*(exp(2R)-1)/(2R)+b": base["(exp(2R)-1)/(2R)"],
    }


def fit_affine_amplitude_candidates(
    R_values: List[float],
    A_resid_values: List[float],
    theta: float,
) -> Dict[str, Dict[str, float]]:
    """
    Fit affine amplitude families a·f(R,θ)+b to residual amplitude data.

    This is a diagnostic helper. It does NOT modify any evaluation code paths.

    Returns:
        Dict mapping candidate name -> {"a","b","rmse","max_error","rel_error_pct"}.
    """
    import numpy as np

    candidates = candidate_affine_amplitude_families()
    y = np.asarray(A_resid_values, dtype=float)
    results: Dict[str, Dict[str, float]] = {}

    for name, func in candidates.items():
        x = np.asarray([func(R, theta) for R in R_values], dtype=float)
        A = np.column_stack([x, np.ones_like(x)])
        params, *_ = np.linalg.lstsq(A, y, rcond=None)
        a, b = float(params[0]), float(params[1])
        y_hat = a * x + b
        errors = y - y_hat

        rmse = float(np.sqrt(np.mean(errors**2)))
        max_error = float(np.max(np.abs(errors))) if errors.size else 0.0
        denom = float(np.mean(np.abs(y))) if y.size else 0.0
        rel_error = rmse / denom * 100 if denom > 1e-15 else float("inf")

        results[name] = {
            "a": a,
            "b": b,
            "rmse": rmse,
            "max_error": max_error,
            "rel_error_pct": float(rel_error),
        }

    return results


def rank_affine_amplitude_fits(
    R_values: List[float],
    A1_resid_values: List[float],
    A2_resid_values: List[float],
    theta: float,
) -> List[Tuple[str, Dict[str, Any]]]:
    """Rank affine amplitude candidates by combined fit quality (A1 and A2)."""
    fits_A1 = fit_affine_amplitude_candidates(R_values, A1_resid_values, theta)
    fits_A2 = fit_affine_amplitude_candidates(R_values, A2_resid_values, theta)

    combined: List[Tuple[str, Dict[str, Any]]] = []
    for name in fits_A1.keys():
        combined_rmse = (fits_A1[name]["rmse"] + fits_A2[name]["rmse"]) / 2
        combined_rel = (fits_A1[name]["rel_error_pct"] + fits_A2[name]["rel_error_pct"]) / 2
        combined.append((name, {
            "A1_a": fits_A1[name]["a"],
            "A1_b": fits_A1[name]["b"],
            "A2_a": fits_A2[name]["a"],
            "A2_b": fits_A2[name]["b"],
            "A1_rmse": fits_A1[name]["rmse"],
            "A2_rmse": fits_A2[name]["rmse"],
            "combined_rmse": combined_rmse,
            "combined_rel_pct": combined_rel,
            "a_ratio": fits_A1[name]["a"] / fits_A2[name]["a"] if abs(fits_A2[name]["a"]) > 1e-15 else float("inf"),
        }))

    combined.sort(key=lambda x: x[1]["combined_rel_pct"])
    return combined


# =============================================================================
# CODEX TASK 2 (GPT Run 2): Compare Operator to Two-Weight Solve
# =============================================================================
# Explicitly separate "shape" from "amplitude" using operator-mode as shape
# and solved weights as comparison only.

@dataclass
class OperatorVsSolvedComparison:
    """Comparison of operator-implied weights vs solved (diagnostic) weights."""
    # Implied weights from operator mode (prediction-only)
    m1_implied: float
    m2_implied: float
    # Solved weights from 2x2 system (diagnostic only)
    m1_solved: float
    m2_solved: float
    # Residual amplitudes: A = m_solved / m_implied
    A1_residual: float  # m1_solved / m1_implied
    A2_residual: float  # m2_solved / m2_implied
    # Benchmark used
    benchmark: str  # "kappa" or "kappa_star"
    R: float
    # Metadata
    sigma: float
    normalization: str
    lift_scope: str


def compare_operator_to_two_weight_solve(
    polys_kappa: Dict[str, PolyLike],
    polys_kappa_star: Dict[str, PolyLike],
    *,
    theta: float = 4.0 / 7.0,
    R_kappa: float = 1.3036,
    R_kappa_star: float = 1.1167,
    c_target_kappa: float = 2.137,
    c_target_kappa_star: float = 1.938,
    sigma: float = 5/32,
    normalization: str = "grid",
    lift_scope: str = "i1_only",
    n: int = 60,
    n_quad_a: int = 40,
) -> Tuple[OperatorVsSolvedComparison, OperatorVsSolvedComparison, Dict[str, Any]]:
    """
    Compare operator-implied weights to two-weight solved weights (DIAGNOSTIC).

    This function uses the operator-mode implied weights as the PREDICTION
    and the two-weight solve as COMPARISON ONLY.

    The key outputs are the residual amplitudes:
        A1 = m1_solved / m1_implied
        A2 = m2_solved / m2_implied

    If A1 ≈ A2 across κ and κ*, then the remaining gap is a GLOBAL amplitude.
    If A1 and A2 differ across benchmarks, the gap is benchmark-dependent.

    Args:
        polys_kappa: Polynomials for κ benchmark
        polys_kappa_star: Polynomials for κ* benchmark
        theta: θ parameter (typically 4/7)
        R_kappa: R for κ benchmark (typically 1.3036)
        R_kappa_star: R for κ* benchmark (typically 1.1167)
        c_target_kappa: Target c for κ benchmark
        c_target_kappa_star: Target c for κ* benchmark
        sigma: Shift magnitude for operator mode
        normalization: How to normalize Q_lift
        lift_scope: Which Q factors to apply lift to
        n: Quadrature points
        n_quad_a: Quadrature points for Case C a-integral

    Returns:
        Tuple of:
        - OperatorVsSolvedComparison for κ benchmark
        - OperatorVsSolvedComparison for κ* benchmark
        - summary dict with:
          - A1_avg, A2_avg: average residuals
          - A1_span, A2_span: relative span across benchmarks
          - is_global: True if A1 and A2 are stable (span < 20%)
          - A_ratio_kappa, A_ratio_kappa_star: A1/A2 for each benchmark
    """
    # Step 1: Get operator-implied weights for both benchmarks
    implied_k = compute_operator_implied_weights(
        theta=theta, R=R_kappa, polynomials=polys_kappa,
        sigma=sigma, normalization=normalization, lift_scope=lift_scope,
        n=n, n_quad_a=n_quad_a,
    )
    implied_ks = compute_operator_implied_weights(
        theta=theta, R=R_kappa_star, polynomials=polys_kappa_star,
        sigma=sigma, normalization=normalization, lift_scope=lift_scope,
        n=n, n_quad_a=n_quad_a,
    )

    # Step 2: Get solved weights (DIAGNOSTIC ONLY)
    # We need the raw results for the solver
    result_k = compute_c_paper_operator_v2(
        theta=theta, R=R_kappa, n=n, polynomials=polys_kappa,
        n_quad_a=n_quad_a, verbose=False,
        normalization=normalization, lift_scope=lift_scope, sigma=sigma,
    )
    result_ks = compute_c_paper_operator_v2(
        theta=theta, R=R_kappa_star, n=n, polynomials=polys_kappa_star,
        n_quad_a=n_quad_a, verbose=False,
        normalization=normalization, lift_scope=lift_scope, sigma=sigma,
    )

    solved = solve_two_weight_operator(
        result_k, result_ks,
        c_target_k=c_target_kappa, c_target_k_star=c_target_kappa_star,
        use_operator_channels=True,
    )
    m1_solved = solved["m1"]
    m2_solved = solved["m2"]

    # Step 3: Compute residual amplitudes
    def safe_div(a, b):
        return a / b if abs(b) > 1e-15 else float('inf')

    A1_k = safe_div(m1_solved, implied_k.m1_implied)
    A2_k = safe_div(m2_solved, implied_k.m2_implied)
    A1_ks = safe_div(m1_solved, implied_ks.m1_implied)
    A2_ks = safe_div(m2_solved, implied_ks.m2_implied)

    # Build comparison objects
    comp_k = OperatorVsSolvedComparison(
        m1_implied=implied_k.m1_implied,
        m2_implied=implied_k.m2_implied,
        m1_solved=m1_solved,
        m2_solved=m2_solved,
        A1_residual=A1_k,
        A2_residual=A2_k,
        benchmark="kappa",
        R=R_kappa,
        sigma=sigma,
        normalization=normalization,
        lift_scope=lift_scope,
    )
    comp_ks = OperatorVsSolvedComparison(
        m1_implied=implied_ks.m1_implied,
        m2_implied=implied_ks.m2_implied,
        m1_solved=m1_solved,
        m2_solved=m2_solved,
        A1_residual=A1_ks,
        A2_residual=A2_ks,
        benchmark="kappa_star",
        R=R_kappa_star,
        sigma=sigma,
        normalization=normalization,
        lift_scope=lift_scope,
    )

    # Compute summary statistics
    A1_avg = (A1_k + A1_ks) / 2
    A2_avg = (A2_k + A2_ks) / 2
    A1_span = abs(A1_k - A1_ks) / A1_avg if A1_avg != 0 else float('inf')
    A2_span = abs(A2_k - A2_ks) / A2_avg if A2_avg != 0 else float('inf')

    is_global = A1_span < 0.20 and A2_span < 0.20

    summary = {
        "A1_avg": A1_avg,
        "A2_avg": A2_avg,
        "A1_span": A1_span,
        "A2_span": A2_span,
        "is_global": is_global,
        "A_ratio_kappa": safe_div(A1_k, A2_k),
        "A_ratio_kappa_star": safe_div(A1_ks, A2_ks),
        "m1_solved": m1_solved,
        "m2_solved": m2_solved,
        "cond": solved["cond"],
    }

    return comp_k, comp_ks, summary


# =============================================================================
# CODEX TASK 3 (GPT Run 2): Moment-Based σ Predictor (Anti-Overfit Probe)
# =============================================================================
# Compute candidate σ values from moments of t under natural weights.

@dataclass
class MomentBasedSigmaCandidates:
    """Candidate σ values derived from moment analysis."""
    # Moments
    E_t: float  # E[t] under w(t)
    E_t2: float  # E[t²] under w(t)
    E_t1mt: float  # E[t(1-t)] under w(t)
    Var_t: float  # Var(t) under w(t)
    # Candidate σ values
    sigma_E_t1mt: float  # E[t(1-t)]
    sigma_E_t_minus_half: float  # E[t] - 1/2
    sigma_E_t_div4: float  # E[t]/4
    sigma_E_t1mt_times_theta: float  # E[t(1-t)] × θ
    sigma_Var_t: float  # Var(t)
    # Metadata
    theta: float
    R: float
    weight_type: str  # "Q2_exp2Rt" or other


def compute_moment_based_sigma_candidates(
    theta: float,
    R: float,
    Q: PolyLike,
    *,
    n_quad: int = 200,
    weight_type: str = "Q2_exp2Rt",
) -> MomentBasedSigmaCandidates:
    """
    Compute candidate σ values from moments of t under natural weights.

    This is an anti-overfit probe. If σ ≈ 5/32 matches one of these moments,
    it suggests a derivable structural relationship. If not, σ = 5/32 may be
    an empirical fit to PRZZ's specific Q structure.

    The natural weight is:
        w(t) = Q(t)² × exp(2Rt)  (from TeX line 1548)

    Candidate σ values are computed from various moments of t under w(t).

    Args:
        theta: θ parameter
        R: R parameter
        Q: Q polynomial
        n_quad: Number of quadrature points
        weight_type: Weight function type

    Returns:
        MomentBasedSigmaCandidates with moments and candidate σ values
    """
    from src.quadrature import gauss_legendre_01
    import numpy as np

    # Set up 1D quadrature on [0, 1]
    nodes, weights = gauss_legendre_01(n_quad)

    # Evaluate Q at nodes using the polynomial's eval method
    Q_vals = Q.eval(nodes)

    # Weight function: Q(t)² × exp(2Rt)
    w = Q_vals**2 * np.exp(2 * R * nodes)

    # Normalize to probability measure
    Z = np.sum(w * weights)
    if abs(Z) < 1e-15:
        # Fallback to uniform if weight is degenerate
        w = np.ones_like(nodes)
        Z = np.sum(w * weights)

    # Compute moments
    E_t = np.sum(w * nodes * weights) / Z
    E_t2 = np.sum(w * nodes**2 * weights) / Z
    E_t1mt = np.sum(w * nodes * (1 - nodes) * weights) / Z
    Var_t = E_t2 - E_t**2

    # Candidate σ values
    sigma_E_t1mt = E_t1mt
    sigma_E_t_minus_half = E_t - 0.5
    sigma_E_t_div4 = E_t / 4
    sigma_E_t1mt_times_theta = E_t1mt * theta
    sigma_Var_t = Var_t

    return MomentBasedSigmaCandidates(
        E_t=E_t,
        E_t2=E_t2,
        E_t1mt=E_t1mt,
        Var_t=Var_t,
        sigma_E_t1mt=sigma_E_t1mt,
        sigma_E_t_minus_half=sigma_E_t_minus_half,
        sigma_E_t_div4=sigma_E_t_div4,
        sigma_E_t1mt_times_theta=sigma_E_t1mt_times_theta,
        sigma_Var_t=sigma_Var_t,
        theta=theta,
        R=R,
        weight_type=weight_type,
    )


def compare_sigma_to_moments(
    sigma_empirical: float,
    moments: MomentBasedSigmaCandidates,
) -> Dict[str, Dict[str, float]]:
    """
    Compare empirical σ to moment-derived candidates.

    Returns dict with candidate name -> {value, diff, rel_diff_pct}
    sorted by closest match.
    """
    candidates = {
        "E[t(1-t)]": moments.sigma_E_t1mt,
        "E[t]-1/2": moments.sigma_E_t_minus_half,
        "E[t]/4": moments.sigma_E_t_div4,
        "E[t(1-t)]×θ": moments.sigma_E_t1mt_times_theta,
        "Var(t)": moments.sigma_Var_t,
    }

    results = {}
    for name, value in candidates.items():
        diff = sigma_empirical - value
        rel_diff = abs(diff) / abs(sigma_empirical) if sigma_empirical != 0 else float('inf')
        results[name] = {
            "value": value,
            "diff": diff,
            "rel_diff_pct": rel_diff * 100,
        }

    return dict(sorted(results.items(), key=lambda x: abs(x[1]["diff"])))


def run_moment_anti_overfit_probe(
    polys_kappa: Dict[str, PolyLike],
    polys_kappa_star: Dict[str, PolyLike],
    *,
    theta: float = 4.0 / 7.0,
    R_kappa: float = 1.3036,
    R_kappa_star: float = 1.1167,
    sigma_empirical: float = 5/32,
    n_quad: int = 200,
) -> Dict[str, Any]:
    """
    Run the moment-based anti-overfit probe for σ.

    Computes moment-derived σ candidates for both κ and κ* Q polynomials
    and compares them to the empirical σ = 5/32.

    Returns:
        Dict with:
        - kappa_moments: MomentBasedSigmaCandidates
        - kappa_star_moments: MomentBasedSigmaCandidates
        - kappa_comparison: sorted comparison dict
        - kappa_star_comparison: sorted comparison dict
        - best_match_kappa: (name, diff)
        - best_match_kappa_star: (name, diff)
        - verdict: "structural" if both match within 20%, else "Q-specific"
    """
    Q_kappa = polys_kappa["Q"]
    Q_kappa_star = polys_kappa_star["Q"]

    moments_k = compute_moment_based_sigma_candidates(
        theta=theta, R=R_kappa, Q=Q_kappa, n_quad=n_quad,
    )
    moments_ks = compute_moment_based_sigma_candidates(
        theta=theta, R=R_kappa_star, Q=Q_kappa_star, n_quad=n_quad,
    )

    comp_k = compare_sigma_to_moments(sigma_empirical, moments_k)
    comp_ks = compare_sigma_to_moments(sigma_empirical, moments_ks)

    # Get best matches
    best_k = list(comp_k.items())[0]
    best_ks = list(comp_ks.items())[0]

    # Verdict: if both best matches are within 20% relative, consider structural
    is_structural = (
        best_k[1]["rel_diff_pct"] < 20 and
        best_ks[1]["rel_diff_pct"] < 20 and
        best_k[0] == best_ks[0]  # Same moment type matches best
    )

    return {
        "kappa_moments": moments_k,
        "kappa_star_moments": moments_ks,
        "kappa_comparison": comp_k,
        "kappa_star_comparison": comp_ks,
        "best_match_kappa": (best_k[0], best_k[1]["diff"]),
        "best_match_kappa_star": (best_ks[0], best_ks[1]["diff"]),
        "sigma_empirical": sigma_empirical,
        "verdict": "structural" if is_structural else "Q-specific",
    }


# =============================================================================
# DERIVED AMPLITUDE FORMULA (GPT Run 2 Path (i))
# =============================================================================
# Based on analysis of A1, A2 residual amplitudes:
#   A₁ = exp(R) + (K-1) + ε
#   A₂ = exp(R) + 2(K-1) + ε
# where K = number of mollifier pieces (3 for PRZZ)
# and ε ≈ 0.27 is a small correction term.

@dataclass
class DerivedAmplitudeFormula:
    """Derived amplitude formula for I₁ and I₂ channels."""
    K: int  # Number of mollifier pieces
    R: float
    epsilon: float  # Correction term
    A1: float  # exp(R) + (K-1) + ε
    A2: float  # exp(R) + 2(K-1) + ε


def compute_derived_amplitude(
    R: float,
    *,
    theta: float = 4.0 / 7.0,
    sigma: float = 5.0 / 32.0,
    K: int = 3,
    epsilon: Optional[float] = None,
) -> DerivedAmplitudeFormula:
    """
    Compute derived amplitude using the formula:
        A₁ = exp(R) + (K-1) + ε
        A₂ = exp(R) + 2(K-1) + ε

    This formula was derived from the residual factorization analysis:
    - A₂ - A₁ = K - 1 (structural relationship)
    - A₁/A₂ ≈ 3/4 for K=3

    Args:
        R: R parameter (typically 1.3036 for κ, 1.1167 for κ*)
        K: Number of mollifier pieces (default 3)
        epsilon: Correction term. If omitted, uses ε = σ/θ (empirically ≈ 0.27).

    Returns:
        DerivedAmplitudeFormula with computed amplitudes
    """
    # Empirically, ε matches σ/θ very closely (e.g. 5/32 divided by 4/7 = 35/128 ≈ 0.2734).
    # Treat ε as a derived quantity by default, but allow explicit override for diagnostics.
    if epsilon is None:
        epsilon = float(sigma) / float(theta)

    exp_R = np.exp(R)
    A1 = exp_R + (K - 1) + float(epsilon)
    A2 = exp_R + 2 * (K - 1) + float(epsilon)

    return DerivedAmplitudeFormula(
        K=K,
        R=R,
        epsilon=epsilon,
        A1=A1,
        A2=A2,
    )


def compute_c_with_derived_amplitude(
    theta: float,
    R: float,
    polynomials: Dict[str, PolyLike],
    *,
    K: int = 3,
    epsilon: Optional[float] = None,
    sigma: float = 5/32,
    normalization: str = "grid",
    lift_scope: str = "i1_only",
    n: int = 60,
    n_quad_a: int = 40,
) -> Dict[str, Any]:
    """
    Compute c using derived amplitude formula (NO FITTING).

    This uses the structural decomposition:
        m₁ = m1_implied × A₁
        m₂ = m2_implied × A₂

    where:
        - m1_implied, m2_implied come from operator σ-shift (shape)
        - A₁ = exp(R) + (K-1) + ε (amplitude for I₁)
        - A₂ = exp(R) + 2(K-1) + ε (amplitude for I₂)

    This eliminates the 2×2 solve entirely.

    Args:
        theta: θ parameter
        R: R parameter
        polynomials: Dict with P1, P2, P3, Q
        K: Number of mollifier pieces (default 3)
        epsilon: Amplitude correction term. If omitted, uses ε = σ/θ.
        sigma: Shape shift (default 5/32)
        normalization: Grid normalization mode
        lift_scope: Operator scope (i1_only recommended)
        n: Quadrature points
        n_quad_a: Case C quadrature points

    Returns:
        Dict with c_computed, m1_full, m2_full, and intermediate values
    """
    # Get shape factors from operator mode
    implied = compute_operator_implied_weights(
        theta=theta, R=R, polynomials=polynomials,
        sigma=sigma, normalization=normalization, lift_scope=lift_scope,
        n=n, n_quad_a=n_quad_a,
    )

    # Get derived amplitudes
    amp = compute_derived_amplitude(
        R=R,
        theta=theta,
        sigma=sigma,
        K=K,
        epsilon=epsilon,
    )

    # Combine shape × amplitude
    m1_full = implied.m1_implied * amp.A1
    m2_full = implied.m2_implied * amp.A2

    # Compute c using the full weights
    c_computed = (
        implied.I1_plus + m1_full * implied.I1_minus_base +
        implied.I2_plus + m2_full * implied.I2_minus_base +
        implied.S34_plus
    )

    return {
        "c_computed": c_computed,
        "m1_implied": implied.m1_implied,
        "m2_implied": implied.m2_implied,
        "A1": amp.A1,
        "A2": amp.A2,
        "m1_full": m1_full,
        "m2_full": m2_full,
        "I1_plus": implied.I1_plus,
        "I2_plus": implied.I2_plus,
        "I1_minus_base": implied.I1_minus_base,
        "I2_minus_base": implied.I2_minus_base,
        "S34_plus": implied.S34_plus,
        "epsilon": amp.epsilon,
        "K": K,
        "R": R,
        "sigma": sigma,
    }


def validate_derived_amplitude_formula(
    polys_kappa: Dict[str, PolyLike],
    polys_kappa_star: Dict[str, PolyLike],
    *,
    theta: float = 4.0 / 7.0,
    R_kappa: float = 1.3036,
    R_kappa_star: float = 1.1167,
    c_target_kappa: float = 2.137,
    c_target_kappa_star: float = 1.938,
    K: int = 3,
    epsilon: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Validate the derived amplitude formula against both benchmarks.

    This is the key validation: does shape × amplitude reproduce targets?

    Returns:
        Dict with:
        - c_kappa, c_kappa_star: computed c values
        - c_gap_kappa, c_gap_kappa_star: gaps from targets
        - m1_full, m2_full: full weights (shape × amplitude)
        - verdict: "PASS" if both gaps < 5%, else "FAIL"
    """
    result_k = compute_c_with_derived_amplitude(
        theta=theta, R=R_kappa, polynomials=polys_kappa,
        K=K, epsilon=epsilon,
    )

    result_ks = compute_c_with_derived_amplitude(
        theta=theta, R=R_kappa_star, polynomials=polys_kappa_star,
        K=K, epsilon=epsilon,
    )

    c_gap_k = (result_k["c_computed"] - c_target_kappa) / c_target_kappa * 100
    c_gap_ks = (result_ks["c_computed"] - c_target_kappa_star) / c_target_kappa_star * 100

    verdict = "PASS" if abs(c_gap_k) < 5 and abs(c_gap_ks) < 5 else "FAIL"

    return {
        "c_kappa": result_k["c_computed"],
        "c_kappa_star": result_ks["c_computed"],
        "c_target_kappa": c_target_kappa,
        "c_target_kappa_star": c_target_kappa_star,
        "c_gap_kappa": c_gap_k,
        "c_gap_kappa_star": c_gap_ks,
        "m1_full_kappa": result_k["m1_full"],
        "m2_full_kappa": result_k["m2_full"],
        "m1_full_kappa_star": result_ks["m1_full"],
        "m2_full_kappa_star": result_ks["m2_full"],
        "A1_kappa": result_k["A1"],
        "A2_kappa": result_k["A2"],
        "A1_kappa_star": result_ks["A1"],
        "A2_kappa_star": result_ks["A2"],
        "epsilon": result_k["epsilon"],
        "K": K,
        "verdict": verdict,
    }


# =============================================================================
# CODEX TASK 2: tex_amplitudes() - First-Class Amplitude Function
# =============================================================================
# This is the canonical API for computing TeX-derived amplitudes.
# All amplitude calculations should use this function.

@dataclass
class TexAmplitudeResult:
    """Result of TeX-derived amplitude computation."""
    R: float
    K: int
    theta: float
    sigma: float
    epsilon: float
    A1: float  # exp(R) + (K-1) + ε
    A2: float  # exp(R) + 2(K-1) + ε
    A_diff: float  # A2 - A1 = K - 1
    A_ratio: float  # A1 / A2 ≈ (K-1+1)/(2K-1+1) = K/(2K)
    # Diagnostic integrals (for validation)
    diagnostics: Dict[str, Any]


def tex_amplitudes(
    theta: float,
    R: float,
    K: int = 3,
    polynomials: Optional[Dict[str, PolyLike]] = None,
    *,
    sigma: float = 5 / 32,
    epsilon: Optional[float] = None,
    exp_component: str = "exp_R",
    R_ref: float = 1.3036,
    compute_diagnostics: bool = True,
    n_quad: int = 200,
) -> TexAmplitudeResult:
    """
    Compute TeX-derived amplitudes A1 and A2.

    This is the first-class API for amplitude computation (Codex Task 2).
    The formula is derived from TeX lines 1502-1548:

        A₁ = exp_component_value + (K-1) + ε
        A₂ = exp_component_value + 2(K-1) + ε

    Key structural relationships:
        A₂ - A₁ = K - 1 (exact, regardless of exp_component)
        A₁/A₂ ≈ K / (2K) = 1/2 for large K; ≈ 3/4 for K=3

    Args:
        theta: θ parameter (typically 4/7)
        R: R parameter (e.g., 1.3036 for κ, 1.1167 for κ*)
        K: Number of mollifier pieces (default 3)
        polynomials: Optional dict with Q polynomial for diagnostic integrals
        sigma: Shape shift parameter (default 5/32)
        epsilon: Correction term. If None, computed as σ/θ.
        exp_component: Which exp-related quantity to use in amplitude:
            - "exp_R": exp(R) (benchmark-specific surrogate)
            - "exp_R_ref": exp(R_ref) (fixed reference, default R_ref=1.3036)
            - "E_exp2Rt_under_Q2": E[exp(2Rt)] under Q² weight (TeX-motivated)
            - "uniform_avg": (exp(2R)-1)/(2R) (uniform t average)
        R_ref: Reference R value for "exp_R_ref" mode (default 1.3036 = R_κ)
        compute_diagnostics: Whether to compute intermediate integrals
        n_quad: Quadrature points for diagnostics

    Returns:
        TexAmplitudeResult with A1, A2, and diagnostic information

    Note:
        GPT Run 4 finding: Using exp_R_ref mode with R_ref=1.3036 (the κ benchmark R)
        gives excellent accuracy (<0.3%) on BOTH κ and κ* benchmarks, whereas
        exp_R mode gives 0.7% error for κ but 8% error for κ*.

    Warning:
        The exp_R_ref mode is a CALIBRATION fix, not TeX-derived.
        It matches both PRZZ benchmarks but has no direct TeX justification.
        Use for benchmark reproduction, not as ground truth.

        The exp(R_ref) choice freezes the amplitude at a single R value (1.3036),
        which makes R-sweep results worse. This is expected because the amplitude
        formula was calibrated at R = 1.3036.

        GPT Run 5 guidance: Treat exp_R_ref as a calibrated stopgap. Do NOT declare
        victory on "reproduced PRZZ from first principles" until a specific TeX step
        justifies this choice (or until direct TeX evaluation removes the need for it).
    """
    from src.quadrature import gauss_legendre_01

    # Validate exp_component
    valid_exp_components = {"exp_R", "exp_R_ref", "E_exp2Rt_under_Q2", "uniform_avg"}
    if exp_component not in valid_exp_components:
        raise ValueError(
            f"exp_component must be one of {valid_exp_components}, got '{exp_component}'"
        )

    # Compute epsilon from structural relationship if not provided
    if epsilon is None:
        epsilon = float(sigma) / float(theta)

    exp_R = np.exp(R)

    # Compute diagnostic integrals (needed for some exp_component modes)
    diagnostics: Dict[str, Any] = {
        "exp_R": exp_R,
        "K_minus_1": K - 1,
        "epsilon": epsilon,
        "exp_component_mode": exp_component,
    }

    # Always compute uniform_avg (doesn't need Q)
    uniform_avg = (np.exp(2 * R) - 1) / (2 * R)
    diagnostics["uniform_avg"] = float(uniform_avg)
    diagnostics["exp_2R"] = float(np.exp(2 * R))

    # Compute Q-weighted moment if Q is available
    E_exp2Rt = None
    if polynomials is not None and "Q" in polynomials:
        Q = polynomials["Q"]
        nodes, weights = gauss_legendre_01(n_quad)

        # Evaluate Q at nodes
        if hasattr(Q, "eval"):
            Q_vals = Q.eval(nodes)
        elif hasattr(Q, "to_monomial"):
            Q_mono = Q.to_monomial()
            Q_coeffs = list(np.asarray(Q_mono.coeffs, dtype=float))
            Q_vals = np.array([sum(c * (t ** j) for j, c in enumerate(Q_coeffs)) for t in nodes])
        else:
            Q_coeffs = list(np.asarray(Q.coeffs, dtype=float))
            Q_vals = np.array([sum(c * (t ** j) for j, c in enumerate(Q_coeffs)) for t in nodes])

        # TeX-motivated integrals (from line 1548)
        # The mirror amplitude involves exp(2Rt) inside the integral
        base_integral = np.sum(Q_vals**2 * weights)
        weighted_integral = np.sum(Q_vals**2 * np.exp(2 * R * nodes) * weights)
        E_exp2Rt = weighted_integral / base_integral if base_integral != 0 else 0

        if compute_diagnostics:
            diagnostics["base_integral_Q2"] = float(base_integral)
            diagnostics["weighted_integral_Q2_exp2Rt"] = float(weighted_integral)
        diagnostics["E_exp2Rt_under_Q2"] = float(E_exp2Rt)

    # Select exp_component_value based on mode
    if exp_component == "exp_R":
        exp_component_value = exp_R
    elif exp_component == "exp_R_ref":
        # Use fixed reference R (default 1.3036 = R_κ)
        # GPT Run 4 finding: this gives <0.3% error on BOTH benchmarks
        exp_component_value = np.exp(R_ref)
        diagnostics["R_ref"] = float(R_ref)
        diagnostics["exp_R_ref"] = float(exp_component_value)
    elif exp_component == "uniform_avg":
        exp_component_value = uniform_avg
    elif exp_component == "E_exp2Rt_under_Q2":
        if E_exp2Rt is None:
            raise ValueError(
                "exp_component='E_exp2Rt_under_Q2' requires polynomials['Q'] to be provided"
            )
        exp_component_value = E_exp2Rt
    else:
        # Should not reach here due to validation above
        exp_component_value = exp_R

    diagnostics["exp_component_value"] = float(exp_component_value)

    # Compute amplitudes using the selected exp component
    # Key: A2 - A1 = K - 1 is preserved regardless of exp_component
    A1 = exp_component_value + (K - 1) + epsilon
    A2 = exp_component_value + 2 * (K - 1) + epsilon

    A_diff = A2 - A1  # Should equal K - 1
    A_ratio = A1 / A2 if A2 != 0 else float('inf')

    return TexAmplitudeResult(
        R=R,
        K=K,
        theta=theta,
        sigma=sigma,
        epsilon=epsilon,
        A1=A1,
        A2=A2,
        A_diff=A_diff,
        A_ratio=A_ratio,
        diagnostics=diagnostics,
    )


# =============================================================================
# CODEX TASK 1: compute_c_paper_tex_mirror() - TeX-Derived Mirror Evaluator
# =============================================================================
# This is the aspirational truth evaluator: uses TeX-derived structure,
# NOT benchmark fitting.

@dataclass
class TexMirrorResult:
    """Result of TeX-derived mirror evaluation."""
    c: float
    # Channel breakdown
    I1_plus: float
    I2_plus: float
    S34_plus: float
    I1_minus_base: float
    I2_minus_base: float
    I1_minus_shape: float
    I2_minus_shape: float
    # Weights
    m1: float  # A1 × (I1_minus_shape / I1_minus_base)
    m2: float  # A2 × (I2_minus_shape / I2_minus_base)
    m1_implied: float  # Shape factor only
    m2_implied: float  # Shape factor only
    # Amplitudes
    A1: float
    A2: float
    # Metadata
    R: float
    theta: float
    K: int
    sigma: float
    epsilon: float
    normalization: str


def compute_c_paper_tex_mirror(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict[str, PolyLike],
    *,
    K: int = 3,
    sigma: float = 5 / 32,
    epsilon: Optional[float] = None,
    tex_exp_component: str = "exp_R",
    tex_R_ref: float = 1.3036,
    normalization: str = "grid",
    lift_scope: str = "i1_only",
    use_factorial_normalization: bool = True,
    n_quad_a: int = 40,
    verbose: bool = False,
    i1_source: str = "dsl",  # GPT Phase 1: "dsl" or "post_identity_operator"
    i2_source: str = "dsl",  # Run 10B: "dsl" or "direct_case_c"
    terms_version: str = "old",  # Run 10A: "old" or "v2"
) -> TexMirrorResult:
    """
    Compute c using TeX-derived mirror assembly (NO FITTING).

    This is the aspirational truth evaluator (Codex Task 1). It:
    - Uses ORDERED pairs (paper truth)
    - Uses σ-shift only as SHAPE operator (i1_only + grid norm)
    - Uses TeX-derived AMPLITUDE (A1, A2)
    - Does NOT call any 2×2 solve

    Assembly formula:
        c = I1(+R) + m1×I1_base(-R) + I2(+R) + m2×I2_base(-R) + S34(+R)

    where:
        m1 = A1 × m1_implied = A1 × (I1_op(-R) / I1_base(-R))
        m2 = A2 × m2_implied = A2 × (I2_op(-R) / I2_base(-R))

    The direct (+R) branch is IDENTICAL to compute_c_paper() (ordered).
    The operator stuff touches MIRROR ONLY.

    Args:
        theta: θ parameter (typically 4/7)
        R: R parameter
        n: Number of quadrature points per dimension
        polynomials: Dict with keys "P1", "P2", "P3", "Q"
        K: Number of mollifier pieces (default 3)
        sigma: Shape shift magnitude (default 5/32)
        epsilon: Amplitude correction. If None, uses ε = σ/θ.
        tex_exp_component: Which exp quantity to use in amplitude formula:
            - "exp_R": exp(R) (benchmark-specific surrogate)
            - "exp_R_ref": exp(tex_R_ref) (fixed reference, default 1.3036)
            - "E_exp2Rt_under_Q2": E[exp(2Rt)] under Q² weight
            - "uniform_avg": (exp(2R)-1)/(2R) (uniform t average)
        tex_R_ref: Reference R for "exp_R_ref" mode (default 1.3036 = R_κ)
        normalization: Q-lift normalization ("grid" recommended)
        lift_scope: Operator scope ("i1_only" recommended)
        use_factorial_normalization: Apply 1/(ℓ₁!×ℓ₂!) normalization
        n_quad_a: Quadrature points for Case C a-integral
        verbose: Print diagnostic information
        i1_source: Source for I1 computation:
            - "dsl" (default): Use DSL-based term evaluation
            - "post_identity_operator": Use post-identity operator (GPT Phase 1)
        i2_source: Source for I2 computation:
            - "dsl" (default): Use DSL-based term evaluation
            - "direct_case_c": Use proven Case C kernel evaluation

    Returns:
        TexMirrorResult with c and full breakdown

    Note:
        This function does NOT call solve_two_weight_operator or any fitting.
        Use solve_two_weight_operator_diagnostic() for cross-validation.

        GPT Run 4 finding: Using tex_exp_component="exp_R_ref" with tex_R_ref=1.3036
        gives <0.3% accuracy on BOTH κ and κ* benchmarks.

    Classification:
        This is an ASPIRATIONAL evaluator, not proven PRZZ reproduction.
        - Shape (m_implied): Operator-lift experiment, not TeX formula
        - Amplitude (A): TeX-motivated surrogate, not direct evaluation
        The exp_R_ref mode is CALIBRATION only.

        Proven components:
        - Ordered pairs required (S34 asymmetry test)
        - S12 symmetry holds (S12 symmetry test)
        - Direct (+R) branch matches compute_c_paper()

        Aspirational components (use with care):
        - tex_amplitudes() formula: A = exp(...) + K-1 + ε
        - exp_R_ref mode: calibrated stopgap, NOT TeX-derived
        - Mirror weight assembly: shape × amplitude factorization

        GPT Run 5 path forward: Replace amplitude model with direct TeX I2 evaluation.
    """
    # GPT Run 14: Hard guard against V2 + tex_mirror combination
    # V2 terms are proven correct individually (Run 9-11) but catastrophically fail
    # under tex_mirror assembly (I1_plus flips sign from +0.085 to -0.111).
    # This causes c to collapse to ~0.775 vs target ~2.137.
    # Use OLD + tex_mirror for production. See GPT Run 12-13 findings.
    if terms_version == "v2":
        raise ValueError(
            "FATAL: terms_version='v2' is FORBIDDEN with tex_mirror. "
            "V2 terms catastrophically fail under mirror assembly (I1_plus sign flip). "
            "Use terms_version='old' (default). See docs/HANDOFF_GPT_RUN12_13.md."
        )

    # Get implied weights from operator mode
    implied = compute_operator_implied_weights(
        theta=theta,
        R=R,
        polynomials=polynomials,
        sigma=sigma,
        normalization=normalization,
        lift_scope=lift_scope,
        n=n,
        n_quad_a=n_quad_a,
        verbose=verbose,
        i1_source=i1_source,  # GPT Phase 1
        i2_source=i2_source,  # Run 10B
        terms_version=terms_version,  # Run 10A
    )

    # Get TeX-derived amplitudes
    # Enable diagnostics when using E_exp2Rt_under_Q2 mode
    needs_diagnostics = tex_exp_component in {"E_exp2Rt_under_Q2", "exp_R_ref"}
    amp = tex_amplitudes(
        theta=theta,
        R=R,
        K=K,
        polynomials=polynomials,
        sigma=sigma,
        epsilon=epsilon,
        exp_component=tex_exp_component,
        R_ref=tex_R_ref,
        compute_diagnostics=needs_diagnostics,
    )

    # Compute full weights: m = A × m_implied
    m1 = amp.A1 * implied.m1_implied
    m2 = amp.A2 * implied.m2_implied

    # Assembly: c = I1(+) + m1×I1_base(-) + I2(+) + m2×I2_base(-) + S34(+)
    c = (
        implied.I1_plus +
        m1 * implied.I1_minus_base +
        implied.I2_plus +
        m2 * implied.I2_minus_base +
        implied.S34_plus
    )

    if verbose:
        print(f"\n{'='*70}")
        print("TEX-DERIVED MIRROR ASSEMBLY (NO FITTING)")
        print(f"{'='*70}")
        print(f"R = {R:.4f}, K = {K}, σ = {sigma:.6f}, ε = {amp.epsilon:.6f}")
        print(f"exp_component = {tex_exp_component}")
        exp_val = amp.diagnostics.get("exp_component_value", np.exp(R))
        print(f"exp_component_value = {exp_val:.6f}")
        print()
        print("Shape (m_implied):")
        print(f"  m1_implied = {implied.m1_implied:.6f}")
        print(f"  m2_implied = {implied.m2_implied:.6f}")
        print()
        print("Amplitude (A from TeX):")
        print(f"  A1 = exp_val + {K-1} + ε = {amp.A1:.6f}")
        print(f"  A2 = exp_val + {2*(K-1)} + ε = {amp.A2:.6f}")
        print()
        print("Full weights (m = A × m_implied):")
        print(f"  m1 = {m1:.6f}")
        print(f"  m2 = {m2:.6f}")
        print()
        print("Channel values:")
        print(f"  I1_plus:       {implied.I1_plus:+.6f}")
        print(f"  I1_minus_base: {implied.I1_minus_base:+.6f}")
        print(f"  I2_plus:       {implied.I2_plus:+.6f}")
        print(f"  I2_minus_base: {implied.I2_minus_base:+.6f}")
        print(f"  S34_plus:      {implied.S34_plus:+.6f}")
        print()
        print(f"c = {c:.6f}")

    return TexMirrorResult(
        c=c,
        I1_plus=implied.I1_plus,
        I2_plus=implied.I2_plus,
        S34_plus=implied.S34_plus,
        I1_minus_base=implied.I1_minus_base,
        I2_minus_base=implied.I2_minus_base,
        I1_minus_shape=implied.I1_minus_op,
        I2_minus_shape=implied.I2_minus_op,
        m1=m1,
        m2=m2,
        m1_implied=implied.m1_implied,
        m2_implied=implied.m2_implied,
        A1=amp.A1,
        A2=amp.A2,
        R=R,
        theta=theta,
        K=K,
        sigma=sigma,
        epsilon=amp.epsilon,
        normalization=normalization,
    )


def validate_tex_mirror_against_diagnostic(
    polys_kappa: Dict[str, PolyLike],
    polys_kappa_star: Dict[str, PolyLike],
    *,
    theta: float = 4.0 / 7.0,
    R_kappa: float = 1.3036,
    R_kappa_star: float = 1.1167,
    c_target_kappa: float = 2.137,
    c_target_kappa_star: float = 1.938,
    K: int = 3,
    sigma: float = 5 / 32,
    epsilon: Optional[float] = None,
    n: int = 60,
    n_quad_a: int = 40,
) -> Dict[str, Any]:
    """
    Cross-check TeX-mirror evaluator against diagnostic 2×2 solve.

    This is the hard gate for Codex Task 3: verify that TeX-derived
    weights are close to diagnostic-solved weights.

    Acceptance criteria:
        - Derived m1,m2 within 10% of solved m1,m2
        - Residuals A1/A2 are benchmark-stable (spans < 2%)

    Returns:
        Dict with comparison metrics and verdict
    """
    import warnings

    # Get TeX-derived results
    tex_k = compute_c_paper_tex_mirror(
        theta=theta, R=R_kappa, n=n, polynomials=polys_kappa,
        K=K, sigma=sigma, epsilon=epsilon, n_quad_a=n_quad_a,
    )
    tex_ks = compute_c_paper_tex_mirror(
        theta=theta, R=R_kappa_star, n=n, polynomials=polys_kappa_star,
        K=K, sigma=sigma, epsilon=epsilon, n_quad_a=n_quad_a,
    )

    # Get diagnostic solved weights (with warning suppression)
    result_k = compute_c_paper_operator_v2(
        theta=theta, R=R_kappa, n=n, polynomials=polys_kappa,
        n_quad_a=n_quad_a, normalization="grid", lift_scope="i1_only", sigma=sigma,
    )
    result_ks = compute_c_paper_operator_v2(
        theta=theta, R=R_kappa_star, n=n, polynomials=polys_kappa_star,
        n_quad_a=n_quad_a, normalization="grid", lift_scope="i1_only", sigma=sigma,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solved = solve_two_weight_operator(
            result_k, result_ks,
            c_target_k=c_target_kappa, c_target_k_star=c_target_kappa_star,
            use_operator_channels=True,
        )

    m1_solved = solved["m1"]
    m2_solved = solved["m2"]

    # Compare derived vs solved
    m1_diff = abs(tex_k.m1 - m1_solved) / abs(m1_solved) * 100 if m1_solved != 0 else float('inf')
    m2_diff = abs(tex_k.m2 - m2_solved) / abs(m2_solved) * 100 if m2_solved != 0 else float('inf')

    # Compute residuals (solved / implied)
    implied_k = compute_operator_implied_weights(
        theta=theta, R=R_kappa, polynomials=polys_kappa,
        sigma=sigma, normalization="grid", lift_scope="i1_only", n=n, n_quad_a=n_quad_a,
    )
    implied_ks = compute_operator_implied_weights(
        theta=theta, R=R_kappa_star, polynomials=polys_kappa_star,
        sigma=sigma, normalization="grid", lift_scope="i1_only", n=n, n_quad_a=n_quad_a,
    )

    A1_k = m1_solved / implied_k.m1_implied if implied_k.m1_implied != 0 else float('inf')
    A2_k = m2_solved / implied_k.m2_implied if implied_k.m2_implied != 0 else float('inf')
    A1_ks = m1_solved / implied_ks.m1_implied if implied_ks.m1_implied != 0 else float('inf')
    A2_ks = m2_solved / implied_ks.m2_implied if implied_ks.m2_implied != 0 else float('inf')

    A1_avg = (A1_k + A1_ks) / 2
    A2_avg = (A2_k + A2_ks) / 2
    A1_span = abs(A1_k - A1_ks) / A1_avg * 100 if A1_avg != 0 else float('inf')
    A2_span = abs(A2_k - A2_ks) / A2_avg * 100 if A2_avg != 0 else float('inf')

    # Compute c gaps
    c_gap_k = (tex_k.c - c_target_kappa) / c_target_kappa * 100
    c_gap_ks = (tex_ks.c - c_target_kappa_star) / c_target_kappa_star * 100

    # Verdict
    m_within_tolerance = m1_diff < 10 and m2_diff < 10
    A_stable = A1_span < 5 and A2_span < 5
    c_acceptable = abs(c_gap_k) < 5 and abs(c_gap_ks) < 5

    verdict = "PASS" if m_within_tolerance and A_stable and c_acceptable else "FAIL"

    return {
        # TeX-derived results
        "c_tex_kappa": tex_k.c,
        "c_tex_kappa_star": tex_ks.c,
        "m1_tex": tex_k.m1,
        "m2_tex": tex_k.m2,
        "A1_tex": tex_k.A1,
        "A2_tex": tex_k.A2,
        # Diagnostic solved results
        "m1_solved": m1_solved,
        "m2_solved": m2_solved,
        # Comparisons
        "m1_diff_pct": m1_diff,
        "m2_diff_pct": m2_diff,
        # Residual stability
        "A1_kappa": A1_k,
        "A2_kappa": A2_k,
        "A1_kappa_star": A1_ks,
        "A2_kappa_star": A2_ks,
        "A1_span_pct": A1_span,
        "A2_span_pct": A2_span,
        # c gaps
        "c_gap_kappa_pct": c_gap_k,
        "c_gap_kappa_star_pct": c_gap_ks,
        # Targets
        "c_target_kappa": c_target_kappa,
        "c_target_kappa_star": c_target_kappa_star,
        # Verdict
        "verdict": verdict,
        "m_within_tolerance": m_within_tolerance,
        "A_stable": A_stable,
        "c_acceptable": c_acceptable,
    }


# =============================================================================
# Run 18: TeX Combined Integral Structure
# =============================================================================

@dataclass
class TexCombinedResult:
    """Result from TeX combined integral computation."""
    I1_combined: float
    scalar_limit: float
    n_quad: int
    n_quad_s: int


def compute_I1_tex_combined_11(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict[str, PolyLike],
    n_quad_s: int = 20,
    verbose: bool = False,
) -> TexCombinedResult:
    """
    Compute I1 for (1,1) using TeX combined integral structure.

    This implements Stage 18B of Run 18: The TeX combined structure from
    lines 1503-1510, where the mirror difference is converted to an integral
    BEFORE applying Q and P operators.

    The combined structure is:
        (1 + θ(x+y)) × ∫_0^1 exp(2sR(1 + θ(x+y))) ds

    This replaces the naive I(+R) + exp(2R)×I(-R) assembly.

    Args:
        theta: θ parameter (typically 4/7)
        R: R parameter
        n: Number of quadrature points for (u, t)
        polynomials: Dict with "P1", "Q"
        n_quad_s: Quadrature points for s-integral
        verbose: Print debug info

    Returns:
        TexCombinedResult with I1_combined value
    """
    from src.quadrature import tensor_grid_2d
    from src.term_dsl import SeriesContext, CombinedMirrorFactor, AffineExpr
    from src.composition import compose_polynomial_on_affine, compose_exp_on_affine

    # Get polynomials
    P1 = polynomials["P1"]
    Q = polynomials["Q"]

    # Build quadrature grid
    U, T, W = tensor_grid_2d(n)

    # Create series context with x, y variables
    ctx = SeriesContext(var_names=("x", "y"))

    # Build the CombinedMirrorFactor
    combined_factor = CombinedMirrorFactor(R=R, theta=theta, n_quad_s=n_quad_s)
    combined_series = combined_factor.evaluate(U, T, ctx)

    # Build P₁(x+u) series
    P1_x_u0 = U  # base: u
    P1_x_lin = {"x": np.ones_like(U)}  # x coefficient = 1
    P1_x_series = compose_polynomial_on_affine(P1, P1_x_u0, P1_x_lin, ctx.var_names)

    # Build P₁(y+u) series
    P1_y_u0 = U  # base: u
    P1_y_lin = {"y": np.ones_like(U)}  # y coefficient = 1
    P1_y_series = compose_polynomial_on_affine(P1, P1_y_u0, P1_y_lin, ctx.var_names)

    # Build Q(Arg_α) series
    # Arg_α = t + θ·t·x + (θ·t-θ)·y
    Q_alpha_u0 = T
    Q_alpha_lin = {
        "x": theta * T,
        "y": theta * T - theta,
    }
    Q_alpha_series = compose_polynomial_on_affine(Q, Q_alpha_u0, Q_alpha_lin, ctx.var_names)

    # Build Q(Arg_β) series
    # Arg_β = t + (θ·t-θ)·x + θ·t·y
    Q_beta_u0 = T
    Q_beta_lin = {
        "x": theta * T - theta,
        "y": theta * T,
    }
    Q_beta_series = compose_polynomial_on_affine(Q, Q_beta_u0, Q_beta_lin, ctx.var_names)

    # Build exp(R·Arg_α) × exp(R·Arg_β) = exp(R(Arg_α + Arg_β))
    # Combined argument: Arg_α + Arg_β = 2t + (2θt - θ)(x + y)
    exp_combined_u0 = 2 * R * T
    exp_combined_lin = {
        "x": R * (2 * theta * T - theta),
        "y": R * (2 * theta * T - theta),
    }
    exp_combined_series = compose_exp_on_affine(1.0, exp_combined_u0, exp_combined_lin, ctx.var_names)

    # Build algebraic prefactor: (1/θ + x + y)
    prefactor_series = ctx.scalar_series(np.ones_like(U) / theta)
    prefactor_series = prefactor_series + ctx.variable_series("x")
    prefactor_series = prefactor_series + ctx.variable_series("y")

    # Multiply all series together
    # Structure: combined_mirror × exp × P₁ × P₁ × Q × Q × prefactor
    integrand = combined_series
    integrand = integrand * exp_combined_series
    integrand = integrand * P1_x_series
    integrand = integrand * P1_y_series
    integrand = integrand * Q_alpha_series
    integrand = integrand * Q_beta_series
    integrand = integrand * prefactor_series

    # Extract d²/dxdy coefficient (mask = 0b11 = 3 for vars ("x", "y"))
    xy_coeff = integrand.extract(("x", "y"))

    # Apply poly prefactor: (1-u)²
    poly_prefactor = (1 - U) ** 2
    xy_coeff = xy_coeff * poly_prefactor

    # Integrate over (u, t)
    I1_combined = np.sum(W * xy_coeff)

    if verbose:
        scalar_val = combined_factor.scalar_limit()
        print(f"I1_tex_combined_11:")
        print(f"  CombinedMirrorFactor scalar limit: {scalar_val:.6f}")
        print(f"  I1_combined: {I1_combined:.6f}")

    return TexCombinedResult(
        I1_combined=I1_combined,
        scalar_limit=combined_factor.scalar_limit(),
        n_quad=n,
        n_quad_s=n_quad_s,
    )


def compute_I1_tex_combined_11_replace(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict[str, PolyLike],
    n_quad_s: int = 20,
    verbose: bool = False,
) -> TexCombinedResult:
    """
    Compute I1 for (1,1) using TeX combined structure REPLACING exp factors.

    This version interprets the TeX combined structure as replacing the
    exp(R·Arg_α) × exp(R·Arg_β) factors entirely, rather than multiplying
    in addition to them.

    The combined structure is:
        (1 + θ(x+y)) × ∫_0^1 exp(2sR(1 + θ(x+y))) ds

    Args:
        theta: θ parameter (typically 4/7)
        R: R parameter
        n: Number of quadrature points for (u, t)
        polynomials: Dict with "P1", "Q"
        n_quad_s: Quadrature points for s-integral
        verbose: Print debug info

    Returns:
        TexCombinedResult with I1_combined value
    """
    from src.quadrature import tensor_grid_2d
    from src.term_dsl import SeriesContext, CombinedMirrorFactor

    # Get polynomials
    P1 = polynomials["P1"]
    Q = polynomials["Q"]

    # Build quadrature grid
    U, T, W = tensor_grid_2d(n)

    # Create series context with x, y variables
    ctx = SeriesContext(var_names=("x", "y"))

    # Build the CombinedMirrorFactor - this REPLACES exp factors
    combined_factor = CombinedMirrorFactor(R=R, theta=theta, n_quad_s=n_quad_s)
    combined_series = combined_factor.evaluate(U, T, ctx)

    # Build P₁(x+u) series
    from src.composition import compose_polynomial_on_affine
    P1_x_u0 = U
    P1_x_lin = {"x": np.ones_like(U)}
    P1_x_series = compose_polynomial_on_affine(P1, P1_x_u0, P1_x_lin, ctx.var_names)

    # Build P₁(y+u) series
    P1_y_u0 = U
    P1_y_lin = {"y": np.ones_like(U)}
    P1_y_series = compose_polynomial_on_affine(P1, P1_y_u0, P1_y_lin, ctx.var_names)

    # Build Q(Arg_α) series - evaluated at t (no exp factors!)
    Q_alpha_u0 = T
    Q_alpha_lin = {
        "x": theta * T,
        "y": theta * T - theta,
    }
    Q_alpha_series = compose_polynomial_on_affine(Q, Q_alpha_u0, Q_alpha_lin, ctx.var_names)

    # Build Q(Arg_β) series
    Q_beta_u0 = T
    Q_beta_lin = {
        "x": theta * T - theta,
        "y": theta * T,
    }
    Q_beta_series = compose_polynomial_on_affine(Q, Q_beta_u0, Q_beta_lin, ctx.var_names)

    # Build algebraic prefactor: (1/θ + x + y)
    prefactor_series = ctx.scalar_series(np.ones_like(U) / theta)
    prefactor_series = prefactor_series + ctx.variable_series("x")
    prefactor_series = prefactor_series + ctx.variable_series("y")

    # Multiply all series together - NO exp factors, combined replaces them
    # Structure: combined_mirror × P₁ × P₁ × Q × Q × prefactor
    integrand = combined_series
    integrand = integrand * P1_x_series
    integrand = integrand * P1_y_series
    integrand = integrand * Q_alpha_series
    integrand = integrand * Q_beta_series
    integrand = integrand * prefactor_series

    # Extract d²/dxdy coefficient
    xy_coeff = integrand.extract(("x", "y"))

    # Apply poly prefactor: (1-u)²
    poly_prefactor = (1 - U) ** 2
    xy_coeff = xy_coeff * poly_prefactor

    # Integrate over (u, t)
    I1_combined = np.sum(W * xy_coeff)

    if verbose:
        scalar_val = combined_factor.scalar_limit()
        print(f"I1_tex_combined_11_replace:")
        print(f"  CombinedMirrorFactor scalar limit: {scalar_val:.6f}")
        print(f"  I1_combined (replace exp): {I1_combined:.6f}")

    return TexCombinedResult(
        I1_combined=I1_combined,
        scalar_limit=combined_factor.scalar_limit(),
        n_quad=n,
        n_quad_s=n_quad_s,
    )


# =============================================================================
# Stage 18C: I2 Channel with Combined Structure
# =============================================================================

@dataclass
class TexCombinedI2Result:
    """Result from I2 combined computation."""
    I2_combined: float
    I2_base: float
    combined_factor_scalar: float
    n_quad: int


def compute_I2_tex_combined_11(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict[str, PolyLike],
    verbose: bool = False,
) -> TexCombinedI2Result:
    """
    Compute I2 for (1,1) using TeX combined structure.

    I2 has no formal variables (x, y), so the combined mirror factor
    reduces to its scalar limit: (exp(2R) - 1) / (2R).

    For I2, the combined structure effectively multiplies the base
    integral by this scalar.

    Args:
        theta: θ parameter (typically 4/7)
        R: R parameter
        n: Number of quadrature points for (u, t)
        polynomials: Dict with "P1", "Q"
        verbose: Print debug info

    Returns:
        TexCombinedI2Result with I2_combined value
    """
    from src.quadrature import tensor_grid_2d
    from src.term_dsl import CombinedMirrorFactor

    # Get polynomials
    P1 = polynomials["P1"]
    Q = polynomials["Q"]

    # Build quadrature grid
    U, T, W = tensor_grid_2d(n)

    # Compute base I2 integrand (no formal variables, just scalar integral)
    # Structure: (1/θ) × P₁(u)² × Q(t)² × exp(2Rt)
    P1_u = P1.eval(U)
    Q_t = Q.eval(T)
    exp_2Rt = np.exp(2 * R * T)

    I2_integrand = (1.0 / theta) * P1_u * P1_u * Q_t * Q_t * exp_2Rt

    # Integrate over (u, t)
    I2_base = np.sum(W * I2_integrand)

    # For I2 (no x, y variables), the combined mirror factor is just the scalar limit
    combined_factor = CombinedMirrorFactor(R=R, theta=theta)
    combined_scalar = combined_factor.scalar_limit()

    # I2 combined = I2_base × combined_factor_scalar
    I2_combined = I2_base * combined_scalar

    if verbose:
        print(f"I2_tex_combined_11:")
        print(f"  I2_base: {I2_base:.6f}")
        print(f"  combined_factor_scalar: {combined_scalar:.6f}")
        print(f"  I2_combined: {I2_combined:.6f}")

    return TexCombinedI2Result(
        I2_combined=I2_combined,
        I2_base=I2_base,
        combined_factor_scalar=combined_scalar,
        n_quad=n,
    )


# =============================================================================
# Stage 18D: S34 Channel with Combined Structure
# =============================================================================

@dataclass
class TexCombinedS34Result:
    """Result from S34 combined computation."""
    I3_combined: float
    I4_combined: float
    S34_combined: float
    n_quad: int
    n_quad_s: int


def compute_S34_tex_combined_11(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict[str, PolyLike],
    n_quad_s: int = 20,
    verbose: bool = False,
    apply_mirror: bool = False,  # SPEC LOCK: Must remain False
) -> TexCombinedS34Result:
    """
    Compute S34 (I3 + I4) for (1,1) using TeX combined structure.

    I3 has derivative d/dx (y=0), I4 has derivative d/dy (x=0).
    For the combined structure:
    - I3: uses (1 + θx) × ∫ exp(2sR(1 + θx)) ds with y=0
    - I4: uses (1 + θy) × ∫ exp(2sR(1 + θy)) ds with x=0

    S34 = I3 + I4.

    SPEC LOCK: I3/I4 do NOT have mirror structure (TRUTH_SPEC.md Section 10).
    The apply_mirror parameter exists only to catch accidental misuse.
    Setting apply_mirror=True will RAISE I34MirrorForbiddenError.

    Args:
        theta: θ parameter (typically 4/7)
        R: R parameter
        n: Number of quadrature points for (u, t)
        polynomials: Dict with "P1", "Q"
        n_quad_s: Quadrature points for s-integral
        verbose: Print debug info
        apply_mirror: FORBIDDEN - raises if True (spec lock)

    Returns:
        TexCombinedS34Result with I3, I4, and S34 values

    Raises:
        I34MirrorForbiddenError: If apply_mirror=True
    """
    _assert_i34_no_mirror(apply_mirror, "compute_S34_tex_combined_11")
    from src.quadrature import tensor_grid_2d, gauss_legendre_01
    from src.term_dsl import SeriesContext
    from src.composition import compose_polynomial_on_affine, compose_exp_on_affine

    P1 = polynomials["P1"]
    Q = polynomials["Q"]

    U, T, W = tensor_grid_2d(n)
    s_nodes, s_weights = gauss_legendre_01(n_quad_s)

    # ===== I3 (d/dx, y=0) =====
    # Single variable x
    ctx_x = SeriesContext(var_names=("x",))

    # Build combined factor for x only: (1 + θx) × ∫ exp(2sR(1 + θx)) ds
    combined_s_x = ctx_x.zero_series()
    for s_idx, s in enumerate(s_nodes):
        s_base = 2 * R * s * np.ones_like(U)
        s_lin = {"x": 2 * R * s * theta * np.ones_like(U)}
        s_exp = compose_exp_on_affine(1.0, s_base, s_lin, ctx_x.var_names)
        combined_s_x = combined_s_x + s_exp * s_weights[s_idx]

    log_factor_x = ctx_x.scalar_series(np.ones_like(U))
    log_factor_x = log_factor_x + ctx_x.variable_series("x") * theta
    combined_x = combined_s_x * log_factor_x

    # Build P₁(x+u) and P₁(u) series
    P1_x_u0 = U
    P1_x_lin = {"x": np.ones_like(U)}
    P1_x_series = compose_polynomial_on_affine(P1, P1_x_u0, P1_x_lin, ctx_x.var_names)

    P1_u_series = ctx_x.scalar_series(P1.eval(U))

    # Build Q(Arg_α) and Q(Arg_β) with y=0
    # Arg_α = t + θ·t·x, Arg_β = t + (θ·t-θ)·x
    Q_alpha_u0 = T
    Q_alpha_lin = {"x": theta * T}
    Q_alpha_x = compose_polynomial_on_affine(Q, Q_alpha_u0, Q_alpha_lin, ctx_x.var_names)

    Q_beta_u0 = T
    Q_beta_lin = {"x": theta * T - theta}
    Q_beta_x = compose_polynomial_on_affine(Q, Q_beta_u0, Q_beta_lin, ctx_x.var_names)

    # Build algebraic prefactor: (1/θ + x)
    prefactor_x = ctx_x.scalar_series(np.ones_like(U) / theta)
    prefactor_x = prefactor_x + ctx_x.variable_series("x")

    # Multiply all together
    integrand_x = combined_x * P1_x_series * P1_u_series * Q_alpha_x * Q_beta_x * prefactor_x

    # Extract d/dx coefficient (mask = 1 for "x")
    x_coeff = integrand_x.extract(("x",))

    # Apply poly prefactor (1-u) and numeric prefactor -1
    poly_pref_x = (1 - U)
    x_coeff = -1.0 * poly_pref_x * x_coeff

    I3_combined = np.sum(W * x_coeff)

    # ===== I4 (d/dy, x=0) =====
    # Single variable y
    ctx_y = SeriesContext(var_names=("y",))

    # Build combined factor for y only
    combined_s_y = ctx_y.zero_series()
    for s_idx, s in enumerate(s_nodes):
        s_base = 2 * R * s * np.ones_like(U)
        s_lin = {"y": 2 * R * s * theta * np.ones_like(U)}
        s_exp = compose_exp_on_affine(1.0, s_base, s_lin, ctx_y.var_names)
        combined_s_y = combined_s_y + s_exp * s_weights[s_idx]

    log_factor_y = ctx_y.scalar_series(np.ones_like(U))
    log_factor_y = log_factor_y + ctx_y.variable_series("y") * theta
    combined_y = combined_s_y * log_factor_y

    # Build P₁(u) and P₁(y+u) series
    P1_u_y_series = ctx_y.scalar_series(P1.eval(U))

    P1_y_u0 = U
    P1_y_lin = {"y": np.ones_like(U)}
    P1_y_series = compose_polynomial_on_affine(P1, P1_y_u0, P1_y_lin, ctx_y.var_names)

    # Build Q(Arg_α) and Q(Arg_β) with x=0
    # Arg_α = t + (θ·t-θ)·y, Arg_β = t + θ·t·y
    Q_alpha_y_u0 = T
    Q_alpha_y_lin = {"y": theta * T - theta}
    Q_alpha_y = compose_polynomial_on_affine(Q, Q_alpha_y_u0, Q_alpha_y_lin, ctx_y.var_names)

    Q_beta_y_u0 = T
    Q_beta_y_lin = {"y": theta * T}
    Q_beta_y = compose_polynomial_on_affine(Q, Q_beta_y_u0, Q_beta_y_lin, ctx_y.var_names)

    # Build algebraic prefactor: (1/θ + y)
    prefactor_y = ctx_y.scalar_series(np.ones_like(U) / theta)
    prefactor_y = prefactor_y + ctx_y.variable_series("y")

    # Multiply all together
    integrand_y = combined_y * P1_u_y_series * P1_y_series * Q_alpha_y * Q_beta_y * prefactor_y

    # Extract d/dy coefficient (mask = 1 for "y")
    y_coeff = integrand_y.extract(("y",))

    # Apply poly prefactor (1-u) and numeric prefactor -1
    poly_pref_y = (1 - U)
    y_coeff = -1.0 * poly_pref_y * y_coeff

    I4_combined = np.sum(W * y_coeff)

    S34_combined = I3_combined + I4_combined

    if verbose:
        print(f"S34_tex_combined_11:")
        print(f"  I3_combined: {I3_combined:.6f}")
        print(f"  I4_combined: {I4_combined:.6f}")
        print(f"  S34_combined: {S34_combined:.6f}")

    return TexCombinedS34Result(
        I3_combined=I3_combined,
        I4_combined=I4_combined,
        S34_combined=S34_combined,
        n_quad=n,
        n_quad_s=n_quad_s,
    )


def compute_S34_base_11(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict[str, PolyLike],
    verbose: bool = False,
    apply_mirror: bool = False,  # SPEC LOCK: Must remain False
) -> float:
    """
    Compute S34 (I3 + I4) for (1,1) WITHOUT combined factor.

    Per PRZZ TRUTH_SPEC Section 10: I₃ and I₄ do NOT require mirror.
    So S34 uses the base structure only, no combined mirror factor.

    SPEC LOCK: I3/I4 do NOT have mirror structure (TRUTH_SPEC.md Section 10).
    The apply_mirror parameter exists only to catch accidental misuse.
    Setting apply_mirror=True will RAISE I34MirrorForbiddenError.

    Args:
        theta: θ parameter
        R: R parameter
        n: Number of quadrature points
        polynomials: Dict with "P1", "Q"
        verbose: Print debug info
        apply_mirror: FORBIDDEN - raises if True (spec lock)

    Returns:
        S34 value (scalar)

    Raises:
        I34MirrorForbiddenError: If apply_mirror=True
    """
    _assert_i34_no_mirror(apply_mirror, "compute_S34_base_11")
    from src.quadrature import tensor_grid_2d
    from src.term_dsl import SeriesContext
    from src.composition import compose_polynomial_on_affine, compose_exp_on_affine

    P1 = polynomials["P1"]
    Q = polynomials["Q"]

    U, T, W = tensor_grid_2d(n)

    # ===== I3 (d/dx, y=0) =====
    ctx_x = SeriesContext(var_names=("x",))

    # Build P₁(x+u) and P₁(u) series
    P1_x_u0 = U
    P1_x_lin = {"x": np.ones_like(U)}
    P1_x_series = compose_polynomial_on_affine(P1, P1_x_u0, P1_x_lin, ctx_x.var_names)
    P1_u_series = ctx_x.scalar_series(P1.eval(U))

    # Build Q series (y=0)
    Q_alpha_u0 = T
    Q_alpha_lin = {"x": theta * T}
    Q_alpha_x = compose_polynomial_on_affine(Q, Q_alpha_u0, Q_alpha_lin, ctx_x.var_names)

    Q_beta_u0 = T
    Q_beta_lin = {"x": theta * T - theta}
    Q_beta_x = compose_polynomial_on_affine(Q, Q_beta_u0, Q_beta_lin, ctx_x.var_names)

    # Build exp factors (y=0)
    # exp(R·Arg_α) × exp(R·Arg_β) where Arg_α = t + θtx, Arg_β = t + (θt-θ)x
    exp_u0_x = 2 * R * T
    exp_lin_x = {"x": R * (2 * theta * T - theta)}
    exp_x = compose_exp_on_affine(1.0, exp_u0_x, exp_lin_x, ctx_x.var_names)

    # Build algebraic prefactor: (1/θ + x)
    prefactor_x = ctx_x.scalar_series(np.ones_like(U) / theta)
    prefactor_x = prefactor_x + ctx_x.variable_series("x")

    # Multiply all
    integrand_x = P1_x_series * P1_u_series * Q_alpha_x * Q_beta_x * exp_x * prefactor_x

    # Extract d/dx coefficient
    x_coeff = integrand_x.extract(("x",))

    # Apply poly prefactor (1-u) and numeric prefactor -1
    x_coeff = -1.0 * (1 - U) * x_coeff
    I3 = np.sum(W * x_coeff)

    # ===== I4 (d/dy, x=0) =====
    ctx_y = SeriesContext(var_names=("y",))

    # Build P₁(u) and P₁(y+u) series
    P1_u_y_series = ctx_y.scalar_series(P1.eval(U))
    P1_y_u0 = U
    P1_y_lin = {"y": np.ones_like(U)}
    P1_y_series = compose_polynomial_on_affine(P1, P1_y_u0, P1_y_lin, ctx_y.var_names)

    # Build Q series (x=0)
    Q_alpha_y_u0 = T
    Q_alpha_y_lin = {"y": theta * T - theta}
    Q_alpha_y = compose_polynomial_on_affine(Q, Q_alpha_y_u0, Q_alpha_y_lin, ctx_y.var_names)

    Q_beta_y_u0 = T
    Q_beta_y_lin = {"y": theta * T}
    Q_beta_y = compose_polynomial_on_affine(Q, Q_beta_y_u0, Q_beta_y_lin, ctx_y.var_names)

    # Build exp factors (x=0)
    exp_u0_y = 2 * R * T
    exp_lin_y = {"y": R * (2 * theta * T - theta)}
    exp_y = compose_exp_on_affine(1.0, exp_u0_y, exp_lin_y, ctx_y.var_names)

    # Build algebraic prefactor: (1/θ + y)
    prefactor_y = ctx_y.scalar_series(np.ones_like(U) / theta)
    prefactor_y = prefactor_y + ctx_y.variable_series("y")

    # Multiply all
    integrand_y = P1_u_y_series * P1_y_series * Q_alpha_y * Q_beta_y * exp_y * prefactor_y

    # Extract d/dy coefficient
    y_coeff = integrand_y.extract(("y",))

    # Apply poly prefactor (1-u) and numeric prefactor -1
    y_coeff = -1.0 * (1 - U) * y_coeff
    I4 = np.sum(W * y_coeff)

    S34 = I3 + I4

    if verbose:
        print(f"S34_base_11:")
        print(f"  I3: {I3:.6f}")
        print(f"  I4: {I4:.6f}")
        print(f"  S34: {S34:.6f}")

    return S34


# =============================================================================
# Run 19: TeX-Exact Mirror Core with Q-Shift Inside Combined Structure
# =============================================================================


@dataclass
class TexExactI1Result:
    """Result from I1 TeX-exact computation (Run 19)."""
    I1_tex_exact: float
    scalar_limit_t05: float  # Scalar limit at t=0.5
    n_quad: int


def compute_I1_tex_exact_11(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict[str, PolyLike],
    verbose: bool = False,
) -> TexExactI1Result:
    """
    Compute I1 for (1,1) using TeX-exact combined structure (Run 19).

    This implements the CORRECT structure where Q operators are applied
    INSIDE the combined object with proper Q-shift in the minus branch.

    Structure:
        Plus branch:  Q(arg_α) × Q(arg_β) × exp(R·arg_α) × exp(R·arg_β)
        Minus branch: Q(arg_α+1) × Q(arg_β+1) × exp(-R·arg_α) × exp(-R·arg_β) × exp(2R)

    Key difference from Run 18:
        - Q factors are INSIDE the combined structure, not multiplied externally
        - Q-shift (sigma=1.0) is applied in the minus branch

    Args:
        theta: θ parameter (typically 4/7)
        R: R parameter
        n: Number of quadrature points for (u, t)
        polynomials: Dict with "P1", "Q"
        verbose: Print debug info

    Returns:
        TexExactI1Result with I1_tex_exact value
    """
    from src.quadrature import tensor_grid_2d
    from src.term_dsl import SeriesContext, CombinedI1Integrand
    from src.composition import compose_polynomial_on_affine
    from src.q_operator import lift_poly_by_shift

    # Get polynomials
    P1 = polynomials["P1"]
    Q = polynomials["Q"]

    # Build Q_shifted = Q(x+1)
    Q_shifted = lift_poly_by_shift(Q, shift=1.0)

    # Build quadrature grid
    U, T, W = tensor_grid_2d(n)

    # Create series context with x, y variables
    ctx = SeriesContext(var_names=("x", "y"))

    # Build the CombinedI1Integrand with Q-shift inside
    combined_integrand = CombinedI1Integrand(
        R=R,
        theta=theta,
        Q=Q,
        Q_shifted=Q_shifted,
    )
    combined_series = combined_integrand.evaluate(U, T, ctx)

    # Build P₁(x+u) series
    P1_x_u0 = U
    P1_x_lin = {"x": np.ones_like(U)}
    P1_x_series = compose_polynomial_on_affine(P1, P1_x_u0, P1_x_lin, ctx.var_names)

    # Build P₁(y+u) series
    P1_y_u0 = U
    P1_y_lin = {"y": np.ones_like(U)}
    P1_y_series = compose_polynomial_on_affine(P1, P1_y_u0, P1_y_lin, ctx.var_names)

    # Build algebraic prefactor: (1/θ + x + y)
    prefactor_series = ctx.scalar_series(np.ones_like(U) / theta)
    prefactor_series = prefactor_series + ctx.variable_series("x")
    prefactor_series = prefactor_series + ctx.variable_series("y")

    # Multiply all series together
    # Structure: combined_integrand (with Q inside) × P₁ × P₁ × prefactor
    # Note: Q factors are ALREADY inside combined_series, not multiplied separately
    integrand = combined_series
    integrand = integrand * P1_x_series
    integrand = integrand * P1_y_series
    integrand = integrand * prefactor_series

    # Extract d²/dxdy coefficient
    xy_coeff = integrand.extract(("x", "y"))

    # Apply poly prefactor: (1-u)²
    poly_prefactor = (1 - U) ** 2
    xy_coeff = xy_coeff * poly_prefactor

    # Integrate over (u, t)
    I1_tex_exact = np.sum(W * xy_coeff)

    if verbose:
        scalar_val = combined_integrand.scalar_limit(t_val=0.5)
        print(f"I1_tex_exact_11:")
        print(f"  CombinedI1Integrand scalar limit (t=0.5): {scalar_val:.6f}")
        print(f"  I1_tex_exact: {I1_tex_exact:.6f}")

    return TexExactI1Result(
        I1_tex_exact=I1_tex_exact,
        scalar_limit_t05=combined_integrand.scalar_limit(t_val=0.5),
        n_quad=n,
    )


# =============================================================================
# Run 20: TeX Combined Mirror Core (difference quotient → log×integral)
# =============================================================================


@dataclass
class TexCombinedCoreResult:
    """Result from I1 TeX-combined-core computation (Run 20).

    Key differences from Run 18/19:
    - Uses TexCombinedMirrorCore with outer exp(-Rθ(x+y)) factor
    - Q operators applied AFTER combined structure (per PRZZ TeX)
    """
    I1_combined_core: float
    scalar_limit: float
    xy_coeff_before_integration: float  # For diagnostics
    n_quad: int
    n_quad_s: int


def compute_I1_tex_combined_core_11(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict[str, PolyLike],
    n_quad_s: int = 20,
    verbose: bool = False,
) -> TexCombinedCoreResult:
    """
    Compute I1 for (1,1) using Run 20 TeX combined mirror core.

    Implements the PRZZ difference quotient → log×integral identity (TeX 1502-1511):
        (N^{αx+βy} - T^{-α-β}N^{-βx-αy}) / (α+β)
        = N^{αx+βy} × log(N^{x+y}T) × ∫₀¹ (N^{x+y}T)^{-s(α+β)} ds

    At α = β = -R/L:
        = exp(-Rθ(x+y)) × (1 + θ(x+y)) × ∫₀¹ exp(2sR(1 + θ(x+y))) ds

    Key innovation (Stage 20C): Q operators are applied AFTER the combined
    structure is formed, matching PRZZ's TeX derivation order.

    Structure:
        combined_core × Q(arg_α) × Q(arg_β) × P₁(x+u) × P₁(y+u) × prefactor

    Args:
        theta: θ parameter (typically 4/7)
        R: R parameter
        n: Number of quadrature points for (u, t)
        polynomials: Dict with "P1", "Q"
        n_quad_s: Quadrature points for s-integral
        verbose: Print debug info

    Returns:
        TexCombinedCoreResult with I1_combined_core value
    """
    from src.quadrature import tensor_grid_2d
    from src.term_dsl import SeriesContext, TexCombinedMirrorCore
    from src.composition import compose_polynomial_on_affine

    # Get polynomials
    P1 = polynomials["P1"]
    Q = polynomials["Q"]

    # Build quadrature grid
    U, T_grid, W = tensor_grid_2d(n)

    # Create series context with x, y variables
    ctx = SeriesContext(var_names=("x", "y"))

    # --- Step 1: Build the TexCombinedMirrorCore (NO Q yet!) ---
    # This implements the TeX combined structure with outer exp factor
    combined_core = TexCombinedMirrorCore(R=R, theta=theta, n_quad_s=n_quad_s)
    core_series = combined_core.evaluate(U, T_grid, ctx)

    # --- Step 2: Build Q factors SEPARATELY (applied after combined core) ---
    # Q(Arg_α) where Arg_α = t + θ·t·x + (θ·t-θ)·y
    Q_alpha_u0 = T_grid
    Q_alpha_lin = {
        "x": theta * T_grid,
        "y": theta * T_grid - theta,
    }
    Q_alpha_series = compose_polynomial_on_affine(Q, Q_alpha_u0, Q_alpha_lin, ctx.var_names)

    # Q(Arg_β) where Arg_β = t + (θ·t-θ)·x + θ·t·y
    Q_beta_u0 = T_grid
    Q_beta_lin = {
        "x": theta * T_grid - theta,
        "y": theta * T_grid,
    }
    Q_beta_series = compose_polynomial_on_affine(Q, Q_beta_u0, Q_beta_lin, ctx.var_names)

    # --- Step 3: Multiply Q AFTER combined core (Stage 20C key insight) ---
    integrand = core_series * Q_alpha_series * Q_beta_series

    # --- Step 4: Build P₁ factors ---
    # P₁(x+u) series
    P1_x_u0 = U
    P1_x_lin = {"x": np.ones_like(U)}
    P1_x_series = compose_polynomial_on_affine(P1, P1_x_u0, P1_x_lin, ctx.var_names)

    # P₁(y+u) series
    P1_y_u0 = U
    P1_y_lin = {"y": np.ones_like(U)}
    P1_y_series = compose_polynomial_on_affine(P1, P1_y_u0, P1_y_lin, ctx.var_names)

    integrand = integrand * P1_x_series * P1_y_series

    # --- Step 5: Build algebraic prefactor: (1/θ + x + y) ---
    prefactor_series = ctx.scalar_series(np.ones_like(U) / theta)
    prefactor_series = prefactor_series + ctx.variable_series("x")
    prefactor_series = prefactor_series + ctx.variable_series("y")
    integrand = integrand * prefactor_series

    # --- Step 6: Extract d²/dxdy coefficient ---
    xy_coeff = integrand.extract(("x", "y"))

    # Diagnostic: save xy_coeff before poly prefactor
    xy_coeff_sample = float(xy_coeff[0, 0]) if xy_coeff.size > 0 else 0.0

    # --- Step 7: Apply poly prefactor: (1-u)² ---
    poly_prefactor = (1 - U) ** 2
    xy_coeff = xy_coeff * poly_prefactor

    # --- Step 8: Integrate over (u, t) ---
    I1_combined_core = np.sum(W * xy_coeff)

    if verbose:
        print(f"I1_tex_combined_core_11 (Run 20):")
        print(f"  TexCombinedMirrorCore scalar limit: {combined_core.scalar_limit():.6f}")
        print(f"  xy_coeff sample (before poly prefactor): {xy_coeff_sample:.6f}")
        print(f"  I1_combined_core: {I1_combined_core:.6f}")

    return TexCombinedCoreResult(
        I1_combined_core=I1_combined_core,
        scalar_limit=combined_core.scalar_limit(),
        xy_coeff_before_integration=xy_coeff_sample,
        n_quad=n,
        n_quad_s=n_quad_s,
    )


# =============================================================================
# Phase 10.3: Derived Mirror Computation
# =============================================================================
#
# This function computes c using the derived mirror operator from Phase 10.2,
# which uses the swap/sign conjugation from (-β,-α) instead of the empirical
# m₁ scalar multiplier.
#
# Assembly formula:
#   c_derived = S12_direct(+R) + S12_mirror_exact(+R) + S34(+R)
#
# where S12_mirror_exact uses the operator transform with swapped eigenvalues,
# NOT m₁ × S12(-R).
# =============================================================================


@dataclass
class DerivedMirrorCResult:
    """Result from derived mirror c computation."""
    c: float
    """Total c value with derived mirror."""

    S12_direct: float
    """S12 computed at +R (direct)."""

    S12_mirror_operator: float
    """S12 mirror computed via operator approach."""

    S34: float
    """S34 at +R (no mirror)."""

    m1_eff: float
    """Effective m₁ = S12_mirror / S12_basis (diagnostic only)."""

    S12_basis: float
    """S12 at -R (DSL minus basis, for m₁_eff calculation)."""

    kappa: float
    """κ = 1 - log(c)/R."""

    R: float
    theta: float
    n: int


def compute_c_derived_mirror(
    theta: float,
    R: float,
    n: int,
    polynomials: Dict[str, PolyLike],
    K: int = 3,
    verbose: bool = False,
) -> DerivedMirrorCResult:
    """
    Compute c using derived mirror operator (Phase 10.3).

    This is the goal of Phase 10: replace empirical m₁ with derived operator.

    Assembly formula:
        c = S12_direct(+R) + S12_mirror_operator(+R) + S34(+R)

    The mirror contribution is computed from the operator transform with
    swapped eigenvalues, NOT from m₁ × S12(-R).

    Args:
        theta: PRZZ θ parameter (4/7)
        R: PRZZ R parameter
        n: Quadrature points
        polynomials: Dict with P1, P2, P3, Q
        K: Number of mollifier pieces
        verbose: Print diagnostics

    Returns:
        DerivedMirrorCResult with c value and breakdown
    """
    from src.mirror_operator_exact import compute_S12_mirror_operator_exact
    from src.mirror_exact import compute_S12_minus_basis

    # Import the harness for S12_direct computation
    from src.mirror_transform_harness import MirrorTransformHarness

    harness = MirrorTransformHarness(theta, R, n, polynomials, K)

    # Compute S12 direct (+R)
    S12_direct = 0.0
    for pair_key in harness.PAIRS:
        ell1 = int(pair_key[0])
        ell2 = int(pair_key[1])
        norm = harness.FACTORIAL_NORM[pair_key] * harness.SYMMETRY[pair_key]
        direct = harness._compute_S12_direct_pair(ell1, ell2)
        S12_direct += norm * direct

    # Compute S12 mirror via operator approach
    S12_mirror_operator = compute_S12_mirror_operator_exact(
        theta=theta, R=R, n=n, polynomials=polynomials, K=K, verbose=verbose
    )

    # Compute S12 minus basis (for m₁_eff diagnostic)
    S12_basis = compute_S12_minus_basis(
        theta=theta, R=R, n=n, polynomials=polynomials, K=K
    )

    # Compute S34 (no mirror)
    # For now, use the harness method (which returns 0 as placeholder)
    S34 = 0.0
    for pair_key in harness.PAIRS:
        ell1 = int(pair_key[0])
        ell2 = int(pair_key[1])
        norm = harness.FACTORIAL_NORM[pair_key] * harness.SYMMETRY[pair_key]
        s34 = harness._compute_S34_pair(ell1, ell2)
        S34 += norm * s34

    # Assembly
    c = S12_direct + S12_mirror_operator + S34

    # Compute κ
    if c > 0:
        kappa = 1.0 - np.log(c) / R
    else:
        kappa = float('nan')

    # Effective m₁ (diagnostic only)
    if abs(S12_basis) > 1e-15:
        m1_eff = S12_mirror_operator / S12_basis
    else:
        m1_eff = float('inf')

    if verbose:
        print(f"\n=== Derived Mirror c Computation ===")
        print(f"R = {R}, θ = {theta:.6f}, n = {n}")
        print(f"S12_direct:         {S12_direct:.6f}")
        print(f"S12_mirror_operator: {S12_mirror_operator:.6f}")
        print(f"S34:                {S34:.6f}")
        print(f"c = {c:.6f}")
        print(f"κ = {kappa:.6f}")
        print(f"")
        print(f"For comparison:")
        print(f"S12_basis (-R):     {S12_basis:.6f}")
        print(f"m₁_eff (diagnostic): {m1_eff:.4f}")
        print(f"m₁_empirical:       {np.exp(R) + 5:.4f}")

    return DerivedMirrorCResult(
        c=c,
        S12_direct=S12_direct,
        S12_mirror_operator=S12_mirror_operator,
        S34=S34,
        m1_eff=m1_eff,
        S12_basis=S12_basis,
        kappa=kappa,
        R=R,
        theta=theta,
        n=n,
    )

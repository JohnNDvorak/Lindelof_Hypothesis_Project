"""
src/evaluator/result_types.py
Core dataclasses and error types for PRZZ evaluation.

Extracted from src/evaluate.py as part of Phase 20.4 refactoring.
Created: 2025-12-24

This module contains:
1. Result dataclasses (TermResult, EvaluationResult, DiagnosticResult)
2. Error classes for spec violation detection
3. Helper functions for S34 triangle pairs

BACKWARDS COMPATIBILITY:
========================
These types are re-exported from src.evaluate for backwards compatibility.
New code should import from src.evaluator.result_types directly.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Any


# =============================================================================
# RESULT DATACLASSES
# =============================================================================


@dataclass
class TermResult:
    """Result of evaluating a single term.

    Attributes:
        name: Term identifier (e.g., "I1_11", "I2_12")
        value: Computed numeric value
        extracted_coeff_sample: Coefficient at grid center (for debugging)
        series_term_count: Number of series terms used
    """
    name: str
    value: float
    # For debugging: intermediate values
    extracted_coeff_sample: Optional[float] = None  # coeff at grid center
    series_term_count: Optional[int] = None  # number of series terms


@dataclass
class EvaluationResult:
    """Result of evaluating multiple terms.

    Attributes:
        total: Sum of all term values
        per_term: Dictionary mapping term names to values
        n: Quadrature order used
        term_results: Optional list of detailed TermResult objects
    """
    total: float
    per_term: Dict[str, float]
    n: int
    # Additional metadata
    term_results: Optional[List[TermResult]] = None


@dataclass
class DiagnosticResult:
    """Complete diagnostic breakdown for debugging discrepancies.

    This result type provides comprehensive information for
    tracking down differences between computed and target values.

    Attributes:
        c_computed: Computed c value
        c_target: PRZZ target c value
        delta_c: Difference (computed - target)
        kappa_computed: Computed kappa value
        kappa_target: PRZZ target kappa value
        delta_kappa: Difference (computed - target)
        pair_raw: Per-pair raw values before normalization
        pair_normalized: Per-pair values after normalization
        per_term: Per-term breakdown (I1, I2, I3, I4 for each pair)
        norm_factors: Normalization factors applied
        flags: Configuration flags used
        n: Quadrature order
    """
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

    # Per-term breakdown (I1, I2, I3, I4 for each pair)
    per_term: Dict[str, float]

    # Normalization factors applied
    norm_factors: Dict[str, float]

    # Effective flags
    flags: Dict[str, Any]

    # Quadrature info
    n: int


# =============================================================================
# ERROR CLASSES - SPEC LOCKS
# =============================================================================


class S34OrderedPairsError(ValueError):
    """Raised when S34 is computed with 9 ordered pairs instead of triangle x2.

    SPEC LOCK: S34 uses TRIANGLE x 2 convention, NOT 9 ordered pairs.

    PRZZ sums over pairs (l1, l2) with l1 <= l2, using a x2 symmetry factor
    for off-diagonal pairs. This is the "triangle x 2" convention.

    Using 9 ordered pairs instead of triangle x 2 causes a +11% OVERSHOOT.
    This is a hard invariant that must be preserved.

    Reference: TRUTH_SPEC.md Section 13 (Ordered vs Triangle Assembly)
    """
    pass


class I34MirrorForbiddenError(ValueError):
    """Raised when someone tries to apply mirror to I3/I4.

    SPEC LOCK: I3/I4 Mirror is FORBIDDEN.

    Per TRUTH_SPEC.md Section 10 (lines 370-388):
      - I1(a,b) + T^{-a-b}I1(-b,-a)  <- HAS MIRROR
      - I2(a,b) + T^{-a-b}I2(-b,-a)  <- HAS MIRROR
      - I3(a,b) and I4(a,b)          <- NO MIRROR

    This prevents accidental reintroduction of incorrect mirror handling.
    """
    pass


# =============================================================================
# S34 TRIANGLE CONVENTION HELPERS
# =============================================================================


def get_s34_triangle_pairs() -> list:
    """
    Return the canonical S34 triangle pairs with symmetry factors.

    SPEC LOCK: S34 uses TRIANGLE x 2 convention.
    - Diagonal pairs (1,1), (2,2), (3,3): symmetry factor = 1
    - Off-diagonal pairs (1,2), (1,3), (2,3): symmetry factor = 2

    Returns:
        List of (pair_key, symmetry_factor) tuples
    """
    return [
        ("11", 1),   # diagonal
        ("22", 1),   # diagonal
        ("33", 1),   # diagonal
        ("12", 2),   # off-diagonal, x2
        ("13", 2),   # off-diagonal, x2
        ("23", 2),   # off-diagonal, x2
    ]


def get_s34_factorial_normalization() -> dict:
    """
    Return factorial normalization factors for S34 pairs.

    Factor for pair (l1, l2) is 1/(l1-1)!/(l2-1)!
    """
    return {
        "11": 1.0,       # 1/0!/0! = 1
        "22": 0.25,      # 1/1!/1! = 1/4
        "33": 1/36,      # 1/2!/2! = 1/36
        "12": 0.5,       # 1/0!/1! = 1/2
        "13": 1/6,       # 1/0!/2! = 1/6
        "23": 1/12,      # 1/1!/2! = 1/12
    }


def assert_s34_triangle_convention(pair_keys: list, caller: str = "") -> None:
    """
    Guard function to enforce S34 triangle x 2 convention.

    RAISES S34OrderedPairsError if pair_keys looks like 9 ordered pairs
    instead of 6 triangle pairs.

    Args:
        pair_keys: List of pair keys being used
        caller: Calling function name for error message
    """
    TRIANGLE_KEYS = {"11", "22", "33", "12", "13", "23"}
    FORBIDDEN_ORDERED_ONLY = {"21", "31", "32"}

    pair_set = set(pair_keys)

    # Check if any forbidden ordered-only keys are present
    forbidden_found = pair_set & FORBIDDEN_ORDERED_ONLY
    if forbidden_found:
        raise S34OrderedPairsError(
            f"{caller}: S34 MUST use triangle x 2 convention (6 pairs with x2 off-diagonal), "
            f"NOT 9 ordered pairs. Found forbidden keys: {forbidden_found}. "
            f"Using 9 ordered pairs causes +11% overshoot. "
            f"Reference: TRUTH_SPEC.md Section 13"
        )


def assert_i34_no_mirror(apply_mirror: bool, caller: str = "") -> None:
    """
    Guard function to prevent mirror application to I3/I4.

    RAISES I34MirrorForbiddenError if apply_mirror is True.

    Args:
        apply_mirror: Whether mirror is being applied
        caller: Calling function name for error message
    """
    if apply_mirror:
        raise I34MirrorForbiddenError(
            f"{caller}: I3/I4 do NOT use mirror terms. "
            f"Per TRUTH_SPEC.md Section 10: only I1 and I2 have mirror structure. "
            f"I3(a,b) and I4(a,b) have NO mirror counterpart."
        )

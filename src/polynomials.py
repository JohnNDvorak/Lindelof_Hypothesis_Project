"""
Polynomial representation and evaluation for PRZZ kappa computation.

This module provides polynomial classes with built-in constraint enforcement:
- P1: P1(0)=0, P1(1)=1 via parameterization P1(x) = x + x(1-x)*P_tilde(x)
- P_ell (ell>=2): P_ell(0)=0 via parameterization P_ell(x) = x*P_tilde(x)
- Q: Q(0)=1 via enforce_Q0 mode or paper-literal mode

All derivatives computed analytically via cached monomial conversion.
NO finite differences are used anywhere.
"""

from __future__ import annotations
import json
import re
import numpy as np
from dataclasses import dataclass
from math import comb
from pathlib import Path
from typing import Dict, List, Tuple, Union


def falling_factorial(n: int, k: int) -> int:
    """
    Compute falling factorial: n(n-1)...(n-k+1) = n!/(n-k)!

    Args:
        n: Non-negative integer
        k: Non-negative integer

    Returns:
        n!/(n-k)! if k <= n
        1 if k == 0
        0 if k > n (safe guardrail)

    Raises:
        ValueError: if k < 0
    """
    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0:
        return 1
    if k > n:
        return 0
    result = 1
    for i in range(k):
        result *= (n - i)
    return result


@dataclass
class Polynomial:
    """
    Polynomial in monomial basis: P(x) = sum_k c_k * x^k

    This is the workhorse class. All constrained polynomials convert to this
    for evaluation and derivatives, avoiding manual product/chain rule.

    Attributes:
        coeffs: Coefficient array where coeffs[k] is the coefficient of x^k
    """
    coeffs: np.ndarray

    def __post_init__(self):
        """Ensure coeffs is a numpy array of floats."""
        self.coeffs = np.asarray(self.coeffs, dtype=float)

    @property
    def degree(self) -> int:
        """Return the degree of the polynomial."""
        if len(self.coeffs) == 0:
            return -1  # Convention for zero polynomial
        # Find highest non-zero coefficient
        for i in range(len(self.coeffs) - 1, -1, -1):
            if abs(self.coeffs[i]) > 1e-15:
                return i
        return 0  # Constant polynomial (including zero)

    def eval(self, x: np.ndarray) -> np.ndarray:
        """
        Evaluate polynomial at points x using Horner's method.

        Args:
            x: Array of evaluation points (any shape)

        Returns:
            Array of same shape as x with polynomial values
        """
        x = np.asarray(x, dtype=float)
        if len(self.coeffs) == 0:
            return np.zeros_like(x)

        # Horner's method: P(x) = c0 + x*(c1 + x*(c2 + ...))
        result = np.full_like(x, self.coeffs[-1], dtype=float)
        for c in reversed(self.coeffs[:-1]):
            result = result * x + c
        return result

    def eval_deriv(self, x: np.ndarray, k: int) -> np.ndarray:
        """
        Evaluate k-th derivative at points x analytically.

        For P(x) = sum_j c_j * x^j:
        P^(k)(x) = sum_{j>=k} c_j * falling_factorial(j,k) * x^{j-k}

        Args:
            x: Array of evaluation points (any shape)
            k: Derivative order (non-negative integer)

        Returns:
            Array of same shape as x with derivative values
        """
        x = np.asarray(x, dtype=float)

        if k < 0:
            raise ValueError("Derivative order must be non-negative")
        if k == 0:
            return self.eval(x)
        if k > len(self.coeffs) - 1:
            return np.zeros_like(x)

        # Build derivative coefficients
        deriv_coeffs = np.zeros(len(self.coeffs) - k)
        for j in range(k, len(self.coeffs)):
            deriv_coeffs[j - k] = self.coeffs[j] * falling_factorial(j, k)

        # Evaluate using Horner
        return Polynomial(deriv_coeffs).eval(x)


class P1Polynomial:
    """
    P1(x) = x + x(1-x)*P_tilde(x) where P_tilde is in (1-x) powers.

    Constraints P1(0)=0 and P1(1)=1 are automatically enforced.

    The tilde polynomial is:
        P_tilde(y) = sum_m a_m * y^m  where y = (1-x)

    Attributes:
        tilde_coeffs: Coefficients of P_tilde in (1-x) powers
        _monomial: Cached monomial expansion
    """

    def __init__(self, tilde_coeffs: Union[List[float], np.ndarray]):
        """
        Create P1 from P_tilde coefficients.

        Args:
            tilde_coeffs: Coefficients [a0, a1, a2, ...] where
                         P_tilde(1-x) = a0 + a1*(1-x) + a2*(1-x)^2 + ...
        """
        self.tilde_coeffs = np.asarray(tilde_coeffs, dtype=float)
        self._monomial = self._build_monomial()

    def _build_monomial(self) -> Polynomial:
        """
        Expand P1(x) = x + x(1-x)*P_tilde(1-x) to monomial form.

        P_tilde(1-x) = sum_m a_m * (1-x)^m
        x(1-x)*P_tilde(1-x) = sum_m a_m * x*(1-x)^{m+1}

        For (1-x)^m = sum_k C(m,k)*(-1)^k * x^k
        We have x*(1-x)^{m+1} = sum_k C(m+1,k)*(-1)^k * x^{k+1}

        So P1(x) = x + sum_m a_m * [sum_k C(m+1,k)*(-1)^k * x^{k+1}]
        """
        if len(self.tilde_coeffs) == 0:
            # P1(x) = x
            return Polynomial(np.array([0.0, 1.0]))

        # Determine maximum degree
        # x*(1-x)^{m+1} has max degree m+2 for m = len(tilde_coeffs)-1
        max_m = len(self.tilde_coeffs) - 1
        max_degree = max_m + 2

        # Initialize coefficients (degree max_degree means max_degree+1 coeffs)
        mono_coeffs = np.zeros(max_degree + 1)

        # Add x term
        mono_coeffs[1] = 1.0

        # Add x(1-x)*P_tilde(1-x) terms
        for m, a_m in enumerate(self.tilde_coeffs):
            if abs(a_m) < 1e-50:
                continue
            # Expand x*(1-x)^{m+1}
            # (1-x)^{m+1} = sum_k C(m+1,k)*(-1)^k * x^k for k=0..m+1
            # x*(1-x)^{m+1} = sum_k C(m+1,k)*(-1)^k * x^{k+1} for k=0..m+1
            for k in range(m + 2):  # k goes from 0 to m+1
                coeff = comb(m + 1, k) * ((-1) ** k)
                mono_coeffs[k + 1] += a_m * coeff

        return Polynomial(mono_coeffs)

    def eval(self, x: np.ndarray) -> np.ndarray:
        """Evaluate P1(x) via cached monomial form."""
        return self._monomial.eval(x)

    def eval_deriv(self, x: np.ndarray, k: int) -> np.ndarray:
        """Evaluate k-th derivative via cached monomial form."""
        return self._monomial.eval_deriv(x, k)

    def to_monomial(self) -> Polynomial:
        """Return cached monomial representation."""
        return self._monomial


class PellPolynomial:
    """
    P_ell(x) = x*P_tilde(x) for ell >= 2.

    Constraint P_ell(0)=0 is automatically enforced by the x factor.

    Attributes:
        tilde_coeffs: Coefficients of P_tilde in monomial basis
        _monomial: Cached monomial expansion
    """

    def __init__(self, tilde_coeffs: Union[List[float], np.ndarray]):
        """
        Create P_ell from P_tilde coefficients.

        Args:
            tilde_coeffs: Coefficients [c0, c1, c2, ...] where
                         P_tilde(x) = c0 + c1*x + c2*x^2 + ...
        """
        self.tilde_coeffs = np.asarray(tilde_coeffs, dtype=float)
        self._monomial = self._build_monomial()

    def _build_monomial(self) -> Polynomial:
        """
        Expand P_ell(x) = x*P_tilde(x) to monomial form.

        x*(c0 + c1*x + c2*x^2 + ...) = c0*x + c1*x^2 + c2*x^3 + ...

        Just prepend a zero coefficient.
        """
        if len(self.tilde_coeffs) == 0:
            return Polynomial(np.array([0.0]))

        mono_coeffs = np.concatenate([[0.0], self.tilde_coeffs])
        return Polynomial(mono_coeffs)

    def eval(self, x: np.ndarray) -> np.ndarray:
        """Evaluate P_ell(x) via cached monomial form."""
        return self._monomial.eval(x)

    def eval_deriv(self, x: np.ndarray, k: int) -> np.ndarray:
        """Evaluate k-th derivative via cached monomial form."""
        return self._monomial.eval_deriv(x, k)

    def to_monomial(self) -> Polynomial:
        """Return cached monomial representation."""
        return self._monomial


class QPolynomial:
    """
    Q(x) = sum_k c_k * (1-2x)^k in PRZZ basis.

    Constraint: Q(0)=1 (since (1-2*0)^k = 1 for all k, Q(0) = sum_k c_k)

    Two modes:
    - enforce_Q0=True: c0 computed as 1 - sum_{k>0} c_k (for optimization)
    - enforce_Q0=False: c0 stored exactly (for paper-literal reproduction)

    Attributes:
        basis_coeffs: Dict mapping power k to coefficient c_k
        enforce_Q0: Whether Q(0)=1 is enforced by construction
        _monomial: Cached monomial expansion
    """

    def __init__(self, basis_coeffs: Dict[int, float], enforce_Q0: bool = True):
        """
        Create Q from (1-2x)^k basis coefficients.

        Args:
            basis_coeffs: {k: c_k} for (1-2x)^k terms
            enforce_Q0: If True, recompute c0 so Q(0)=1 exactly.
                       If False, use provided c0 value (or 0 if not provided).
        """
        self.enforce_Q0 = enforce_Q0

        # Make a copy of the coefficients
        self.basis_coeffs = dict(basis_coeffs)

        if enforce_Q0:
            # Compute c0 = 1 - sum_{k>0} c_k
            sum_nonzero = sum(
                c for k, c in self.basis_coeffs.items() if k > 0
            )
            self.basis_coeffs[0] = 1.0 - sum_nonzero
        elif 0 not in self.basis_coeffs:
            self.basis_coeffs[0] = 0.0

        self._monomial = self._build_monomial()

    def _build_monomial(self) -> Polynomial:
        """
        Expand Q(x) = sum_k c_k * (1-2x)^k to monomial form.

        (1-2x)^k = sum_j C(k,j)*(-2x)^j = sum_j C(k,j)*(-2)^j * x^j
        """
        if not self.basis_coeffs:
            return Polynomial(np.array([0.0]))

        # Determine maximum degree
        max_k = max(self.basis_coeffs.keys())

        # Initialize monomial coefficients
        mono_coeffs = np.zeros(max_k + 1)

        # Expand each term
        for k, c_k in self.basis_coeffs.items():
            if abs(c_k) < 1e-50:
                continue
            # (1-2x)^k = sum_j C(k,j)*(-2)^j * x^j for j=0..k
            for j in range(k + 1):
                coeff = comb(k, j) * ((-2) ** j)
                mono_coeffs[j] += c_k * coeff

        return Polynomial(mono_coeffs)

    def eval(self, x: np.ndarray) -> np.ndarray:
        """Evaluate Q(x) via cached monomial form."""
        return self._monomial.eval(x)

    def eval_deriv(self, x: np.ndarray, k: int) -> np.ndarray:
        """Evaluate k-th derivative via cached monomial form."""
        return self._monomial.eval_deriv(x, k)

    def to_monomial(self) -> Polynomial:
        """Return cached monomial representation."""
        return self._monomial

    def Q_at_zero(self) -> float:
        """Return Q(0) = sum_k c_k."""
        return sum(self.basis_coeffs.values())


# =============================================================================
# Factory Functions
# =============================================================================

def make_P1_from_tilde(tilde_coeffs: List[float]) -> P1Polynomial:
    """
    Create P1 from P_tilde coefficients in (1-x) basis.

    Args:
        tilde_coeffs: [a0, a1, a2, ...] where P_tilde(1-x) = sum_m a_m*(1-x)^m

    Returns:
        P1Polynomial with P1(x) = x + x(1-x)*P_tilde(1-x)
    """
    return P1Polynomial(tilde_coeffs)


def make_Pell_from_tilde(tilde_coeffs: List[float]) -> PellPolynomial:
    """
    Create P_ell (ell>=2) from P_tilde monomial coefficients.

    Args:
        tilde_coeffs: [c0, c1, c2, ...] where P_tilde(x) = sum_k c_k*x^k

    Returns:
        PellPolynomial with P(x) = x*P_tilde(x), enforcing P(0)=0
    """
    return PellPolynomial(tilde_coeffs)


def make_Q_from_basis(
    basis_coeffs: Dict[int, float],
    enforce_Q0: bool = True
) -> QPolynomial:
    """
    Create Q from (1-2x)^k basis coefficients.

    Args:
        basis_coeffs: {k: c_k} for (1-2x)^k terms
        enforce_Q0: If True, recompute c0 so Q(0)=1 exactly.
                   If False, use provided c0 value.

    Returns:
        QPolynomial in PRZZ basis
    """
    return QPolynomial(basis_coeffs, enforce_Q0=enforce_Q0)


def _get_p1_tilde_coeffs(p1_data: dict) -> List[float]:
    """Extract P1 tilde coefficients from JSON data (schema-flexible)."""
    if "tilde_coeffs" in p1_data:
        return p1_data["tilde_coeffs"]
    if "P_tilde_coeffs" in p1_data:
        return p1_data["P_tilde_coeffs"]
    raise KeyError("P1 missing both 'tilde_coeffs' and 'P_tilde_coeffs'")


def _get_pell_tilde_coeffs(pell_data: dict) -> List[float]:
    """Extract P_ell tilde coefficients from JSON data (schema-flexible)."""
    if "tilde_coeffs" in pell_data:
        return pell_data["tilde_coeffs"]
    if "coeffs" in pell_data:
        coeffs = pell_data["coeffs"]
        # Validate that coeffs[0] is zero (constraint)
        if abs(coeffs[0]) > 1e-14:
            raise ValueError(
                f"coeffs[0] must be 0 for constrained form, got {coeffs[0]}"
            )
        return coeffs[1:]
    raise KeyError("P_ell missing both 'tilde_coeffs' and 'coeffs'")


def _get_q_basis_coeffs(q_data: dict) -> Dict[int, float]:
    """Extract Q basis coefficients from JSON data (schema-flexible)."""
    # Prefer new format
    if "coeffs_in_basis_terms" in q_data:
        terms = q_data["coeffs_in_basis_terms"]
        return {term["k"]: term["c"] for term in terms}

    # Fall back to old format
    if "coeffs_in_basis" in q_data:
        old_format = q_data["coeffs_in_basis"]
        result = {}
        for key, val in old_format.items():
            if key == "constant":
                result[0] = val
            else:
                # Parse "power_N" -> N
                match = re.match(r"power_(\d+)", key)
                if match:
                    result[int(match.group(1))] = val
                else:
                    raise ValueError(f"Unknown Q basis key: {key}")
        return result

    raise KeyError("Q missing both 'coeffs_in_basis_terms' and 'coeffs_in_basis'")


def load_przz_polynomials(
    enforce_Q0: bool = False,
    json_path: Union[str, Path, None] = None
) -> Tuple[P1Polynomial, PellPolynomial, PellPolynomial, QPolynomial]:
    """
    Load PRZZ polynomials from przz_parameters.json.

    Schema-flexible: accepts both old and new JSON formats.

    Args:
        enforce_Q0: If True, recompute Q's c0 so Q(0)=1 exactly.
                   If False (default), use printed c0 value.
        json_path: Optional path to JSON file. Defaults to data/przz_parameters.json.

    Returns:
        Tuple of (P1, P2, P3, Q) polynomials
    """
    if json_path is None:
        # Default path relative to this module
        json_path = (
            Path(__file__).parent.parent / "data" / "przz_parameters.json"
        )
    else:
        json_path = Path(json_path)

    with open(json_path, "r") as f:
        data = json.load(f)

    polys = data["polynomials"]

    # Load P1 (schema-flexible)
    p1_tilde = _get_p1_tilde_coeffs(polys["P1"])
    P1 = P1Polynomial(p1_tilde)

    # Load P2 (schema-flexible)
    p2_tilde = _get_pell_tilde_coeffs(polys["P2"])
    P2 = PellPolynomial(p2_tilde)

    # Load P3 (schema-flexible)
    p3_tilde = _get_pell_tilde_coeffs(polys["P3"])
    P3 = PellPolynomial(p3_tilde)

    # Load Q (schema-flexible)
    q_coeffs = _get_q_basis_coeffs(polys["Q"])
    Q = QPolynomial(q_coeffs, enforce_Q0=enforce_Q0)

    return P1, P2, P3, Q


def load_przz_polynomials_kappa_star(
    enforce_Q0: bool = False
) -> Tuple[P1Polynomial, PellPolynomial, PellPolynomial, QPolynomial]:
    """
    Load PRZZ polynomials for the kappa* (simple zeros) benchmark.

    These are DIFFERENT polynomials than the main kappa benchmark!
    - R = 1.1167 (vs 1.3036 for main)
    - Linear Q (only constant and (1-2x)^1 terms)
    - Different P1, P2, P3 coefficients

    From PRZZ TeX lines 2587-2598.

    Args:
        enforce_Q0: If True, recompute Q's c0 so Q(0)=1 exactly.
                   If False (default), use printed c0 value.

    Returns:
        Tuple of (P1, P2, P3, Q) polynomials for kappa* benchmark
    """
    json_path = (
        Path(__file__).parent.parent / "data" / "przz_parameters_kappa_star.json"
    )
    return load_przz_polynomials(enforce_Q0=enforce_Q0, json_path=json_path)


# =============================================================================
# PRZZ Constraint Verification
# =============================================================================

@dataclass
class ConstraintResult:
    """Result of a single constraint check."""
    name: str
    expected: float
    actual: float
    passed: bool
    tolerance: float

    @property
    def error(self) -> float:
        return abs(self.actual - self.expected)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"{self.name}: expected={self.expected:.6f}, actual={self.actual:.6f}, error={self.error:.2e} [{status}]"


@dataclass
class VerificationReport:
    """Complete verification report for a polynomial set."""
    results: List[ConstraintResult]

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def num_passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def num_failed(self) -> int:
        return sum(1 for r in self.results if not r.passed)

    def __str__(self) -> str:
        lines = ["PRZZ Constraint Verification Report", "=" * 40]
        for r in self.results:
            lines.append(str(r))
        lines.append("-" * 40)
        lines.append(f"Passed: {self.num_passed}/{len(self.results)}")
        status = "ALL CONSTRAINTS SATISFIED" if self.all_passed else "CONSTRAINTS VIOLATED"
        lines.append(status)
        return "\n".join(lines)


def verify_przz_constraints(
    P1: P1Polynomial,
    P2: PellPolynomial,
    P3: PellPolynomial,
    Q: QPolynomial,
    tolerance: float = 1e-6
) -> VerificationReport:
    """
    Verify that polynomials satisfy PRZZ admissibility constraints.

    Constraints (from PRZZ paper):
    1. P₁(0) = 0 (boundary condition)
    2. P₁(1) = 1 (normalization)
    3. P₂(0) = 0 (boundary condition)
    4. P₃(0) = 0 (boundary condition)
    5. Q(0) = 1 (normalization)

    Note: Q symmetry (Q(x) = Q(1-x)) is NOT a PRZZ requirement.

    Args:
        P1: P1 polynomial
        P2: P2 polynomial
        P3: P3 polynomial
        Q: Q polynomial
        tolerance: Tolerance for floating point comparison (default 1e-6
                   to allow for printed coefficient rounding)

    Returns:
        VerificationReport with all constraint check results
    """
    results = []

    # Get monomial forms for evaluation
    p1_mono = P1.to_monomial()
    p2_mono = P2.to_monomial()
    p3_mono = P3.to_monomial()
    q_mono = Q.to_monomial()

    # 1. P₁(0) = 0
    p1_at_0 = float(p1_mono.eval(np.array([0.0]))[0])
    results.append(ConstraintResult(
        name="P₁(0) = 0",
        expected=0.0,
        actual=p1_at_0,
        passed=abs(p1_at_0) < tolerance,
        tolerance=tolerance
    ))

    # 2. P₁(1) = 1
    p1_at_1 = float(p1_mono.eval(np.array([1.0]))[0])
    results.append(ConstraintResult(
        name="P₁(1) = 1",
        expected=1.0,
        actual=p1_at_1,
        passed=abs(p1_at_1 - 1.0) < tolerance,
        tolerance=tolerance
    ))

    # 3. P₂(0) = 0
    p2_at_0 = float(p2_mono.eval(np.array([0.0]))[0])
    results.append(ConstraintResult(
        name="P₂(0) = 0",
        expected=0.0,
        actual=p2_at_0,
        passed=abs(p2_at_0) < tolerance,
        tolerance=tolerance
    ))

    # 4. P₃(0) = 0
    p3_at_0 = float(p3_mono.eval(np.array([0.0]))[0])
    results.append(ConstraintResult(
        name="P₃(0) = 0",
        expected=0.0,
        actual=p3_at_0,
        passed=abs(p3_at_0) < tolerance,
        tolerance=tolerance
    ))

    # 5. Q(0) = 1
    q_at_0 = float(q_mono.eval(np.array([0.0]))[0])
    results.append(ConstraintResult(
        name="Q(0) = 1",
        expected=1.0,
        actual=q_at_0,
        passed=abs(q_at_0 - 1.0) < tolerance,
        tolerance=tolerance
    ))

    return VerificationReport(results=results)


def verify_przz_baseline() -> VerificationReport:
    """
    Verify PRZZ baseline (kappa) polynomials satisfy constraints.

    Returns:
        VerificationReport
    """
    P1, P2, P3, Q = load_przz_polynomials(enforce_Q0=False)
    return verify_przz_constraints(P1, P2, P3, Q)


def verify_przz_kappa_star() -> VerificationReport:
    """
    Verify PRZZ kappa* polynomials satisfy constraints.

    Returns:
        VerificationReport
    """
    P1, P2, P3, Q = load_przz_polynomials_kappa_star(enforce_Q0=False)
    return verify_przz_constraints(P1, P2, P3, Q)

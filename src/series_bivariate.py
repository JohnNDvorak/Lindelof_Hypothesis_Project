"""
src/series_bivariate.py
Phase 26B: Bivariate Truncated Series Engine

A bivariate polynomial series that tracks coefficients of x^i y^j for:
    0 ≤ i ≤ max_dx
    0 ≤ j ≤ max_dy

This replaces the nilpotent TruncatedSeries for cases where we need
higher-order terms (x^2, xy^2, etc.) for general (ℓ₁, ℓ₂) pairs.

Key Operations:
    - Addition/subtraction (truncated)
    - Multiplication (truncated at max degrees)
    - Exponential of linear form: exp(a0 + ax*x + ay*y)
    - Polynomial composition: P(a0 + ax*x + ay*y)
    - Coefficient extraction: [x^i y^j]

PRZZ Application:
    For pair (ℓ₁, ℓ₂), we build the unified bracket as a BivariateSeries
    with max_dx=ℓ₁, max_dy=ℓ₂, then extract the coefficient of x^ℓ₁ y^ℓ₂.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Union, List
import math


@dataclass
class BivariateSeries:
    """
    Bivariate polynomial series truncated at (max_dx, max_dy).

    Represents: sum_{i=0}^{max_dx} sum_{j=0}^{max_dy} c_{i,j} x^i y^j

    Coefficients are stored as dict[(i, j)] -> float.
    Missing keys are treated as 0.
    """

    max_dx: int
    max_dy: int
    coeffs: Dict[Tuple[int, int], float] = field(default_factory=dict)

    def __post_init__(self):
        """Ensure max degrees are non-negative."""
        if self.max_dx < 0 or self.max_dy < 0:
            raise ValueError(f"Max degrees must be non-negative: ({self.max_dx}, {self.max_dy})")

    # =========================================================================
    # CONSTRUCTORS
    # =========================================================================

    @classmethod
    def from_scalar(cls, value: float, max_dx: int, max_dy: int) -> BivariateSeries:
        """Create constant series: value * x^0 y^0."""
        if value == 0.0:
            return cls(max_dx=max_dx, max_dy=max_dy, coeffs={})
        return cls(max_dx=max_dx, max_dy=max_dy, coeffs={(0, 0): value})

    @classmethod
    def x(cls, max_dx: int, max_dy: int) -> BivariateSeries:
        """Create series for variable x: 0 + 1*x."""
        if max_dx < 1:
            return cls(max_dx=max_dx, max_dy=max_dy, coeffs={})
        return cls(max_dx=max_dx, max_dy=max_dy, coeffs={(1, 0): 1.0})

    @classmethod
    def y(cls, max_dx: int, max_dy: int) -> BivariateSeries:
        """Create series for variable y: 0 + 1*y."""
        if max_dy < 1:
            return cls(max_dx=max_dx, max_dy=max_dy, coeffs={})
        return cls(max_dx=max_dx, max_dy=max_dy, coeffs={(0, 1): 1.0})

    @classmethod
    def zero(cls, max_dx: int, max_dy: int) -> BivariateSeries:
        """Create zero series."""
        return cls(max_dx=max_dx, max_dy=max_dy, coeffs={})

    @classmethod
    def one(cls, max_dx: int, max_dy: int) -> BivariateSeries:
        """Create unit series: 1."""
        return cls.from_scalar(1.0, max_dx, max_dy)

    # =========================================================================
    # ACCESSORS
    # =========================================================================

    def extract(self, i: int, j: int) -> float:
        """Extract coefficient of x^i y^j."""
        if i < 0 or j < 0:
            return 0.0
        if i > self.max_dx or j > self.max_dy:
            return 0.0
        return self.coeffs.get((i, j), 0.0)

    def __getitem__(self, key: Tuple[int, int]) -> float:
        """Allow series[(i, j)] access."""
        return self.extract(key[0], key[1])

    # =========================================================================
    # ARITHMETIC OPERATIONS
    # =========================================================================

    def _check_compatible(self, other: BivariateSeries) -> None:
        """Ensure two series have same max degrees."""
        if self.max_dx != other.max_dx or self.max_dy != other.max_dy:
            raise ValueError(
                f"Incompatible series degrees: ({self.max_dx}, {self.max_dy}) vs ({other.max_dx}, {other.max_dy})"
            )

    def __add__(self, other: Union[BivariateSeries, float, int]) -> BivariateSeries:
        """Add two series or series + scalar."""
        if isinstance(other, (int, float)):
            other = BivariateSeries.from_scalar(float(other), self.max_dx, self.max_dy)
        self._check_compatible(other)

        new_coeffs = dict(self.coeffs)
        for key, val in other.coeffs.items():
            if key in new_coeffs:
                new_coeffs[key] += val
            else:
                new_coeffs[key] = val

        # Remove zero coefficients
        new_coeffs = {k: v for k, v in new_coeffs.items() if v != 0.0}
        return BivariateSeries(max_dx=self.max_dx, max_dy=self.max_dy, coeffs=new_coeffs)

    def __radd__(self, other: Union[float, int]) -> BivariateSeries:
        """Scalar + series."""
        return self.__add__(other)

    def __sub__(self, other: Union[BivariateSeries, float, int]) -> BivariateSeries:
        """Subtract: self - other."""
        if isinstance(other, (int, float)):
            other = BivariateSeries.from_scalar(float(other), self.max_dx, self.max_dy)
        return self + (-other)

    def __rsub__(self, other: Union[float, int]) -> BivariateSeries:
        """Scalar - series."""
        return BivariateSeries.from_scalar(float(other), self.max_dx, self.max_dy) - self

    def __neg__(self) -> BivariateSeries:
        """Negate series."""
        return BivariateSeries(
            max_dx=self.max_dx,
            max_dy=self.max_dy,
            coeffs={k: -v for k, v in self.coeffs.items()}
        )

    def __mul__(self, other: Union[BivariateSeries, float, int]) -> BivariateSeries:
        """
        Multiply two series (truncated at max degrees).

        If series × scalar, scale all coefficients.
        If series × series, convolve and truncate.
        """
        if isinstance(other, (int, float)):
            if other == 0.0:
                return BivariateSeries.zero(self.max_dx, self.max_dy)
            return BivariateSeries(
                max_dx=self.max_dx,
                max_dy=self.max_dy,
                coeffs={k: v * other for k, v in self.coeffs.items()}
            )

        self._check_compatible(other)

        new_coeffs: Dict[Tuple[int, int], float] = {}

        for (i1, j1), c1 in self.coeffs.items():
            for (i2, j2), c2 in other.coeffs.items():
                new_i = i1 + i2
                new_j = j1 + j2

                # Truncate at max degrees
                if new_i > self.max_dx or new_j > self.max_dy:
                    continue

                key = (new_i, new_j)
                if key in new_coeffs:
                    new_coeffs[key] += c1 * c2
                else:
                    new_coeffs[key] = c1 * c2

        # Remove zero coefficients
        new_coeffs = {k: v for k, v in new_coeffs.items() if v != 0.0}
        return BivariateSeries(max_dx=self.max_dx, max_dy=self.max_dy, coeffs=new_coeffs)

    def __rmul__(self, other: Union[float, int]) -> BivariateSeries:
        """Scalar * series."""
        return self.__mul__(other)

    def __pow__(self, n: int) -> BivariateSeries:
        """
        Raise series to integer power n.

        Uses binary exponentiation for efficiency.
        """
        if n < 0:
            raise ValueError("Negative powers not supported")
        if n == 0:
            return BivariateSeries.one(self.max_dx, self.max_dy)
        if n == 1:
            return BivariateSeries(
                max_dx=self.max_dx,
                max_dy=self.max_dy,
                coeffs=dict(self.coeffs)
            )

        # Binary exponentiation
        result = BivariateSeries.one(self.max_dx, self.max_dy)
        base = self
        while n > 0:
            if n % 2 == 1:
                result = result * base
            base = base * base
            n //= 2
        return result

    # =========================================================================
    # SPECIAL FUNCTIONS
    # =========================================================================

    @classmethod
    def exp_linear(cls, a0: float, ax: float, ay: float, max_dx: int, max_dy: int) -> BivariateSeries:
        """
        Compute exp(a0 + ax*x + ay*y) truncated to (max_dx, max_dy).

        Uses factorization:
            exp(a0 + ax*x + ay*y) = exp(a0) * exp(ax*x) * exp(ay*y)

        Where:
            exp(ax*x) = sum_{i=0}^{max_dx} (ax*x)^i / i! = sum_{i=0}^{max_dx} ax^i/i! * x^i
            exp(ay*y) = sum_{j=0}^{max_dy} (ay*y)^j / j! = sum_{j=0}^{max_dy} ay^j/j! * y^j

        The product is:
            exp(a0) * sum_{i,j} (ax^i/i!) * (ay^j/j!) * x^i * y^j
        """
        exp_a0 = math.exp(a0)

        # Precompute ax^i / i! for i = 0..max_dx
        ax_terms = [1.0]  # ax^0 / 0! = 1
        for i in range(1, max_dx + 1):
            ax_terms.append(ax_terms[-1] * ax / i)

        # Precompute ay^j / j! for j = 0..max_dy
        ay_terms = [1.0]
        for j in range(1, max_dy + 1):
            ay_terms.append(ay_terms[-1] * ay / j)

        # Build coefficients
        coeffs: Dict[Tuple[int, int], float] = {}
        for i in range(max_dx + 1):
            for j in range(max_dy + 1):
                val = exp_a0 * ax_terms[i] * ay_terms[j]
                if val != 0.0:
                    coeffs[(i, j)] = val

        return cls(max_dx=max_dx, max_dy=max_dy, coeffs=coeffs)

    def compose_polynomial(
        self,
        poly_coeffs: List[float],
        a0: float,
        ax: float,
        ay: float,
    ) -> BivariateSeries:
        """
        Compute P(a0 + ax*x + ay*y) for polynomial P.

        poly_coeffs: [c0, c1, c2, ...] where P(z) = c0 + c1*z + c2*z^2 + ...

        Uses Horner's method for efficiency:
            P(z) = c0 + z*(c1 + z*(c2 + z*(...)))
        """
        if not poly_coeffs:
            return BivariateSeries.zero(self.max_dx, self.max_dy)

        # Build linear form z = a0 + ax*x + ay*y
        z = BivariateSeries.from_scalar(a0, self.max_dx, self.max_dy)
        z = z + BivariateSeries.x(self.max_dx, self.max_dy) * ax
        z = z + BivariateSeries.y(self.max_dx, self.max_dy) * ay

        # Horner's method: P(z) = c0 + z*(c1 + z*(c2 + ...))
        result = BivariateSeries.from_scalar(poly_coeffs[-1], self.max_dx, self.max_dy)
        for i in range(len(poly_coeffs) - 2, -1, -1):
            result = result * z + poly_coeffs[i]

        return result

    def exp(self) -> BivariateSeries:
        """
        Compute exp(self) for series with zero constant term.

        For series N with N(0,0) = 0:
            exp(N) = sum_{k=0}^∞ N^k / k!

        The sum terminates because N^k is zero for k > max_dx + max_dy
        (due to truncation).
        """
        const = self.extract(0, 0)
        if const != 0.0:
            # exp(const + N) = exp(const) * exp(N)
            nilpotent = self - const
            exp_nilpotent = nilpotent.exp()
            return exp_nilpotent * math.exp(const)

        # Self has zero constant term - compute exp via Taylor series
        result = BivariateSeries.one(self.max_dx, self.max_dy)
        term = BivariateSeries.one(self.max_dx, self.max_dy)

        # Maximum useful power is max_dx + max_dy + 1
        max_k = self.max_dx + self.max_dy + 1
        for k in range(1, max_k + 1):
            term = term * self * (1.0 / k)
            result = result + term

            # Early termination if term is zero
            if not term.coeffs:
                break

        return result

    # =========================================================================
    # UTILITY
    # =========================================================================

    def __repr__(self) -> str:
        """String representation."""
        if not self.coeffs:
            return f"BivariateSeries(0, max=({self.max_dx},{self.max_dy}))"

        terms = []
        for (i, j), c in sorted(self.coeffs.items()):
            if i == 0 and j == 0:
                terms.append(f"{c:.6g}")
            elif i == 0:
                terms.append(f"{c:.6g}*y^{j}")
            elif j == 0:
                terms.append(f"{c:.6g}*x^{i}")
            else:
                terms.append(f"{c:.6g}*x^{i}*y^{j}")

        expr = " + ".join(terms[:10])
        if len(terms) > 10:
            expr += f" + ... ({len(terms)} terms)"
        return f"BivariateSeries({expr}, max=({self.max_dx},{self.max_dy}))"

    def total_terms(self) -> int:
        """Return number of non-zero terms."""
        return len(self.coeffs)

    def evaluate(self, x_val: float, y_val: float) -> float:
        """Evaluate series at (x_val, y_val)."""
        result = 0.0
        for (i, j), c in self.coeffs.items():
            result += c * (x_val ** i) * (y_val ** j)
        return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def build_exp_bracket(
    a0: float,
    ax: float,
    ay: float,
    max_dx: int,
    max_dy: int,
) -> BivariateSeries:
    """
    Build exp(a0 + ax*x + ay*y) as bivariate series.

    For the unified bracket exp factor:
        exp(2Rt + Rθ(2t-1)(x+y)) = exp(2Rt) * exp(Rθ(2t-1)*x) * exp(Rθ(2t-1)*y)

    Call with:
        a0 = 2Rt
        ax = Rθ(2t-1)
        ay = Rθ(2t-1)
    """
    return BivariateSeries.exp_linear(a0, ax, ay, max_dx, max_dy)


def build_log_factor(theta: float, max_dx: int, max_dy: int) -> BivariateSeries:
    """
    Build log factor (1/θ + x + y) as bivariate series.

    This is the log(N^{x+y}T) = L(1+θ(x+y)) = (1+θ(x+y))/θ = 1/θ + x + y
    """
    result = BivariateSeries.from_scalar(1.0 / theta, max_dx, max_dy)
    result = result + BivariateSeries.x(max_dx, max_dy)
    result = result + BivariateSeries.y(max_dx, max_dy)
    return result


def build_P_factor(
    poly_coeffs: List[float],
    u: float,
    var: str,
    max_dx: int,
    max_dy: int,
) -> BivariateSeries:
    """
    Build P(u + var) as bivariate series, where var is "x" or "y".

    For P(u + x): argument a0=u, ax=1, ay=0
    For P(u + y): argument a0=u, ax=0, ay=1
    """
    if var == "x":
        ax, ay = 1.0, 0.0
    elif var == "y":
        ax, ay = 0.0, 1.0
    else:
        raise ValueError(f"var must be 'x' or 'y', got '{var}'")

    dummy = BivariateSeries.zero(max_dx, max_dy)
    return dummy.compose_polynomial(poly_coeffs, a0=u, ax=ax, ay=ay)


def build_Q_factor(
    Q_coeffs: List[float],
    a0: float,
    ax: float,
    ay: float,
    max_dx: int,
    max_dy: int,
) -> BivariateSeries:
    """
    Build Q(a0 + ax*x + ay*y) as bivariate series.

    For Q(A_α) where A_α = t + θ(t-1)x + θt*y:
        a0 = t, ax = θ(t-1), ay = θt

    For Q(A_β) where A_β = t + θt*x + θ(t-1)*y:
        a0 = t, ax = θt, ay = θ(t-1)
    """
    dummy = BivariateSeries.zero(max_dx, max_dy)
    return dummy.compose_polynomial(Q_coeffs, a0=a0, ax=ax, ay=ay)

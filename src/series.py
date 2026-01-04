"""
src/series.py
Truncated Multi-Variable Taylor Series Engine.

Implements arithmetic for series of the form:
S = c_0 + c_x*x + c_y*y + c_xy*xy + ...

Constraints:
1. Variables are nilpotent: x^2 = y^2 = 0.
2. Cross-terms commute: xy = yx.
3. Representation: Terms are indexed by integer bitmasks.

Bitmask convention:
- Variable at index i maps to bit i (mask = 1 << i)
- Constant term has mask 0
- Product xy has mask (mask_x | mask_y)
- Multiplication truncates when masks overlap (nilpotency)
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Union, Optional

# Type alias for coefficient values
CoeffType = Union[float, np.ndarray]


class TruncatedSeries:
    """
    Multi-variable truncated Taylor series with nilpotent variables.

    Each variable satisfies x^2 = 0, so terms are indexed by bitmasks
    where each bit can appear at most once.

    Attributes:
        var_names: Tuple of variable names (order defines bit positions)
        n_vars: Number of variables
        coeffs: Dict mapping bitmask (int) to coefficient (float or ndarray)
    """

    def __init__(
        self,
        var_names: Tuple[str, ...],
        coeffs: Optional[Dict[int, CoeffType]] = None
    ):
        """
        Initialize a TruncatedSeries.

        Args:
            var_names: Tuple of variable names. Position determines bit index.
                      Must contain unique names.
            coeffs: Dict mapping bitmask to coefficient. If None, initializes
                   to zero constant term {0: 0.0}.

        Raises:
            ValueError: If var_names contains duplicate entries.
        """
        # Validate uniqueness of variable names
        if len(set(var_names)) != len(var_names):
            raise ValueError(
                f"var_names must contain unique entries, got: {var_names}"
            )

        self.var_names = var_names
        self.n_vars = len(var_names)

        # Precompute name -> bit mapping for O(1) lookups
        self._name_to_bit: Dict[str, int] = {
            name: 1 << i for i, name in enumerate(var_names)
        }

        if coeffs is None:
            self.coeffs: Dict[int, CoeffType] = {0: np.array(0.0)}
        else:
            # Ensure coefficients are numpy arrays for consistent arithmetic
            self.coeffs = {
                mask: np.asarray(val) for mask, val in coeffs.items()
            }

    @classmethod
    def from_scalar(
        cls,
        value: CoeffType,
        var_names: Tuple[str, ...]
    ) -> TruncatedSeries:
        """
        Create a constant series (only mask-0 term).

        Args:
            value: The constant value (scalar or array)
            var_names: Variable names for this series

        Returns:
            TruncatedSeries with only constant term
        """
        return cls(var_names, {0: np.asarray(value)})

    @classmethod
    def variable(
        cls,
        name: str,
        var_names: Tuple[str, ...]
    ) -> TruncatedSeries:
        """
        Create a series representing a single linear variable.

        Args:
            name: Variable name (must be in var_names)
            var_names: Variable names tuple

        Returns:
            TruncatedSeries with coefficient 1 at the variable's bit position

        Raises:
            ValueError: If name is not in var_names
        """
        if name not in var_names:
            raise ValueError(f"Variable '{name}' not in {var_names}")

        idx = var_names.index(name)
        mask = 1 << idx
        # Note: We could use _name_to_bit here, but cls() hasn't been called yet.
        # The mapping is created in __init__ for use in extract() etc.
        return cls(var_names, {mask: np.array(1.0)})

    def _assert_same_vars(self, other: TruncatedSeries) -> None:
        """
        Verify that another TruncatedSeries has the same var_names.

        This is critical: bitmasks only have meaning relative to var_names.
        Combining series with different var_names would silently produce
        mathematically incorrect results.

        Raises:
            ValueError: If var_names differ between self and other.
        """
        if self.var_names != other.var_names:
            raise ValueError(
                f"Cannot combine series with different var_names: "
                f"{self.var_names} vs {other.var_names}"
            )

    def _coerce_other(
        self,
        other: Union[TruncatedSeries, float, np.ndarray]
    ) -> TruncatedSeries:
        """Convert scalar to TruncatedSeries if needed, with compatibility check."""
        if isinstance(other, TruncatedSeries):
            self._assert_same_vars(other)
            return other
        return TruncatedSeries.from_scalar(other, self.var_names)

    def __add__(
        self,
        other: Union[TruncatedSeries, float, np.ndarray]
    ) -> TruncatedSeries:
        """
        Add two series or a series and a scalar.

        Terms with the same mask have their coefficients summed.
        """
        other_series = self._coerce_other(other)

        # Collect all masks
        all_masks = set(self.coeffs.keys()) | set(other_series.coeffs.keys())

        new_coeffs: Dict[int, CoeffType] = {}
        for mask in all_masks:
            c1 = self.coeffs.get(mask, 0.0)
            c2 = other_series.coeffs.get(mask, 0.0)
            new_coeffs[mask] = np.asarray(c1) + np.asarray(c2)

        return TruncatedSeries(self.var_names, new_coeffs)

    def __radd__(
        self,
        other: Union[float, np.ndarray]
    ) -> TruncatedSeries:
        """Right addition (scalar + series)."""
        return self.__add__(other)

    def __sub__(
        self,
        other: Union[TruncatedSeries, float, np.ndarray]
    ) -> TruncatedSeries:
        """Subtract series or scalar."""
        return self + (-self._coerce_other(other))

    def __rsub__(
        self,
        other: Union[float, np.ndarray]
    ) -> TruncatedSeries:
        """Right subtraction (scalar - series)."""
        return (-self) + other

    def __neg__(self) -> TruncatedSeries:
        """Unary negation."""
        new_coeffs = {mask: -coeff for mask, coeff in self.coeffs.items()}
        return TruncatedSeries(self.var_names, new_coeffs)

    def __mul__(
        self,
        other: Union[TruncatedSeries, float, np.ndarray]
    ) -> TruncatedSeries:
        """
        Multiply two series or a series and a scalar.

        For series multiplication:
        - If masks overlap (m1 & m2 != 0), the term vanishes (nilpotency)
        - Otherwise, new mask is m1 | m2
        """
        # Scalar multiplication
        if not isinstance(other, TruncatedSeries):
            other_val = np.asarray(other)
            new_coeffs = {
                mask: coeff * other_val for mask, coeff in self.coeffs.items()
            }
            return TruncatedSeries(self.var_names, new_coeffs)

        # Series multiplication - verify compatibility first
        self._assert_same_vars(other)
        new_coeffs: Dict[int, CoeffType] = {}

        for m1, c1 in self.coeffs.items():
            for m2, c2 in other.coeffs.items():
                # Nilpotent truncation: overlapping bits vanish
                if m1 & m2:
                    continue

                new_mask = m1 | m2
                product = np.asarray(c1) * np.asarray(c2)

                if new_mask in new_coeffs:
                    new_coeffs[new_mask] = new_coeffs[new_mask] + product
                else:
                    new_coeffs[new_mask] = product

        # If no terms survived, return zero
        if not new_coeffs:
            new_coeffs = {0: np.array(0.0)}

        return TruncatedSeries(self.var_names, new_coeffs)

    def __rmul__(
        self,
        other: Union[float, np.ndarray]
    ) -> TruncatedSeries:
        """Right multiplication (scalar * series)."""
        return self.__mul__(other)

    def exp(self) -> TruncatedSeries:
        """
        Compute exp(S) for this series S.

        Uses the identity: exp(c + N) = exp(c) * exp(N)
        where c is the constant term and N contains nilpotent terms.

        Since N is nilpotent (N^k = 0 for k > n_vars), the Taylor series
        for exp(N) terminates:
            exp(N) = 1 + N + N^2/2! + N^3/3! + ... + N^{n_vars}/n_vars!
        """
        # Extract constant term
        c0 = self.coeffs.get(0, np.array(0.0))
        exp_c0 = np.exp(c0)

        # Build nilpotent part (non-constant terms)
        nilpotent_coeffs = {
            mask: coeff for mask, coeff in self.coeffs.items() if mask != 0
        }

        if not nilpotent_coeffs:
            # Pure constant: exp(c) = e^c
            return TruncatedSeries.from_scalar(exp_c0, self.var_names)

        N = TruncatedSeries(self.var_names, nilpotent_coeffs)

        # exp(N) = sum_{k=0}^{n_vars} N^k / k!
        # Start with 1
        exp_N = TruncatedSeries.from_scalar(1.0, self.var_names)
        N_power = TruncatedSeries.from_scalar(1.0, self.var_names)  # N^0
        factorial = 1.0

        for k in range(1, self.n_vars + 1):
            N_power = N_power * N
            factorial *= k

            # Check if N_power is all zeros (truncated)
            if all(
                np.all(np.asarray(c) == 0)
                for c in N_power.coeffs.values()
            ):
                break

            exp_N = exp_N + N_power * (1.0 / factorial)

        # exp(c + N) = exp(c) * exp(N)
        return exp_N * exp_c0

    def extract(self, deriv_vars: Tuple[str, ...]) -> CoeffType:
        """
        Extract the coefficient for a specific derivative/monomial.

        Args:
            deriv_vars: Tuple of variable names whose product defines the term.
                       Empty tuple () extracts the constant term.
                       ('x', 'y') extracts the coefficient of xy.

        Returns:
            The coefficient value (float or ndarray). Returns 0.0 if the
            term doesn't exist or if deriv_vars contains duplicates
            (since x^2 = 0 in the nilpotent model).
        """
        # Check for duplicate variables - duplicates mean x^2 which is 0
        if len(deriv_vars) != len(set(deriv_vars)):
            return np.array(0.0)

        # Build mask from variable names using cached mapping
        mask = 0
        for name in deriv_vars:
            if name not in self._name_to_bit:
                raise ValueError(f"Variable '{name}' not in {self.var_names}")
            mask |= self._name_to_bit[name]

        return self.coeffs.get(mask, np.array(0.0))

    def __repr__(self) -> str:
        """String representation for debugging."""
        terms = []
        for mask in sorted(self.coeffs.keys()):
            coeff = self.coeffs[mask]
            if mask == 0:
                term_str = f"{coeff}"
            else:
                vars_in_term = [
                    self.var_names[i]
                    for i in range(self.n_vars)
                    if mask & (1 << i)
                ]
                term_str = f"{coeff}*{''.join(vars_in_term)}"
            terms.append(term_str)
        return f"TruncatedSeries({', '.join(terms)})"

"""
src/term_dsl.py
Term DSL for PRZZ pair computations.

This module defines the data structures for representing integral terms
in the PRZZ framework. Each term is a derivative-at-0 of a multi-dimensional
integral with factors: P, Q, exp, and algebraic prefactors.

Key components:
- SeriesContext: Owns canonical var_names for all series operations
- AffineExpr: Represents a0(u,t) + Σᵢ aᵢ(u,t) · varᵢ
- PolyFactor: Polynomial factor P(argument)^power
- ExpFactor: Exponential factor exp(scale * argument)
- Term: Complete specification of one integral term

Design principles (from GPT review):
- Dtype-safe lifting: preserves complex types
- Shape invariants: all evaluations return grid-shaped arrays
- Zero-pruning: skip identically-zero coefficients to keep len(lin) minimal
- composition.py is the ONLY Taylor expansion engine
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Callable, Union, Tuple, List, Optional
import numpy as np

from src.series import TruncatedSeries
from src.composition import (
    compose_polynomial_on_affine,
    compose_exp_on_affine,
    compose_profile_on_affine_grid,
    PolyLike
)


# Type alias: scalar, precomputed array, or callable
GridFunc = Union[float, complex, np.ndarray, Callable[[np.ndarray, np.ndarray], np.ndarray]]


def _eval_gridfunc(
    f: GridFunc,
    U: np.ndarray,
    T: np.ndarray,
    context: str = "GridFunc"
) -> np.ndarray:
    """
    Evaluate a GridFunc on the grid, enforcing shape invariants.

    This centralizes grid evaluation and catches shape mismatches early.

    Args:
        f: GridFunc to evaluate (scalar, array, or callable)
        U: Grid array for u variable (defines expected shape)
        T: Grid array for t variable
        context: Description for error messages (e.g., "a0" or "coeff['x1']")

    Returns:
        Array with shape U.shape and appropriate dtype

    Raises:
        ValueError: If callable returns wrong shape
    """
    # Handle callable
    if callable(f):
        val = f(U, T)
    else:
        val = f

    arr = np.asarray(val)

    # Scalar -> broadcast to grid shape
    if arr.shape == ():
        return np.full(U.shape, arr, dtype=np.result_type(U, arr))

    # Already an array - verify shape matches
    if arr.shape != U.shape:
        raise ValueError(
            f"{context} returned shape {arr.shape}, expected {U.shape}. "
            f"GridFunc must return scalar or array matching grid shape."
        )

    # Ensure consistent dtype (preserve complex)
    return arr.astype(np.result_type(U, arr), copy=False)


@dataclass
class SeriesContext:
    """
    Context for creating series with canonical variable names.

    All series created through this context share the same var_names,
    preventing mismatches in binary operations.

    Attributes:
        var_names: Tuple of variable names in canonical order

    Example:
        >>> ctx = SeriesContext(var_names=("x1", "y1"))
        >>> x = ctx.variable_series("x1")
        >>> y = ctx.variable_series("y1")
        >>> product = x * y  # Safe - same var_names
    """
    var_names: Tuple[str, ...]

    def zero_series(self) -> TruncatedSeries:
        """Create a zero series with this context's var_names."""
        return TruncatedSeries.from_scalar(0.0, self.var_names)

    def scalar_series(self, value: Union[float, np.ndarray]) -> TruncatedSeries:
        """Create a scalar (constant) series with this context's var_names."""
        return TruncatedSeries.from_scalar(value, self.var_names)

    def variable_series(self, name: str) -> TruncatedSeries:
        """Create a series representing a single variable."""
        if name not in self.var_names:
            raise ValueError(f"Variable '{name}' not in var_names {self.var_names}")
        return TruncatedSeries.variable(name, self.var_names)


@dataclass(frozen=True)
class AffineExpr:
    """
    Represents an affine expression: a₀(u,t) + Σᵢ aᵢ(u,t) · varᵢ

    The coefficients can be either:
    - Constants (float/complex): lifted to grid shape via np.full
    - Callables: called with (U, T) arrays to evaluate on grid

    Attributes:
        a0: Base term - constant or callable(U, T) -> array
        var_coeffs: Dict mapping variable name to its coefficient
                    Each coefficient is constant or callable(U, T) -> array

    Shape invariants:
    - evaluate_a0(U, T) returns array with shape U.shape
    - evaluate_coeff(name, U, T) returns array with shape U.shape

    Example:
        >>> # Expression: t + θ·t·x + (θ·t - θ)·y
        >>> expr = AffineExpr(
        ...     a0=lambda U, T: T,
        ...     var_coeffs={
        ...         "x": lambda U, T: theta * T,
        ...         "y": lambda U, T: theta * T - theta
        ...     }
        ... )
    """
    a0: GridFunc
    var_coeffs: Dict[str, GridFunc]

    def evaluate_a0(self, U: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Evaluate the base term on grid. Preserves dtype for complex.

        Args:
            U: Grid array for u variable
            T: Grid array for t variable

        Returns:
            Array with shape U.shape

        Raises:
            ValueError: If callable returns wrong shape
        """
        return _eval_gridfunc(self.a0, U, T, context="a0")

    def evaluate_coeff(self, var: str, U: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Evaluate coefficient of var on grid. Preserves dtype for complex.

        Args:
            var: Variable name
            U: Grid array for u variable
            T: Grid array for t variable

        Returns:
            Array with shape U.shape (zeros if var not in var_coeffs)

        Raises:
            ValueError: If callable returns wrong shape
        """
        c = self.var_coeffs.get(var, 0.0)
        return _eval_gridfunc(c, U, T, context=f"coeff['{var}']")

    def to_u0_lin(
        self,
        U: np.ndarray,
        T: np.ndarray,
        ctx: SeriesContext
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Extract (u0, lin) for use with compose_polynomial_on_affine.

        Zero-pruning: coefficients that are identically zero are excluded
        from lin, keeping len(lin) minimal for efficient truncation.

        Args:
            U: Grid array for u variable
            T: Grid array for t variable
            ctx: SeriesContext for variable names

        Returns:
            Tuple of:
            - u0: evaluated base term
            - lin: dict mapping var_name to evaluated coefficient (non-zero only)

        Raises:
            ValueError: If any var_coeffs key is not in ctx.var_names
        """
        u0 = self.evaluate_a0(U, T)
        lin = {}
        for var in self.var_coeffs.keys():
            # Validate var is in context
            if var not in ctx.var_names:
                raise ValueError(
                    f"AffineExpr var_coeffs key '{var}' not in "
                    f"ctx.var_names {ctx.var_names}"
                )
            coeff = self.evaluate_coeff(var, U, T)
            # Zero-pruning: skip identically zero coefficients
            if not np.all(coeff == 0):
                lin[var] = coeff
        return u0, lin

    def to_series(
        self,
        U: np.ndarray,
        T: np.ndarray,
        ctx: SeriesContext
    ) -> TruncatedSeries:
        """
        Convert to a TruncatedSeries representing the affine expression.

        Zero-pruning: coefficients that are identically zero are not added
        to the series, keeping the coefficient dict minimal.

        Args:
            U: Grid array for u variable
            T: Grid array for t variable
            ctx: SeriesContext for variable names

        Returns:
            TruncatedSeries: a0 + Σ aᵢ·varᵢ (non-zero terms only)

        Raises:
            ValueError: If any var_coeffs key is not in ctx.var_names
        """
        result = ctx.scalar_series(self.evaluate_a0(U, T))
        for var in self.var_coeffs.keys():
            # Validate var is in context
            if var not in ctx.var_names:
                raise ValueError(
                    f"AffineExpr var_coeffs key '{var}' not in "
                    f"ctx.var_names {ctx.var_names}"
                )
            coeff = self.evaluate_coeff(var, U, T)
            # Zero-pruning: skip identically zero coefficients
            if not np.all(coeff == 0):
                result = result + ctx.variable_series(var) * coeff
        return result


@dataclass(frozen=True)
class PolyFactor:
    """
    A polynomial factor: P_name(argument)^power

    Supports omega-case handling for PRZZ kernel structure:
    - omega=None or 0: Case B - standard P(u+X) evaluation
    - omega>0: Case C - kernel with auxiliary a-integral

    Attributes:
        poly_name: Name identifier ("P1", "P2", "P3", "Q")
        argument: AffineExpr for the polynomial argument
        power: Exponent (default 1). For Q² use power=2.
        omega: Omega value for Case C handling (None or 0 = Case B, >0 = Case C)

    Example:
        >>> # P₁(x + u) with Case B handling
        >>> factor = PolyFactor(
        ...     poly_name="P1",
        ...     argument=AffineExpr(
        ...         a0=lambda U, T: U,
        ...         var_coeffs={"x1": 1.0}
        ...     ),
        ...     omega=0  # Case B
        ... )
        >>> # P₂(x₁+x₂ + u) with Case C handling (omega=1)
        >>> factor = PolyFactor(
        ...     poly_name="P2",
        ...     argument=AffineExpr(
        ...         a0=lambda U, T: U,
        ...         var_coeffs={"x1": 1.0, "x2": 1.0}
        ...     ),
        ...     omega=1  # Case C
        ... )
    """
    poly_name: str
    argument: AffineExpr
    power: int = 1
    omega: Optional[int] = None  # None or 0 = Case B, >0 = Case C

    def evaluate(
        self,
        poly: PolyLike,
        U: np.ndarray,
        T: np.ndarray,
        ctx: SeriesContext,
        R: Optional[float] = None,
        theta: Optional[float] = None,
        n_quad_a: int = 40
    ) -> TruncatedSeries:
        """
        Evaluate the polynomial factor as a TruncatedSeries.

        For Case B (omega=None or 0): Uses standard Taylor expansion P(u+X)
        For Case C (omega>0): Uses kernel-based Taylor coefficients K_ω(u;R,θ)

        Args:
            poly: Polynomial object with eval_deriv(x, k)
            U: Grid array for u variable
            T: Grid array for t variable
            ctx: SeriesContext for variable names
            R: R parameter (required for Case C)
            theta: θ parameter (required for Case C)
            n_quad_a: Quadrature points for Case C a-integral

        Returns:
            TruncatedSeries representing profile(argument)^power
        """
        u0, lin = self.argument.to_u0_lin(U, T, ctx)

        # Determine the derivative order needed (number of active variables)
        max_order = len(lin)

        # Dispatch based on omega
        if self.omega is None or self.omega == 0:
            # Case B: standard polynomial composition
            series = compose_polynomial_on_affine(poly, u0, lin, ctx.var_names)
        else:
            # Case C: kernel-based Taylor coefficients
            if R is None or theta is None:
                raise ValueError(
                    f"Case C (omega={self.omega}) requires R and theta parameters"
                )

            # Import here to avoid circular dependency
            from src.mollifier_profiles import case_c_taylor_coeffs

            # u0 is a 2D grid; in this repo's PRZZ term tables, profile arguments
            # are always of the form u + (nilpotent perturbation), so u0 depends
            # only on the u-node (i index) and is constant across the t-axis.
            #
            # Computing Case C coefficients for every (u,t) node is therefore
            # needlessly expensive. We detect common separable shapes and
            # compute only along the varying axis, then broadcast.
            grid_shape = u0.shape
            taylor_grid = np.zeros(grid_shape + (max_order + 1,), dtype=float)

            if u0.ndim == 2 and grid_shape[0] > 0 and grid_shape[1] > 0 and np.all(u0 == u0[:, [0]]):
                # u0 varies only along axis 0 (U-grid): broadcast across t-axis
                u_vals = u0[:, 0]
                for i, u_val in enumerate(u_vals):
                    coeffs = case_c_taylor_coeffs(
                        poly, float(u_val), self.omega, R, theta, max_order, n_quad_a
                    )
                    taylor_grid[i, :, :] = coeffs
            elif u0.ndim == 2 and grid_shape[0] > 0 and grid_shape[1] > 0 and np.all(u0 == u0[[0], :]):
                # u0 varies only along axis 1 (T-grid): broadcast across u-axis
                u_vals = u0[0, :]
                for j, u_val in enumerate(u_vals):
                    coeffs = case_c_taylor_coeffs(
                        poly, float(u_val), self.omega, R, theta, max_order, n_quad_a
                    )
                    taylor_grid[:, j, :] = coeffs
            else:
                # Fallback: general u0(u,t) dependence
                for i in range(grid_shape[0]):
                    for j in range(grid_shape[1]):
                        coeffs = case_c_taylor_coeffs(
                            poly, float(u0[i, j]), self.omega, R, theta, max_order, n_quad_a
                        )
                        taylor_grid[i, j, :] = coeffs

            # Use profile composition with pre-computed grid coefficients
            series = compose_profile_on_affine_grid(taylor_grid, lin, ctx.var_names)

        # Apply power if > 1
        result = series
        for _ in range(self.power - 1):
            result = result * series

        return result


@dataclass(frozen=True)
class ExpFactor:
    """
    An exponential factor: exp(scale * argument)

    Attributes:
        scale: Scaling constant (typically R or 2R)
        argument: AffineExpr for the exponential argument

    Example:
        >>> # exp(R * (t + θ·t·x + (θ·t-θ)·y))
        >>> factor = ExpFactor(
        ...     scale=R,
        ...     argument=Q_arg_alpha
        ... )
    """
    scale: float
    argument: AffineExpr

    def evaluate(
        self,
        U: np.ndarray,
        T: np.ndarray,
        ctx: SeriesContext
    ) -> TruncatedSeries:
        """
        Evaluate the exponential factor as a TruncatedSeries.

        Uses compose_exp_on_affine from composition.py for consistency
        with the "single Taylor engine" principle.

        Args:
            U: Grid array for u variable
            T: Grid array for t variable
            ctx: SeriesContext for variable names

        Returns:
            TruncatedSeries representing exp(scale * argument)
        """
        # Extract (u0, lin) with zero-pruning and validation
        u0, lin = self.argument.to_u0_lin(U, T, ctx)

        # Delegate to composition module
        return compose_exp_on_affine(self.scale, u0, lin, ctx.var_names)


@dataclass(frozen=True)
class CombinedMirrorFactor:
    """
    TeX combined integral factor: ∫_0^1 (N^{x+y}T)^{-s(α+β)} ds × log(N^{x+y}T)

    This represents the combined mirror structure from TeX lines 1503-1510:
        (N^{αx+βy} - T^{-α-β}N^{-βx-αy}) / (α+β)
        = N^{αx+βy} × log(N^{x+y}T) × ∫_0^1 (N^{x+y}T)^{-s(α+β)} ds

    At α = β = -R/L (PRZZ evaluation point):
        - The s-integral becomes: ∫_0^1 exp(2sR(1 + θ(x+y))) ds
        - The log factor becomes: L(1 + θ(x+y)) where L = log T

    The L factor is asymptotic and absorbed elsewhere, so this class computes:
        (1 + θ(x+y)) × ∫_0^1 exp(2sR(1 + θ(x+y))) ds

    Attributes:
        R: The PRZZ R parameter
        theta: The θ = log N / log T parameter (typically 4/7)
        n_quad_s: Number of quadrature points for s-integral (default 20)

    Gate test (scalar limit at x=y=0):
        (exp(2R) - 1) / (2R)

    Example:
        >>> factor = CombinedMirrorFactor(R=1.3036, theta=4/7)
        >>> ctx = SeriesContext(var_names=("x", "y"))
        >>> U = np.array([[0.5]])
        >>> T = np.array([[0.5]])
        >>> series = factor.evaluate(U, T, ctx)
    """
    R: float
    theta: float
    n_quad_s: int = 20

    def evaluate(
        self,
        U: np.ndarray,
        T: np.ndarray,
        ctx: SeriesContext
    ) -> TruncatedSeries:
        """
        Evaluate the combined mirror integral as a TruncatedSeries.

        Computes: (1 + θ(x+y)) × ∫_0^1 exp(2sR(1 + θ(x+y))) ds

        The s-integral is evaluated via Gauss-Legendre quadrature, keeping
        formal variables (x, y) active throughout.

        Args:
            U: Grid array for u variable
            T: Grid array for t variable
            ctx: SeriesContext for variable names (must contain "x" and "y")

        Returns:
            TruncatedSeries with s-integral pre-computed, formal vars active
        """
        from src.quadrature import gauss_legendre_01

        s_nodes, s_weights = gauss_legendre_01(self.n_quad_s)

        # Initialize result as zero series
        result = ctx.zero_series()

        # Integrate over s
        for s_idx, s in enumerate(s_nodes):
            # Exponent: 2*R*s*(1 + θ*(x+y))
            # = 2Rs + 2Rsθx + 2Rsθy
            s_base = 2 * self.R * s * np.ones_like(U)
            s_lin = {
                "x": 2 * self.R * s * self.theta * np.ones_like(U),
                "y": 2 * self.R * s * self.theta * np.ones_like(U),
            }

            # exp(2Rs(1 + θ(x+y))) as a series
            s_exp = compose_exp_on_affine(1.0, s_base, s_lin, ctx.var_names)

            # Accumulate weighted contribution
            result = result + s_exp * s_weights[s_idx]

        # Multiply by log factor: (1 + θ*(x+y))
        log_factor = ctx.scalar_series(np.ones_like(U))
        log_factor = log_factor + ctx.variable_series("x") * self.theta
        log_factor = log_factor + ctx.variable_series("y") * self.theta

        return result * log_factor

    def scalar_limit(self) -> float:
        """
        Compute the scalar limit (x=y=0) analytically.

        Returns:
            (exp(2R) - 1) / (2R)
        """
        return (np.exp(2 * self.R) - 1) / (2 * self.R)


@dataclass(frozen=True)
class CombinedI1Integrand:
    """
    TeX-exact I₁ integrand with Q-shift in mirror branch (Run 19).

    This implements the COMBINED structure where Q operators are applied
    INSIDE the combined object, with proper Q-shift in the minus branch.

    Structure:
        Plus branch:  Q(arg_α) × Q(arg_β) × exp(R·arg_α) × exp(R·arg_β)
        Minus branch: Q(arg_α+σ) × Q(arg_β+σ) × exp(-R·arg_α) × exp(-R·arg_β) × exp(2R)

    Where:
        arg_α = t + θt·x + (θt-θ)·y
        arg_β = t + (θt-θ)·x + θt·y
        σ = 1.0 (Q-shift, NOT derived from α+β)

    Key difference from Run 18's CombinedMirrorFactor:
        - Q factors are INSIDE the combined structure, not multiplied externally
        - Q-shift is applied in the minus branch
        - No separate s-integral factor

    Attributes:
        R: The PRZZ R parameter
        theta: The θ = log N / log T parameter (typically 4/7)
        Q: The Q polynomial
        Q_shifted: The shifted polynomial Q(x+σ)

    Gate test (scalar limit at x=y=0):
        Q(t)² exp(2Rt) + Q(t+σ)² exp(-2Rt) exp(2R)
        = Q(t)² exp(2Rt) + Q(t+σ)² exp(2R(1-t))
    """
    R: float
    theta: float
    Q: PolyLike
    Q_shifted: PolyLike

    def evaluate(
        self,
        U: np.ndarray,
        T: np.ndarray,
        ctx: SeriesContext
    ) -> TruncatedSeries:
        """
        Evaluate the combined I₁ integrand as a TruncatedSeries.

        Computes:
            Plus branch:  Q(arg_α) × Q(arg_β) × exp(R·arg_α) × exp(R·arg_β)
            Minus branch: Q(arg_α+σ) × Q(arg_β+σ) × exp(-R·arg_α) × exp(-R·arg_β) × exp(2R)

        Both branches are combined as a SINGLE series before derivative extraction.

        Args:
            U: Grid array for u variable
            T: Grid array for t variable
            ctx: SeriesContext for variable names (must contain "x" and "y")

        Returns:
            TruncatedSeries combining both branches
        """
        # Build arg_α = t + θt·x + (θt-θ)·y
        arg_alpha_u0 = T
        arg_alpha_lin = {
            "x": self.theta * T,
            "y": self.theta * T - self.theta,
        }

        # Build arg_β = t + (θt-θ)·x + θt·y
        arg_beta_u0 = T
        arg_beta_lin = {
            "x": self.theta * T - self.theta,
            "y": self.theta * T,
        }

        # --- PLUS BRANCH: Q(arg_α) × Q(arg_β) × exp(R·arg_α) × exp(R·arg_β) ---

        # Q(arg_α) at +R
        Q_alpha_plus = compose_polynomial_on_affine(
            self.Q, arg_alpha_u0, arg_alpha_lin, ctx.var_names
        )

        # Q(arg_β) at +R
        Q_beta_plus = compose_polynomial_on_affine(
            self.Q, arg_beta_u0, arg_beta_lin, ctx.var_names
        )

        # exp(R·arg_α)
        exp_alpha_plus = compose_exp_on_affine(
            self.R, arg_alpha_u0, arg_alpha_lin, ctx.var_names
        )

        # exp(R·arg_β)
        exp_beta_plus = compose_exp_on_affine(
            self.R, arg_beta_u0, arg_beta_lin, ctx.var_names
        )

        plus_branch = Q_alpha_plus * Q_beta_plus * exp_alpha_plus * exp_beta_plus

        # --- MINUS BRANCH: Q_shifted(arg_α) × Q_shifted(arg_β) × exp(-R·arg_α) × exp(-R·arg_β) × exp(2R) ---

        # Q_shifted(arg_α) in mirror branch
        Q_alpha_minus = compose_polynomial_on_affine(
            self.Q_shifted, arg_alpha_u0, arg_alpha_lin, ctx.var_names
        )

        # Q_shifted(arg_β) in mirror branch
        Q_beta_minus = compose_polynomial_on_affine(
            self.Q_shifted, arg_beta_u0, arg_beta_lin, ctx.var_names
        )

        # exp(-R·arg_α)
        exp_alpha_minus = compose_exp_on_affine(
            -self.R, arg_alpha_u0, arg_alpha_lin, ctx.var_names
        )

        # exp(-R·arg_β)
        exp_beta_minus = compose_exp_on_affine(
            -self.R, arg_beta_u0, arg_beta_lin, ctx.var_names
        )

        # Mirror prefactor: exp(2R)
        exp_2R = np.exp(2 * self.R)

        minus_branch = Q_alpha_minus * Q_beta_minus * exp_alpha_minus * exp_beta_minus * exp_2R

        # Combine as a SINGLE series
        return plus_branch + minus_branch

    def scalar_limit(self, t_val: float = 0.5) -> float:
        """
        Compute the scalar limit (x=y=0) at a specific t value.

        At x=y=0:
            arg_α = arg_β = t
            Plus: Q(t)² × exp(2Rt)
            Minus: Q(t+σ)² × exp(-2Rt) × exp(2R) = Q(t+σ)² × exp(2R(1-t))

        Returns:
            Q(t)² × exp(2Rt) + Q(t+σ)² × exp(2R(1-t))
        """
        # Evaluate Q(t) and Q_shifted(t) = Q(t+σ)
        t_arr = np.array([t_val])
        Q_t = float(self.Q.eval(t_arr)[0])
        Q_t_shifted = float(self.Q_shifted.eval(t_arr)[0])

        plus_contrib = Q_t ** 2 * np.exp(2 * self.R * t_val)
        minus_contrib = Q_t_shifted ** 2 * np.exp(2 * self.R * (1 - t_val))

        return plus_contrib + minus_contrib


@dataclass(frozen=True)
class TexCombinedMirrorCore:
    """
    TeX-exact combined mirror structure (Run 20A).

    Implements the PRZZ difference quotient → log×integral identity (TeX 1502-1511):

        (N^{αx+βy} - T^{-α-β}N^{-βx-αy}) / (α+β)
        = N^{αx+βy} × log(N^{x+y}T) × ∫₀¹ (N^{x+y}T)^{-s(α+β)} ds

    At α = β = -R/L (PRZZ evaluation point, where L = log T):
        - N = T^θ, so N^{αx+βy} = exp(-Rθ(x+y))
        - log(N^{x+y}T) = L × (1 + θ(x+y))
        - (N^{x+y}T)^{-s(α+β)} = exp(2sR(1 + θ(x+y)))

    Combined (absorbing asymptotic L factor):
        exp(-Rθ(x+y)) × (1 + θ(x+y)) × ∫₀¹ exp(2sR(1 + θ(x+y))) ds

    Key differences from CombinedMirrorFactor (Run 18):
        - Includes outer exp(-Rθ(x+y)) factor from N^{αx+βy}
        - This class is for the combined structure BEFORE Q operators
        - Q operators should be applied AFTER this object is formed

    Key differences from CombinedI1Integrand (Run 19):
        - Uses correct log×integral structure (not naive plus+minus)
        - Does NOT include Q factors inside

    Attributes:
        R: The PRZZ R parameter (shift in critical line)
        theta: The θ = log N / log T parameter (typically 4/7)
        n_quad_s: Number of quadrature points for s-integral (default 20)

    Gate test (scalar limit at x=y=0):
        = 1 × 1 × ∫₀¹ exp(2sR) ds
        = (exp(2R) - 1) / (2R)
    """
    R: float
    theta: float
    n_quad_s: int = 20

    def evaluate(
        self,
        U: np.ndarray,
        T: np.ndarray,
        ctx: SeriesContext
    ) -> TruncatedSeries:
        """
        Evaluate the TeX combined mirror structure as a TruncatedSeries.

        Computes:
            exp(-Rθ(x+y)) × (1 + θ(x+y)) × ∫₀¹ exp(2sR(1 + θ(x+y))) ds

        The s-integral is evaluated via Gauss-Legendre quadrature, keeping
        formal variables (x, y) active throughout.

        Args:
            U: Grid array for u variable (not used but passed for consistency)
            T: Grid array for t variable (not used but passed for consistency)
            ctx: SeriesContext for variable names (must contain "x" and "y")

        Returns:
            TruncatedSeries with combined structure, formal vars active
        """
        from src.quadrature import gauss_legendre_01

        s_nodes, s_weights = gauss_legendre_01(self.n_quad_s)

        # --- Part 1: Outer exponential factor exp(-Rθ(x+y)) ---
        # This is N^{αx+βy} at α=β=-R/L
        outer_base = np.zeros_like(U)  # exp(0) = 1 at x=y=0
        outer_lin = {
            "x": -self.R * self.theta * np.ones_like(U),
            "y": -self.R * self.theta * np.ones_like(U),
        }
        outer_exp = compose_exp_on_affine(1.0, outer_base, outer_lin, ctx.var_names)

        # --- Part 2: Log factor (1 + θ(x+y)) ---
        # The L = log T is asymptotic and absorbed elsewhere
        log_factor = ctx.scalar_series(np.ones_like(U))
        log_factor = log_factor + ctx.variable_series("x") * self.theta
        log_factor = log_factor + ctx.variable_series("y") * self.theta

        # --- Part 3: s-integral ∫₀¹ exp(2sR(1 + θ(x+y))) ds ---
        s_integral = ctx.zero_series()
        for s_idx, s in enumerate(s_nodes):
            # Exponent: 2Rs(1 + θ(x+y)) = 2Rs + 2Rsθx + 2Rsθy
            s_base = 2 * self.R * s * np.ones_like(U)
            s_lin = {
                "x": 2 * self.R * s * self.theta * np.ones_like(U),
                "y": 2 * self.R * s * self.theta * np.ones_like(U),
            }
            s_exp = compose_exp_on_affine(1.0, s_base, s_lin, ctx.var_names)
            s_integral = s_integral + s_exp * s_weights[s_idx]

        # --- Combine all three parts ---
        return outer_exp * log_factor * s_integral

    def scalar_limit(self) -> float:
        """
        Compute the scalar limit (x=y=0) analytically.

        At x=y=0:
            outer_exp = exp(0) = 1
            log_factor = 1
            s_integral = ∫₀¹ exp(2sR) ds = (exp(2R) - 1) / (2R)

        Returns:
            (exp(2R) - 1) / (2R)
        """
        return (np.exp(2 * self.R) - 1) / (2 * self.R)

    def difference_quotient_test(
        self,
        x_val: float,
        y_val: float,
        L: float = 10.0
    ) -> Tuple[float, float]:
        """
        Compare against direct difference quotient for gate testing.

        At α = β = -R/L, computes:
            LHS: This combined structure evaluated at (x, y)
            RHS: (N^{αx+βy} - T^{-α-β}N^{-βx-αy}) / (α+β)

        The RHS is computed directly (with finite L) for validation.

        Args:
            x_val: Test x value (should be small, e.g. 0.01)
            y_val: Test y value (should be small, e.g. 0.01)
            L: log T value for direct computation (larger = more asymptotic)

        Returns:
            Tuple of (combined_value, direct_quotient_value)
        """
        # Parameters
        T_val = np.exp(L)
        N_val = T_val ** self.theta
        alpha = -self.R / L
        beta = alpha  # α = β = -R/L

        # Direct difference quotient:
        # (N^{αx+βy} - T^{-α-β} × N^{-βx-αy}) / (α+β)
        term1 = N_val ** (alpha * x_val + beta * y_val)
        term2 = T_val ** (-alpha - beta) * N_val ** (-beta * x_val - alpha * y_val)
        denominator = alpha + beta
        direct_quotient = (term1 - term2) / denominator

        # Combined structure (without L factor, which we absorb):
        # exp(-Rθ(x+y)) × (1 + θ(x+y)) × ∫₀¹ exp(2sR(1 + θ(x+y))) ds
        from src.quadrature import gauss_legendre_01
        s_nodes, s_weights = gauss_legendre_01(self.n_quad_s)

        outer = np.exp(-self.R * self.theta * (x_val + y_val))
        log_term = 1 + self.theta * (x_val + y_val)

        s_integral = 0.0
        for s, w in zip(s_nodes, s_weights):
            s_integral += w * np.exp(2 * s * self.R * (1 + self.theta * (x_val + y_val)))

        combined = outer * log_term * s_integral

        # Scale by L to match the log(N^{x+y}T) = L(1+θ(x+y)) factor
        combined_scaled = combined * L

        return combined_scaled, direct_quotient


@dataclass
class Term:
    """
    Complete specification of one integral term.

    Each Term represents a derivative-at-0 of a multi-dimensional integral
    with polynomial, exponential, and algebraic factors.

    Attributes:
        name: Identifier (e.g., "I1_11", "I2_12")
        pair: Tuple (ℓ₁, ℓ₂) identifying the pair type
        przz_reference: Reference to PRZZ paper section
        vars: Formal variables in canonical order
        deriv_orders: Dict mapping var -> derivative order (all 1 for d=1)
        domain: Integration domain description
        numeric_prefactor: Constant multiplier
        algebraic_prefactor: Optional AffineExpr prefactor (e.g., (θS+1)/θ)
        poly_prefactors: List of grid functions (e.g., (1-u)^2)
        poly_factors: List of PolyFactor objects
        exp_factors: List of ExpFactor objects

    Invariants (enforced in __post_init__):
        - Phase 0: all derivative orders must be 0 or 1
        - vars and deriv_orders.keys() must match exactly
        - Higher-order derivatives require d>1 infrastructure not yet implemented

    Example:
        >>> term = Term(
        ...     name="I1_11",
        ...     pair=(1, 1),
        ...     przz_reference="Section 6.2.1, I₁",
        ...     vars=("x1", "y1"),
        ...     deriv_orders={"x1": 1, "y1": 1},
        ...     domain="[0,1]^2",
        ...     numeric_prefactor=1.0,
        ...     algebraic_prefactor=None,
        ...     poly_prefactors=[lambda U, T: (1 - U)**2],
        ...     poly_factors=[PolyFactor("P1", P1_arg), ...],
        ...     exp_factors=[ExpFactor(R, Q_arg_alpha), ...]
        ... )
    """
    # Identification
    name: str
    pair: Tuple[int, int]
    przz_reference: Optional[str]

    # Formal variables
    vars: Tuple[str, ...]
    deriv_orders: Dict[str, int]

    # Integration
    domain: str

    # Prefactors
    numeric_prefactor: float
    algebraic_prefactor: Optional[AffineExpr]
    poly_prefactors: List[GridFunc]

    # Factors to expand and multiply
    poly_factors: List[PolyFactor]
    exp_factors: List[ExpFactor]

    def __post_init__(self):
        """Validate Term invariants after construction."""
        # Validate vars/deriv_orders consistency
        vars_set = set(self.vars)
        deriv_set = set(self.deriv_orders.keys())

        extra = deriv_set - vars_set
        if extra:
            raise ValueError(
                f"deriv_orders contains variables not in vars: {extra}. "
                f"vars={self.vars}, deriv_orders keys={list(self.deriv_orders.keys())}"
            )

        missing = vars_set - deriv_set
        if missing:
            raise ValueError(
                f"vars contains variables not in deriv_orders: {missing}. "
                f"For d=1, every var in vars must have an entry in deriv_orders."
            )

        # Phase 0: Enforce d=1 (all derivative orders must be 0 or 1)
        # Higher-order derivatives (d>1) require infrastructure not yet implemented
        for var, order in self.deriv_orders.items():
            if order > 1:
                raise ValueError(
                    f"Derivative order > 1 not supported in Phase 0 (d=1 only). "
                    f"Got deriv_orders['{var}'] = {order}. "
                    f"Higher-order derivatives require d>1 infrastructure."
                )
            if order < 0:
                raise ValueError(
                    f"Derivative order must be non-negative, "
                    f"got deriv_orders['{var}'] = {order}"
                )

    def total_vars(self) -> int:
        """Number of formal variables."""
        return len(self.vars)

    def target_mask(self) -> int:
        """
        Bitmask for the target derivative (all 1's for d=1).

        For n variables, returns 2^n - 1 (all bits set).
        This is the mask to extract the coefficient of x₁x₂...xₙ.
        """
        return (1 << len(self.vars)) - 1

    def deriv_tuple(self) -> Tuple[str, ...]:
        """
        Return tuple of variables to differentiate (for series.extract()).

        For d=1, this returns all vars with deriv_orders[var] == 1.
        The order matches self.vars (canonical order).

        Returns:
            Tuple of variable names to pass to series.extract()
        """
        return tuple(v for v in self.vars if self.deriv_orders.get(v, 0) == 1)

    def create_context(self) -> SeriesContext:
        """Create a SeriesContext from this term's variables."""
        return SeriesContext(var_names=self.vars)

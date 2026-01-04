"""
Fejer band-limited window kernel for localized moments.

The Fejer kernel is the unique L1-minimizing band-limited window:
- Compact support in frequency: w_hat(xi) = 0 for |xi| > Delta
- Triangle profile: w_hat(xi) = (1 - |xi|/Delta) * 1_{|xi| <= Delta}
- Time domain: sinc-squared with t^{-2} decay

Key property: w(t) >= 0 for all t (positivity preserving).

Mathematical definitions:
    w_hat(xi) = (1 - |xi|/Delta) * 1_{|xi| <= Delta}   [frequency domain, triangle]
    w(t) = (Delta / 2*pi) * (sin(Delta*t/2) / (Delta*t/2))^2   [time domain, sinc^2]

The first zero of w(t) occurs at t = 2*pi/Delta.
"""

from dataclasses import dataclass
from typing import Union
import numpy as np


ArrayLike = Union[float, np.ndarray]


@dataclass
class FejerKernel:
    """Fejer band-limited window kernel.

    Attributes:
        Delta: Bandwidth parameter (frequency support is [-Delta, Delta])
    """
    Delta: float

    @property
    def first_zero(self) -> float:
        """First zero of w(t) occurs at t = 2*pi/Delta."""
        return 2 * np.pi / self.Delta

    @classmethod
    def from_first_zero(cls, L: float) -> "FejerKernel":
        """Create kernel with first zero at t = L.

        Args:
            L: Desired location of first zero

        Returns:
            FejerKernel with Delta = 2*pi/L
        """
        if L <= 0:
            raise ValueError("First zero location L must be positive")
        return cls(Delta=2 * np.pi / L)

    def w_time(self, t: ArrayLike) -> np.ndarray:
        """Evaluate window in time domain.

        w_Delta(t) = (Delta / 2*pi) * sinc^2(Delta*t / 2*pi)

        where sinc(x) = sin(pi*x) / (pi*x), with sinc(0) = 1.

        Uses numpy's normalized sinc: np.sinc(x) = sin(pi*x)/(pi*x)

        Args:
            t: Time points (scalar or array)

        Returns:
            Window values w_Delta(t)
        """
        t = np.asarray(t, dtype=np.float64)
        # np.sinc computes sin(pi*x)/(pi*x), so need Delta*t/(2*pi)
        # sin(Delta*t/2) / (Delta*t/2) = sinc(Delta*t/(2*pi))
        arg = self.Delta * t / (2 * np.pi)
        sinc_vals = np.sinc(arg)  # = sin(Delta*t/2) / (Delta*t/2) when t != 0
        return (self.Delta / (2 * np.pi)) * sinc_vals**2

    def w_freq(self, xi: ArrayLike) -> np.ndarray:
        """Evaluate window in frequency domain.

        w_hat(xi) = (1 - |xi|/Delta) * 1_{|xi| <= Delta}

        This is a triangle (tent) function.

        Args:
            xi: Frequency points (scalar or array)

        Returns:
            Window values w_hat(xi)
        """
        xi = np.asarray(xi, dtype=np.float64)
        abs_xi = np.abs(xi)
        return np.maximum(0.0, 1.0 - abs_xi / self.Delta)

    def effective_support(self, tolerance: float = 1e-4) -> float:
        """Approximate time-domain support where w(t) > tolerance * w(0).

        Since w(t) ~ t^{-2} for large |t|, this gives a finite truncation.

        Args:
            tolerance: Relative threshold for effective support

        Returns:
            Value T such that w(T)/w(0) approx tolerance
        """
        # w(0) = Delta / (2*pi)
        # w(t) ~ Delta / (2*pi) * (2*pi / (Delta*t))^2 = 2*pi / (Delta * t^2)
        # Ratio: w(t)/w(0) ~ 4*pi^2 / (Delta^2 * t^2)
        # Solve for w(t)/w(0) = tolerance: t = 2*pi / (Delta * sqrt(tolerance))
        return 2 * np.pi / (self.Delta * np.sqrt(tolerance))

    def required_halfwidth(self, eps_mass: float = 1e-3) -> float:
        """Compute halfwidth U such that tail mass is bounded by eps_mass.

        The Fejér kernel has tail mass:
            ∫_{|t|>U} w(t) dt ≈ 1/(π·Δ·U)

        So to achieve tail mass < eps_mass, we need:
            U > 1/(π·Δ·eps_mass)

        Args:
            eps_mass: Target tail mass (fraction of total integral that's dropped)

        Returns:
            Halfwidth U in absolute time units

        Example:
            With Delta=0.5 and eps_mass=0.01 (1% tail):
            U = 1/(π·0.5·0.01) ≈ 63.66

            With Delta=1.0 and eps_mass=0.001 (0.1% tail):
            U = 1/(π·1.0·0.001) ≈ 318.3
        """
        if eps_mass <= 0:
            raise ValueError("eps_mass must be positive")
        return 1.0 / (np.pi * self.Delta * eps_mass)

    def required_halfwidth_in_zeros(self, eps_mass: float = 1e-3) -> float:
        """Compute halfwidth in units of first_zero for given tail mass.

        This is convenient for setting n_halfwidth in LocalMomentConfig.

        Args:
            eps_mass: Target tail mass

        Returns:
            n_halfwidth = U / first_zero
        """
        U = self.required_halfwidth(eps_mass)
        return U / self.first_zero


def delta_from_first_zero(L: float) -> float:
    """Compute Delta such that first zero is at t = L.

    Args:
        L: Desired first zero location

    Returns:
        Bandwidth Delta = 2*pi/L
    """
    if L <= 0:
        raise ValueError("First zero location L must be positive")
    return 2 * np.pi / L

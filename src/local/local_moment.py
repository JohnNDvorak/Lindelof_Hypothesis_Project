"""
Localized moment computation with Fejer window.

Computes:
    M(T, Delta, sigma) = integral_{-infty}^{infty} w_Delta(t - T) |D(sigma + it)|^2 dt

Using:
- Numerical quadrature on finite grid
- Fejer window decay (t^{-2}) for truncation
- Consistency check via ratio-domain sum
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from src.local.fejer import FejerKernel
from src.local.dirichlet_poly import evaluate_dirichlet_poly


@dataclass
class LocalMomentConfig:
    """Configuration for localized moment computation.

    Attributes:
        T: Center of localization window
        Delta: Bandwidth parameter
        sigma: Real part of s (typically 0.5)
        n_halfwidth: Number of first-zero widths for integration
        n_points_per_zero: Quadrature points per first-zero width
    """
    T: float
    Delta: float
    sigma: float = 0.5
    n_halfwidth: float = 4.0  # truncate at +/- n_halfwidth * first_zero
    n_points_per_zero: int = 20  # points per 2*pi/Delta interval


@dataclass
class RatioMomentDecomposition:
    """Decomposition of ratio-domain moment into diagonal and off-diagonal parts.

    The total moment M = M_diag + M_off where:
    - M_diag = Σ_n |a_n|² n^{-2σ}  (diagonal terms, always positive)
    - M_off = M - M_diag  (off-diagonal terms, oscillatory in T)

    Attributes:
        total: Total moment (real part)
        diagonal: Diagonal contribution (always positive)
        off_diagonal: Off-diagonal contribution (can be negative)
        off_over_diag: Ratio off_diagonal / diagonal (measures relative contribution)
    """
    total: float
    diagonal: float
    off_diagonal: float
    off_over_diag: float


@dataclass
class LocalMomentResult:
    """Result of localized moment computation.

    Attributes:
        moment: The computed localized moment
        T: Center time
        Delta: Bandwidth
        sigma: Real part
        t_grid: Time grid used
        weights: Fejer weights at grid points
        D_squared: |D(sigma + it)|^2 values
        ratio_domain_moment: (Optional) consistency check value
    """
    moment: float
    T: float
    Delta: float
    sigma: float
    t_grid: np.ndarray
    weights: np.ndarray
    D_squared: np.ndarray
    ratio_domain_moment: Optional[float] = None


def compute_local_moment(
    coeffs: np.ndarray,
    config: LocalMomentConfig,
) -> LocalMomentResult:
    """Compute localized moment via time-domain quadrature.

    M(T) = integral w_Delta(t - T) |D(sigma + it)|^2 dt
         approx sum_k w_Delta(t_k - T) |D(sigma + it_k)|^2 * dt

    Args:
        coeffs: Dirichlet polynomial coefficients
        config: Localization configuration

    Returns:
        LocalMomentResult
    """
    kernel = FejerKernel(config.Delta)
    first_zero = kernel.first_zero

    # Determine grid extent
    half_extent = config.n_halfwidth * first_zero
    n_points = int(2 * config.n_halfwidth * config.n_points_per_zero) + 1

    # Symmetric grid around T
    t_grid = np.linspace(config.T - half_extent, config.T + half_extent, n_points)
    dt = t_grid[1] - t_grid[0]

    # Evaluate Fejer window
    weights = kernel.w_time(t_grid - config.T)

    # Evaluate Dirichlet polynomial
    D_result = evaluate_dirichlet_poly(coeffs, t_grid=t_grid, sigma=config.sigma)

    # Trapezoidal quadrature (adjust endpoints)
    trap_weights = np.ones(n_points)
    trap_weights[0] = 0.5
    trap_weights[-1] = 0.5

    # Compute moment
    integrand = weights * D_result.abs_squared
    moment = np.sum(integrand * trap_weights) * dt

    return LocalMomentResult(
        moment=moment,
        T=config.T,
        Delta=config.Delta,
        sigma=config.sigma,
        t_grid=t_grid,
        weights=weights,
        D_squared=D_result.abs_squared,
    )


def compute_ratio_domain_moment(
    coeffs: np.ndarray,
    config: LocalMomentConfig,
) -> float:
    """Compute localized moment via ratio-domain sum (consistency check).

    M(T) = sum_{n,m <= N} a_n conj(a_m) (nm)^{-sigma} w_hat(log(n/m)) * exp(-iT log(n/m))

    Since w_hat is a triangle:
    w_hat(log(n/m)) = max(0, 1 - |log(n/m)| / Delta)

    This is non-zero only when |log(n/m)| <= Delta, i.e., exp(-Delta) <= n/m <= exp(Delta).

    Args:
        coeffs: Dirichlet polynomial coefficients
        config: Localization configuration

    Returns:
        Ratio-domain moment value
    """
    kernel = FejerKernel(config.Delta)
    N = len(coeffs) - 1
    sigma = config.sigma
    T = config.T

    a = coeffs[1:N+1]  # shape (N,)
    n = np.arange(1, N + 1, dtype=np.float64)

    # Precompute n^{-sigma} and log(n)
    n_pow = n ** (-sigma)
    log_n = np.log(n)

    total = 0.0 + 0.0j

    # Double loop (can be optimized but O(N^2) is acceptable for small N)
    for i in range(N):
        for j in range(N):
            log_ratio = log_n[i] - log_n[j]
            w_hat = kernel.w_freq(log_ratio)
            if w_hat > 0:
                term = (a[i] * np.conj(a[j]) * n_pow[i] * n_pow[j]
                        * w_hat * np.exp(-1j * T * log_ratio))
                total += term

    return np.real(total)


def compute_ratio_domain_decomposed(
    coeffs: np.ndarray,
    config: LocalMomentConfig,
) -> RatioMomentDecomposition:
    """Compute ratio-domain moment with diagonal/off-diagonal decomposition.

    Separates the total moment into:
    - Diagonal: Σ_n |a_n|² n^{-2σ} · ŵ(0) = Σ_n |a_n|² n^{-2σ}  (since ŵ(0) = 1)
    - Off-diagonal: Everything else

    This decomposition reveals whether interference effects are hiding
    in the off-diagonal terms, masked by diagonal drift.

    Args:
        coeffs: Dirichlet polynomial coefficients
        config: Localization configuration

    Returns:
        RatioMomentDecomposition with total, diagonal, off_diagonal, and ratio
    """
    kernel = FejerKernel(config.Delta)
    N = len(coeffs) - 1
    sigma = config.sigma
    T = config.T

    a = coeffs[1:N+1]  # shape (N,)
    n = np.arange(1, N + 1, dtype=np.float64)

    # Precompute n^{-sigma} and log(n)
    n_pow = n ** (-sigma)
    log_n = np.log(n)

    # Diagonal: Σ |a_n|² n^{-2σ}
    # Note: ŵ(0) = 1 and exp(-iT·0) = 1, so diagonal is simple
    diagonal = np.sum(np.abs(a)**2 * n_pow**2)

    # Off-diagonal: sum over i != j
    off_diagonal = 0.0 + 0.0j

    for i in range(N):
        for j in range(N):
            if i == j:
                continue  # Skip diagonal
            log_ratio = log_n[i] - log_n[j]
            w_hat = kernel.w_freq(log_ratio)
            if w_hat > 0:
                term = (a[i] * np.conj(a[j]) * n_pow[i] * n_pow[j]
                        * w_hat * np.exp(-1j * T * log_ratio))
                off_diagonal += term

    off_diagonal_real = np.real(off_diagonal)
    total = diagonal + off_diagonal_real

    # Compute ratio (avoid div by zero)
    if abs(diagonal) > 1e-15:
        off_over_diag = off_diagonal_real / diagonal
    else:
        off_over_diag = 0.0

    return RatioMomentDecomposition(
        total=total,
        diagonal=diagonal,
        off_diagonal=off_diagonal_real,
        off_over_diag=off_over_diag,
    )


def verify_moment_consistency(
    coeffs: np.ndarray,
    config: LocalMomentConfig,
    rtol: float = 1e-3,
) -> Tuple[float, float, bool]:
    """Verify time-domain and ratio-domain moments agree.

    Args:
        coeffs: Dirichlet polynomial coefficients
        config: Localization configuration
        rtol: Relative tolerance

    Returns:
        (time_domain, ratio_domain, passed)
    """
    time_result = compute_local_moment(coeffs, config)
    ratio_result = compute_ratio_domain_moment(coeffs, config)

    if abs(ratio_result) < 1e-15:
        # Avoid division by zero
        passed = abs(time_result.moment) < 1e-15
        return time_result.moment, ratio_result, passed

    rel_error = abs(time_result.moment - ratio_result) / abs(ratio_result)
    passed = rel_error < rtol

    return time_result.moment, ratio_result, passed

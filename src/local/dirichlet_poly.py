"""
Dirichlet polynomial evaluator for localized moments.

D(s) = sum_{n <= N} a_n n^{-s}

For s = sigma + it on a uniform time grid t_k = T + k*dt:
- Mode A: Brute vectorized (good for small N)
- Mode B: Incremental phase update for large uniform grids
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class DirichletPolyResult:
    """Result of Dirichlet polynomial evaluation.

    Attributes:
        t_grid: Time points
        values: D(sigma + it) values (complex)
        abs_squared: |D(sigma + it)|^2 values
    """
    t_grid: np.ndarray
    values: np.ndarray  # complex
    abs_squared: np.ndarray  # real


def evaluate_brute(
    coeffs: np.ndarray,
    t_grid: np.ndarray,
    sigma: float = 0.5,
) -> DirichletPolyResult:
    """Evaluate D(s) = sum_{n=1}^N a_n n^{-s} via brute-force vectorization.

    Args:
        coeffs: Array of length N+1 with coeffs[n] = a_n (coeffs[0] unused)
        t_grid: 1D array of time points
        sigma: Real part of s

    Returns:
        DirichletPolyResult
    """
    N = len(coeffs) - 1
    t_grid = np.asarray(t_grid, dtype=np.float64)
    n = np.arange(1, N + 1, dtype=np.float64)

    # n^{-s} = n^{-sigma} * n^{-it} = n^{-sigma} * exp(-it * log(n))
    n_pow_neg_sigma = n ** (-sigma)  # shape (N,)
    log_n = np.log(n)  # shape (N,)

    # Outer product for phase: exp(-i * t_grid[:, None] * log_n[None, :])
    # Shape: (len(t_grid), N)
    phases = np.exp(-1j * np.outer(t_grid, log_n))

    # D(s) = sum_n a_n * n^{-sigma} * exp(-it log n)
    # coeffs[1:N+1] has shape (N,)
    weighted = coeffs[1:N+1] * n_pow_neg_sigma  # shape (N,)
    values = np.dot(phases, weighted)  # shape (len(t_grid),)

    return DirichletPolyResult(
        t_grid=t_grid,
        values=values,
        abs_squared=np.abs(values) ** 2,
    )


def evaluate_incremental(
    coeffs: np.ndarray,
    T: float,
    dt: float,
    K: int,
    sigma: float = 0.5,
) -> DirichletPolyResult:
    """Evaluate D(s) on uniform grid t_k = T + k*dt using incremental phases.

    For large N and many grid points, this avoids recomputing exp(-it log n)
    from scratch at each point.

    z_n(t_k) = n^{-sigma} * exp(-i * t_k * log n)
    z_n(t_{k+1}) = z_n(t_k) * r_n  where r_n = exp(-i * dt * log n)

    Args:
        coeffs: Array of length N+1 with coeffs[n] = a_n
        T: Starting time (center of grid)
        dt: Time step
        K: Number of points on each side of T (total 2K+1 points)
        sigma: Real part of s

    Returns:
        DirichletPolyResult with t_grid = [T - K*dt, ..., T, ..., T + K*dt]
    """
    N = len(coeffs) - 1
    n = np.arange(1, N + 1, dtype=np.float64)

    n_pow_neg_sigma = n ** (-sigma)
    log_n = np.log(n)

    # Multipliers for phase update
    r_n = np.exp(-1j * dt * log_n)  # shape (N,)

    # Initialize at T
    z_n = n_pow_neg_sigma * np.exp(-1j * T * log_n)  # shape (N,)

    # Weighted coefficients
    a_n = coeffs[1:N+1]  # shape (N,)

    # Build grid from T outward
    t_grid = np.arange(-K, K + 1) * dt + T  # shape (2K+1,)
    values = np.zeros(2 * K + 1, dtype=np.complex128)

    # Compute at T first
    center_idx = K
    values[center_idx] = np.dot(a_n, z_n)

    # Forward sweep: T+dt, T+2*dt, ...
    z_forward = z_n.copy()
    for k in range(1, K + 1):
        z_forward *= r_n
        values[center_idx + k] = np.dot(a_n, z_forward)

    # Backward sweep: T-dt, T-2*dt, ...
    r_n_inv = 1.0 / r_n  # = exp(+i * dt * log n)
    z_backward = z_n.copy()
    for k in range(1, K + 1):
        z_backward *= r_n_inv
        values[center_idx - k] = np.dot(a_n, z_backward)

    return DirichletPolyResult(
        t_grid=t_grid,
        values=values,
        abs_squared=np.abs(values) ** 2,
    )


def evaluate_dirichlet_poly(
    coeffs: np.ndarray,
    t_grid: Optional[np.ndarray] = None,
    T: Optional[float] = None,
    dt: Optional[float] = None,
    K: Optional[int] = None,
    sigma: float = 0.5,
    mode: str = 'auto',
) -> DirichletPolyResult:
    """Unified interface for Dirichlet polynomial evaluation.

    Args:
        coeffs: Coefficient array
        t_grid: Explicit time grid (for brute mode)
        T, dt, K: Uniform grid parameters (for incremental mode)
        sigma: Real part of s
        mode: 'brute', 'incremental', or 'auto'

    Returns:
        DirichletPolyResult
    """
    N = len(coeffs) - 1

    if mode == 'auto':
        # Use incremental for large N with uniform grid
        if T is not None and dt is not None and K is not None:
            if N > 1000:
                mode = 'incremental'
            else:
                mode = 'brute'
                t_grid = np.arange(-K, K + 1) * dt + T
        else:
            mode = 'brute'

    if mode == 'incremental':
        if T is None or dt is None or K is None:
            raise ValueError("Incremental mode requires T, dt, K")
        return evaluate_incremental(coeffs, T, dt, K, sigma)
    else:
        if t_grid is None:
            if T is not None and dt is not None and K is not None:
                t_grid = np.arange(-K, K + 1) * dt + T
            else:
                raise ValueError("Brute mode requires t_grid or (T, dt, K)")
        return evaluate_brute(coeffs, t_grid, sigma)

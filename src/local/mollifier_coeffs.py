"""
Mollifier coefficient generators for psi_1, psi_2, psi_3.

Each mollifier psi_k(s) = sum_{n <= N} a_n^(k) n^{-s} where:
- psi_1: a_n^(1) = mu(n) * P_1(u_n)
- psi_2: a_n^(2) = (mu * Lambda)(n) / log(N) * P_2(u_n)
- psi_3: a_n^(3) = (mu * Lambda * Lambda)(n) / (log(N))^2 * P_3(u_n)

with u_n = log(N/n) / log(N) in [0, 1] for n <= N.

The polynomials P_1, P_2, P_3 are from the PRZZ infrastructure,
loaded from data/optimal_polynomials.json (c=1 saturation coefficients).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import json
import numpy as np

from src.local.sieve import SieveArrays, compute_sieve_arrays
from src.polynomials import P1Polynomial, PellPolynomial


@dataclass
class MollifierCoeffs:
    """Coefficient arrays for mollifiers.

    Attributes:
        N: Mollifier length
        log_N: log(N)
        u: Array of u_n = log(N/n) / log(N) for n = 1, ..., N
        a1: Coefficients for psi_1 (or None if not computed)
        a2: Coefficients for psi_2 (or None if not computed)
        a3: Coefficients for psi_3 (or None if not computed)
    """
    N: int
    log_N: float
    u: np.ndarray  # shape (N+1,), u[0] unused
    a1: Optional[np.ndarray] = None  # shape (N+1,)
    a2: Optional[np.ndarray] = None
    a3: Optional[np.ndarray] = None


def compute_u_array(N: int) -> np.ndarray:
    """Compute u_n = log(N/n) / log(N) for n = 1, ..., N.

    u_n lies in [0, 1]:
    - u_1 = log(N/1) / log(N) = 1
    - u_N = log(N/N) / log(N) = 0

    Args:
        N: Mollifier length

    Returns:
        Array of length N+1 with u[0] = 0 (unused), u[n] = log(N/n)/log(N)
    """
    log_N = np.log(float(N))
    u = np.zeros(N + 1, dtype=np.float64)
    n = np.arange(1, N + 1, dtype=np.float64)
    u[1:] = np.log(N / n) / log_N
    return u


def load_optimal_polynomials() -> Tuple[P1Polynomial, PellPolynomial, PellPolynomial]:
    """Load optimal polynomials from data/optimal_polynomials.json.

    These are the c=1 saturation coefficients that maximize destructive interference.

    Returns:
        (P1, P2, P3) polynomial objects
    """
    # Find the data file relative to this module
    module_dir = Path(__file__).parent.parent.parent
    data_file = module_dir / "data" / "optimal_polynomials.json"

    with open(data_file, 'r') as f:
        data = json.load(f)

    P1 = P1Polynomial(tilde_coeffs=data["P1_tilde"])
    P2 = PellPolynomial(tilde_coeffs=data["P2_tilde"])
    P3 = PellPolynomial(tilde_coeffs=data["P3_tilde"])

    return P1, P2, P3


def load_przz_polynomials() -> Tuple[P1Polynomial, PellPolynomial, PellPolynomial]:
    """Load PRZZ baseline polynomials from data/przz_parameters.json.

    These are the original PRZZ coefficients for comparison.

    Returns:
        (P1, P2, P3) polynomial objects
    """
    # Find the data file relative to this module
    module_dir = Path(__file__).parent.parent.parent
    data_file = module_dir / "data" / "przz_parameters.json"

    with open(data_file, 'r') as f:
        data = json.load(f)

    P1 = P1Polynomial(tilde_coeffs=data["polynomials"]["P1"]["tilde_coeffs"])
    P2 = PellPolynomial(tilde_coeffs=data["polynomials"]["P2"]["tilde_coeffs"])
    P3 = PellPolynomial(tilde_coeffs=data["polynomials"]["P3"]["tilde_coeffs"])

    return P1, P2, P3


def compute_psi1_coeffs(
    N: int,
    sieve: SieveArrays,
    P1: P1Polynomial,
) -> np.ndarray:
    """Compute psi_1 coefficients: a_n^(1) = mu(n) * P_1(u_n).

    Args:
        N: Mollifier length
        sieve: Precomputed sieve arrays
        P1: P1 polynomial

    Returns:
        Array of length N+1 with a1[n] = mu(n) * P1(u_n)
    """
    u = compute_u_array(N)
    P1_vals = P1.eval(u[1:])  # Evaluate P1 at u_1, ..., u_N

    a1 = np.zeros(N + 1, dtype=np.float64)
    a1[1:] = sieve.mobius[1:N+1].astype(np.float64) * P1_vals
    return a1


def compute_psi2_coeffs(
    N: int,
    sieve: SieveArrays,
    P2: PellPolynomial,
) -> np.ndarray:
    """Compute psi_2 coefficients: a_n^(2) = (mu*Lambda)(n) / log(N) * P_2(u_n).

    Args:
        N: Mollifier length
        sieve: Precomputed sieve arrays
        P2: P2 polynomial

    Returns:
        Array of length N+1 with a2[n] = (mu*Lambda)(n)/log(N) * P2(u_n)
    """
    log_N = np.log(float(N))
    u = compute_u_array(N)
    P2_vals = P2.eval(u[1:])

    a2 = np.zeros(N + 1, dtype=np.float64)
    a2[1:] = (sieve.mu_star_Lambda[1:N+1] / log_N) * P2_vals
    return a2


def compute_psi3_coeffs(
    N: int,
    sieve: SieveArrays,
    P3: PellPolynomial,
) -> np.ndarray:
    """Compute psi_3 coefficients: a_n^(3) = (mu*Lambda*Lambda)(n) / (log N)^2 * P_3(u_n).

    Args:
        N: Mollifier length
        sieve: Precomputed sieve arrays (must have mu_star_Lambda_Lambda)
        P3: P3 polynomial

    Returns:
        Array of length N+1 with a3[n] = (mu*Lambda*Lambda)(n)/(log N)^2 * P3(u_n)

    Raises:
        ValueError: If sieve.mu_star_Lambda_Lambda is None
    """
    if sieve.mu_star_Lambda_Lambda is None:
        raise ValueError("Sieve arrays must include mu_star_Lambda_Lambda for psi_3")

    log_N = np.log(float(N))
    u = compute_u_array(N)
    P3_vals = P3.eval(u[1:])

    a3 = np.zeros(N + 1, dtype=np.float64)
    a3[1:] = (sieve.mu_star_Lambda_Lambda[1:N+1] / (log_N ** 2)) * P3_vals
    return a3


def compute_mollifier_coeffs(
    N: int,
    which: Tuple[bool, bool, bool] = (True, False, False),
    use_optimal: bool = True,
    P1: Optional[P1Polynomial] = None,
    P2: Optional[PellPolynomial] = None,
    P3: Optional[PellPolynomial] = None,
) -> MollifierCoeffs:
    """Compute mollifier coefficient arrays.

    Args:
        N: Mollifier length
        which: (compute_psi1, compute_psi2, compute_psi3)
        use_optimal: If True, load optimal polynomials (c=1); if False, use PRZZ baseline
        P1, P2, P3: Polynomials (override loading if provided)

    Returns:
        MollifierCoeffs dataclass
    """
    # Load polynomials if not provided
    if P1 is None or P2 is None or P3 is None:
        if use_optimal:
            p1_loaded, p2_loaded, p3_loaded = load_optimal_polynomials()
        else:
            p1_loaded, p2_loaded, p3_loaded = load_przz_polynomials()
        P1 = P1 if P1 is not None else p1_loaded
        P2 = P2 if P2 is not None else p2_loaded
        P3 = P3 if P3 is not None else p3_loaded

    # Compute sieve arrays
    need_psi3 = which[2]
    sieve = compute_sieve_arrays(N, include_psi3=need_psi3)

    # Compute requested coefficients
    a1 = compute_psi1_coeffs(N, sieve, P1) if which[0] else None
    a2 = compute_psi2_coeffs(N, sieve, P2) if which[1] else None
    a3 = compute_psi3_coeffs(N, sieve, P3) if which[2] else None

    return MollifierCoeffs(
        N=N,
        log_N=np.log(float(N)),
        u=compute_u_array(N),
        a1=a1,
        a2=a2,
        a3=a3,
    )

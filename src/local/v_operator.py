"""
V-operator implementation for localized |Vζψ|² moments.

The V-operator applied to zeta gives:
    V[ζ](s) = Σ_{m≤M} Q(log(m)/log(M)) · m^{-s}

This is convolved with the mollifier ψ to get:
    (V[ζ] · ψ)(s) = Σ_k c_k · k^{-s}

where c_k = Σ_{mn=k} b_m · a_n (Dirichlet convolution).
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import json
import numpy as np
from pathlib import Path

from src.polynomials import Polynomial


@dataclass
class VZetaCoeffs:
    """Coefficients of V[ζ] Dirichlet polynomial.

    V[ζ](s) = Σ_{m=1}^{M} b_m · m^{-s}

    where b_m = Q(log(m)/log(M)).

    Attributes:
        M: Length of Dirichlet polynomial
        b: Array of coefficients, shape (M+1,) with b[0] unused
        log_M: log(M) for reference
    """
    M: int
    b: np.ndarray
    log_M: float


def compute_vzeta_coeffs(M: int, Q: Polynomial) -> VZetaCoeffs:
    """Compute V[ζ] coefficients: b_m = Q(log(m)/log(M)).

    Args:
        M: Dirichlet polynomial length
        Q: Q polynomial in monomial form (coefficients [q_0, q_1, ...])

    Returns:
        VZetaCoeffs with b[m] = Q(log(m)/log(M)) for m = 1..M
    """
    log_M = np.log(float(M))
    b = np.zeros(M + 1, dtype=np.float64)

    # b[m] = Q(log(m)/log(M))
    # For m = 1: log(1) = 0, so b[1] = Q(0)
    # For m = M: log(M)/log(M) = 1, so b[M] = Q(1)
    m_arr = np.arange(1, M + 1, dtype=np.float64)
    u_m = np.log(m_arr) / log_M  # Arguments in [0, 1]
    b[1:] = Q.eval(u_m)

    return VZetaCoeffs(M=M, b=b, log_M=log_M)


def dirichlet_convolve(b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Compute Dirichlet convolution c_k = Σ_{mn=k} b_m · a_n.

    The Dirichlet convolution of two arithmetic functions:
        c_k = Σ_{d|k} b_d · a_{k/d}

    For Dirichlet polynomials with finite support:
        c_k = Σ_{mn=k, m≤M, n≤N} b_m · a_n

    Args:
        b: Coefficients b[1:M], shape (M+1,) with b[0] = 0
        a: Coefficients a[1:N], shape (N+1,) with a[0] = 0

    Returns:
        c: Array of length L+1 where L = M*N, with c[k] = Σ_{mn=k} b_m·a_n
    """
    M = len(b) - 1
    N = len(a) - 1
    L = M * N

    c = np.zeros(L + 1, dtype=np.float64)

    # Direct O(M*N) convolution
    # For each m with nonzero b[m], add b[m]*a[n] to c[m*n]
    for m in range(1, M + 1):
        bm = b[m]
        if abs(bm) < 1e-50:
            continue
        for n in range(1, N + 1):
            an = a[n]
            if abs(an) < 1e-50:
                continue
            k = m * n
            if k <= L:
                c[k] += bm * an

    return c


def compute_vzeta_psi_coeffs(
    vzeta_coeffs: VZetaCoeffs,
    psi_coeffs: np.ndarray,
) -> np.ndarray:
    """Compute (V[ζ] · ψ) coefficients via Dirichlet convolution.

    Args:
        vzeta_coeffs: V[ζ] coefficients from compute_vzeta_coeffs()
        psi_coeffs: Mollifier coefficients (ψ = ψ₁ + ψ₂ + ψ₃)

    Returns:
        Convolved coefficients c_k = Σ_{mn=k} b_m · a_n
    """
    return dirichlet_convolve(vzeta_coeffs.b, psi_coeffs)


def load_optimal_Q() -> Tuple[Polynomial, float]:
    """Load optimal Q polynomial from optimized_polynomials_c1.json.

    Returns:
        (Q_polynomial, R) where Q is in monomial form and R is the optimal R value
    """
    data_dir = Path(__file__).parent.parent.parent / "data"
    json_path = data_dir / "optimized_polynomials_c1.json"

    with open(json_path) as f:
        data = json.load(f)

    # Use kappa config (the main benchmark)
    kappa_config = data["kappa_config"]
    Q_monomial = kappa_config["Q_monomial"]
    R = kappa_config["R"]

    # Create Polynomial in monomial form
    Q = Polynomial(coeffs=np.array(Q_monomial, dtype=np.float64))

    return Q, R


def load_przz_Q() -> Tuple[Polynomial, float]:
    """Load PRZZ baseline Q polynomial.

    Returns:
        (Q_polynomial, R) where Q is in monomial form and R is the PRZZ R value
    """
    from src.polynomials import load_przz_polynomials

    # Load PRZZ polynomials
    _, _, _, Q = load_przz_polynomials(enforce_Q0=False)

    # Convert to monomial form
    Q_mono = Q.to_monomial()

    # Enforce PRZZ constraint: Q(0) = 1
    # The first coefficient in monomial form is Q(0)
    coeffs = Q_mono.coeffs.copy()
    coeffs[0] = 1.0
    Q_mono = Polynomial(coeffs=coeffs)

    # PRZZ R value
    R = 1.3036

    return Q_mono, R


def load_optimal_polynomials_full():
    """Load all optimal polynomials (P1, P2, P3, Q) from optimized_polynomials_c1.json.

    Returns:
        Dictionary with keys:
        - 'P1_tilde': P1 tilde coefficients
        - 'P2_tilde': P2 tilde coefficients
        - 'P3_tilde': P3 tilde coefficients
        - 'Q_monomial': Q coefficients in monomial form
        - 'R': Optimal R value
    """
    data_dir = Path(__file__).parent.parent.parent / "data"
    json_path = data_dir / "optimized_polynomials_c1.json"

    with open(json_path) as f:
        data = json.load(f)

    return {
        'P1_tilde': np.array(data['universal_P1']['P1_tilde'], dtype=np.float64),
        'P2_tilde': np.array(data['kappa_config']['P2_tilde'], dtype=np.float64),
        'P3_tilde': np.array(data['kappa_config']['P3_tilde'], dtype=np.float64),
        'Q_monomial': np.array(data['kappa_config']['Q_monomial'], dtype=np.float64),
        'R': data['kappa_config']['R'],
    }

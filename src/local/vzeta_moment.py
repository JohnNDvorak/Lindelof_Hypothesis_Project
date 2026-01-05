"""
Localized |V[ζ] · ψ|² moment computation.

This module computes the localized moment of the PRZZ integrand |Vζψ|²
where:
- V is the differential operator applied to ζ via Q polynomial
- ζ is approximated by a Dirichlet polynomial
- ψ = ψ₁ + ψ₂ + ψ₃ is the full mollifier

Key insight: V[ζ](s) = Σ_m Q(log(m)/log(M)) · m^{-s} is just another
Dirichlet polynomial. The product Vζ · ψ is computed via Dirichlet convolution.

Phase 3 Goal: Measure |Vζψ|² where the c=1 geometry enters at O(1) rather
than being log(N)-suppressed as in |ψ|².
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import json
import numpy as np
from pathlib import Path

from src.polynomials import Polynomial, P1Polynomial, PellPolynomial
from src.local.v_operator import (
    VZetaCoeffs,
    compute_vzeta_coeffs,
    compute_vzeta_psi_coeffs,
    load_optimal_Q,
    load_przz_Q,
    load_optimal_polynomials_full,
)
from src.local.local_moment import (
    LocalMomentConfig,
    LocalMomentResult,
    RatioMomentDecomposition,
    compute_local_moment,
    compute_ratio_domain_decomposed,
)
from src.local.mollifier_coeffs import compute_mollifier_coeffs
from src.local.sieve import compute_sieve_arrays


# PRZZ parameters
PRZZ_R = 1.3036
PRZZ_THETA = 4/7


@dataclass
class VZetaMomentConfig:
    """Configuration for V[ζ]·ψ moment computation.

    Attributes:
        T: Height parameter (center of window)
        Delta: Bandwidth parameter for Fejér window
        R: Shift parameter (controls Levinson line σ = 1/2 - R/log(T))
        theta: Mollifier exponent (N = T^θ)
        N: Override mollifier length (if None, uses T^θ)
        use_levinson_line: If True, σ = 1/2 - R/log(T); if False, σ = 1/2
        n_halfwidth: Number of Fejér zeros for truncation
        n_points_per_zero: Quadrature resolution
    """
    T: float
    Delta: float
    R: float = 1.14976  # Optimal R for c=1 (PRZZ uses 1.3036)
    theta: float = 4/7
    N: Optional[int] = None
    use_levinson_line: bool = True
    n_halfwidth: float = 4.0
    n_points_per_zero: int = 20

    @property
    def sigma(self) -> float:
        """Evaluation point: σ = 1/2 - R/log(T) on Levinson/Conrey line."""
        if self.use_levinson_line:
            return 0.5 - self.R / np.log(self.T)
        return 0.5

    @property
    def mollifier_length(self) -> int:
        """Compute mollifier length N = T^θ."""
        if self.N is not None:
            return self.N
        return int(self.T ** self.theta)


@dataclass
class VZetaMomentResult:
    """Result of localized |V[ζ]·ψ|² computation.

    Attributes:
        moment: The localized moment value
        config: Configuration used
        local_result: Full LocalMomentResult from time-domain computation
        vzeta_coeffs: V[ζ] coefficients
        psi_coeffs: Mollifier coefficients (before convolution)
        convolved_coeffs: (V[ζ] · ψ) coefficients after convolution
        ratio_decomposition: Optional diagonal/off-diagonal decomposition
        diagnostics: Additional diagnostic info
    """
    moment: float
    config: VZetaMomentConfig
    local_result: LocalMomentResult
    vzeta_coeffs: VZetaCoeffs
    psi_coeffs: np.ndarray
    convolved_coeffs: np.ndarray
    ratio_decomposition: Optional[RatioMomentDecomposition] = None
    diagnostics: Optional[Dict[str, Any]] = None


def load_c1_optimal_mollifier_polynomials() -> Tuple[P1Polynomial, PellPolynomial, PellPolynomial]:
    """Load c=1 optimal mollifier polynomials from optimized_polynomials_c1.json.

    These are the true c=1 saturation coefficients.

    Returns:
        (P1, P2, P3) polynomial objects
    """
    opt = load_optimal_polynomials_full()

    P1 = P1Polynomial(tilde_coeffs=opt['P1_tilde'])
    P2 = PellPolynomial(tilde_coeffs=opt['P2_tilde'])
    P3 = PellPolynomial(tilde_coeffs=opt['P3_tilde'])

    return P1, P2, P3


def load_przz_mollifier_polynomials() -> Tuple[P1Polynomial, PellPolynomial, PellPolynomial]:
    """Load PRZZ baseline mollifier polynomials.

    Returns:
        (P1, P2, P3) polynomial objects
    """
    from src.polynomials import load_przz_polynomials

    # Load P1, P2, P3 (ignore Q)
    P1, P2, P3, _ = load_przz_polynomials()

    return P1, P2, P3


def compute_full_mollifier_coeffs(
    N: int,
    use_optimal: bool = True,
) -> np.ndarray:
    """Compute full mollifier ψ = ψ₁ + ψ₂ + ψ₃ coefficients.

    Args:
        N: Mollifier length
        use_optimal: If True, use c=1 optimal polynomials; else PRZZ baseline

    Returns:
        Combined coefficient array of shape (N+1,)
    """
    # Load polynomials
    if use_optimal:
        P1, P2, P3 = load_c1_optimal_mollifier_polynomials()
    else:
        P1, P2, P3 = load_przz_mollifier_polynomials()

    # Compute sieve arrays (need psi3 support)
    sieve = compute_sieve_arrays(N, include_psi3=True)

    # Compute individual mollifier coefficients
    coeffs = compute_mollifier_coeffs(
        N=N,
        which=(True, True, True),  # All three ψ components
        use_optimal=False,  # We pass polynomials directly below
        P1=P1,
        P2=P2,
        P3=P3,
    )

    # Combine: ψ = ψ₁ + ψ₂ + ψ₃
    result = np.zeros(N + 1, dtype=np.float64)
    if coeffs.a1 is not None:
        result += coeffs.a1
    if coeffs.a2 is not None:
        result += coeffs.a2
    if coeffs.a3 is not None:
        result += coeffs.a3

    return result


def compute_vzeta_psi_moment(
    config: VZetaMomentConfig,
    use_optimal: bool = True,
    include_decomposition: bool = False,
) -> VZetaMomentResult:
    """Compute localized |V[ζ]·ψ|² moment.

    This is the main entry point for Phase 3.

    Algorithm:
    1. Load Q polynomial (optimal or PRZZ)
    2. Compute V[ζ] coefficients: b_m = Q(log(m)/log(M))
    3. Get ψ = ψ₁ + ψ₂ + ψ₃ coefficients
    4. Dirichlet convolve: c_k = Σ_{mn=k} b_m · a_n
    5. Compute localized moment of |D_c(s)|²

    Args:
        config: VZetaMomentConfig with T, Delta, R, etc.
        use_optimal: Use c=1 optimal polynomials vs PRZZ baseline
        include_decomposition: Include ratio-domain decomposition

    Returns:
        VZetaMomentResult with moment and diagnostics
    """
    N = config.mollifier_length

    # Step 1: Load Q polynomial
    if use_optimal:
        Q, R_from_file = load_optimal_Q()
    else:
        Q, R_from_file = load_przz_Q()

    # Step 2: Compute V[ζ] coefficients
    vzeta_coeffs = compute_vzeta_coeffs(N, Q)

    # Step 3: Get mollifier coefficients
    psi_coeffs = compute_full_mollifier_coeffs(N, use_optimal=use_optimal)

    # Step 4: Dirichlet convolution
    convolved_coeffs = compute_vzeta_psi_coeffs(vzeta_coeffs, psi_coeffs)

    # Step 5: Compute localized moment using existing machinery
    local_config = LocalMomentConfig(
        T=config.T,
        Delta=config.Delta,
        sigma=config.sigma,
        n_halfwidth=config.n_halfwidth,
        n_points_per_zero=config.n_points_per_zero,
    )
    local_result = compute_local_moment(convolved_coeffs, local_config)

    # Step 6: Optional decomposition
    decomposition = None
    if include_decomposition:
        decomposition = compute_ratio_domain_decomposed(convolved_coeffs, local_config)

    # Diagnostics
    diagnostics = {
        'Q_type': 'optimal' if use_optimal else 'PRZZ',
        'R_used': config.R,
        'R_from_file': R_from_file,
        'sigma': config.sigma,
        'N': N,
        'convolution_length': len(convolved_coeffs) - 1,
        'b_1': vzeta_coeffs.b[1],  # Should be Q(0) = 1
        'psi_nonzero': np.count_nonzero(psi_coeffs[1:]),
        'c_nonzero': np.count_nonzero(convolved_coeffs[1:]),
    }

    return VZetaMomentResult(
        moment=local_result.moment,
        config=config,
        local_result=local_result,
        vzeta_coeffs=vzeta_coeffs,
        psi_coeffs=psi_coeffs,
        convolved_coeffs=convolved_coeffs,
        ratio_decomposition=decomposition,
        diagnostics=diagnostics,
    )


def delta_sweep(
    T: float,
    delta_values: np.ndarray,
    use_optimal: bool = True,
    **config_kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sweep Delta values to study localization behavior.

    As Delta → 0 (wide window), the moment should approach the global c value:
    - Optimal: c ≈ 1.0 at R = 1.14976
    - PRZZ: c ≈ 2.137 at R = 1.3036

    Args:
        T: Height parameter
        delta_values: Array of Delta values to sweep
        use_optimal: Use optimal or PRZZ polynomials
        **config_kwargs: Additional VZetaMomentConfig parameters

    Returns:
        (delta_values, moments) arrays
    """
    moments = []

    # Set appropriate R for the polynomial set
    if 'R' not in config_kwargs:
        config_kwargs['R'] = 1.14976 if use_optimal else PRZZ_R

    for Delta in delta_values:
        config = VZetaMomentConfig(T=T, Delta=Delta, **config_kwargs)
        result = compute_vzeta_psi_moment(config, use_optimal=use_optimal)
        moments.append(result.moment)

    return delta_values, np.array(moments)


def validate_global_limit(
    T: float,
    use_optimal: bool = True,
    target_c: Optional[float] = None,
    delta_max: float = 10.0,
    n_deltas: int = 20,
    rtol: float = 0.1,
) -> Tuple[bool, float, float]:
    """Validate that Delta→0 limit recovers global c.

    Args:
        T: Height parameter
        use_optimal: Use optimal or PRZZ polynomials
        target_c: Expected c value (defaults to 1.0 for optimal, 2.137 for PRZZ)
        delta_max: Maximum Delta value
        n_deltas: Number of Delta values
        rtol: Relative tolerance for validation

    Returns:
        (passed, extrapolated_c, error)
    """
    if target_c is None:
        target_c = 1.0 if use_optimal else 2.137

    deltas = np.linspace(0.1, delta_max, n_deltas)
    _, moments = delta_sweep(T, deltas, use_optimal=use_optimal)

    # Extrapolate to Delta=0 using linear fit on small-Delta region
    small_idx = deltas < delta_max / 3
    if np.sum(small_idx) >= 3:
        coeffs = np.polyfit(deltas[small_idx], moments[small_idx], 1)
        extrapolated = coeffs[1]  # y-intercept
    else:
        extrapolated = moments[0]

    error = abs(extrapolated - target_c) / target_c
    passed = error < rtol

    return passed, extrapolated, error


def compare_optimal_vs_przz(
    T: float,
    Delta: float,
    **config_kwargs,
) -> Dict[str, Any]:
    """Compare optimal and PRZZ moments at the same (T, Delta).

    Args:
        T: Height parameter
        Delta: Bandwidth
        **config_kwargs: Additional config parameters

    Returns:
        Dictionary with comparison results
    """
    # Compute with optimal polynomials (R = 1.14976)
    config_opt = VZetaMomentConfig(T=T, Delta=Delta, R=1.14976, **config_kwargs)
    result_opt = compute_vzeta_psi_moment(config_opt, use_optimal=True, include_decomposition=True)

    # Compute with PRZZ polynomials (R = 1.3036)
    config_przz = VZetaMomentConfig(T=T, Delta=Delta, R=PRZZ_R, **config_kwargs)
    result_przz = compute_vzeta_psi_moment(config_przz, use_optimal=False, include_decomposition=True)

    return {
        'T': T,
        'Delta': Delta,
        'optimal': {
            'moment': result_opt.moment,
            'R': config_opt.R,
            'sigma': config_opt.sigma,
            'diagonal': result_opt.ratio_decomposition.diagonal if result_opt.ratio_decomposition else None,
            'off_diagonal': result_opt.ratio_decomposition.off_diagonal if result_opt.ratio_decomposition else None,
        },
        'przz': {
            'moment': result_przz.moment,
            'R': config_przz.R,
            'sigma': config_przz.sigma,
            'diagonal': result_przz.ratio_decomposition.diagonal if result_przz.ratio_decomposition else None,
            'off_diagonal': result_przz.ratio_decomposition.off_diagonal if result_przz.ratio_decomposition else None,
        },
        'ratio': result_opt.moment / result_przz.moment if result_przz.moment != 0 else float('nan'),
        'target_ratio': 1.0 / 2.137,  # Expected at global limit
    }

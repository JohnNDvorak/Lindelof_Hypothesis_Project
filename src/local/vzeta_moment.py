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
        sigma_override: If set, use this σ directly (bypasses Levinson line)
        n_halfwidth: Number of Fejér zeros for truncation
        n_points_per_zero: Quadrature resolution
    """
    T: float
    Delta: float
    R: float = 1.14976  # Optimal R for c=1 (PRZZ uses 1.3036)
    theta: float = 4/7
    N: Optional[int] = None
    use_levinson_line: bool = True
    sigma_override: Optional[float] = None  # Phase 4: bypass Levinson line
    n_halfwidth: float = 4.0
    n_points_per_zero: int = 20

    @property
    def sigma(self) -> float:
        """Evaluation point: σ = 1/2 - R/log(T) on Levinson/Conrey line."""
        if self.sigma_override is not None:
            return self.sigma_override
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
        diagonal: Σ|c_k|²k^{-2σ} — the Δ→0 limit (always computed)
        off_diagonal: moment - diagonal (near-diagonal interference)
        off_over_diag: off_diagonal / diagonal (normalized cancellation)
        config: Configuration used
        local_result: Full LocalMomentResult from time-domain computation
        vzeta_coeffs: V[ζ] coefficients
        psi_coeffs: Mollifier coefficients (before convolution)
        convolved_coeffs: (V[ζ] · ψ) coefficients after convolution
        ratio_decomposition: Optional full ratio-domain decomposition
        diagnostics: Additional diagnostic info
    """
    moment: float
    diagonal: float  # Phase 4: always computed
    off_diagonal: float  # Phase 4: always computed
    off_over_diag: float  # Phase 4: always computed
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

    # Step 6: Always compute diagonal/off-diagonal (Phase 4)
    # diagonal = Σ|c_k|²k^{-2σ} — this is the exact Δ→0 limit
    k_arr = np.arange(1, len(convolved_coeffs), dtype=np.float64)
    diagonal = np.sum(np.abs(convolved_coeffs[1:])**2 * k_arr**(-2 * config.sigma))
    off_diagonal = local_result.moment - diagonal
    off_over_diag = off_diagonal / diagonal if diagonal != 0 else 0.0

    # Step 7: Optional full decomposition
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
        'diagonal': diagonal,
        'off_diagonal': off_diagonal,
        'off_over_diag': off_over_diag,
    }

    return VZetaMomentResult(
        moment=local_result.moment,
        diagonal=diagonal,
        off_diagonal=off_diagonal,
        off_over_diag=off_over_diag,
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


# ============================================================================
# Phase 4: Apples-to-Apples Diagnostics
# ============================================================================

def compare_same_sigma(
    T: float,
    Delta: float,
    sigma: float,
    **config_kwargs,
) -> Dict[str, Any]:
    """Compare optimal and PRZZ moments at identical σ (removes confounding).

    This is the key Phase 4 diagnostic: by fixing σ, we remove the confounding
    effect of different Levinson lines and get a true apples-to-apples comparison.

    Args:
        T: Height parameter
        Delta: Bandwidth
        sigma: Fixed σ value for both polynomial sets
        **config_kwargs: Additional config parameters

    Returns:
        Dictionary with comparison results including off/diag analysis
    """
    # Compute with optimal polynomials at fixed σ
    config_opt = VZetaMomentConfig(
        T=T, Delta=Delta, sigma_override=sigma, **config_kwargs
    )
    result_opt = compute_vzeta_psi_moment(config_opt, use_optimal=True)

    # Compute with PRZZ polynomials at same fixed σ
    config_przz = VZetaMomentConfig(
        T=T, Delta=Delta, sigma_override=sigma, **config_kwargs
    )
    result_przz = compute_vzeta_psi_moment(config_przz, use_optimal=False)

    return {
        'T': T,
        'Delta': Delta,
        'sigma': sigma,
        'optimal': {
            'moment': result_opt.moment,
            'diagonal': result_opt.diagonal,
            'off_diagonal': result_opt.off_diagonal,
            'off_over_diag': result_opt.off_over_diag,
        },
        'przz': {
            'moment': result_przz.moment,
            'diagonal': result_przz.diagonal,
            'off_diagonal': result_przz.off_diagonal,
            'off_over_diag': result_przz.off_over_diag,
        },
        'moment_ratio': result_opt.moment / result_przz.moment if result_przz.moment != 0 else float('nan'),
        'diag_ratio': result_opt.diagonal / result_przz.diagonal if result_przz.diagonal != 0 else float('nan'),
        'off_diag_ratio': result_opt.off_diagonal / result_przz.off_diagonal if result_przz.off_diagonal != 0 else float('nan'),
    }


def validate_global_limit_v2(
    T: float,
    use_optimal: bool = True,
    target_c: Optional[float] = None,
    rtol: float = 0.3,
    **config_kwargs,
) -> Tuple[bool, float, float]:
    """Validate using diagonal as exact Δ→0 limit (not extrapolation).

    The key insight: as Δ→0, the Fejér window shrinks to a delta function,
    so the only surviving terms are k=k' (diagonal). Therefore:

        lim_{Δ→0} M_Δ(T) = Σ_k |c_k|² k^{-2σ} = diagonal

    This is computed exactly, no extrapolation needed.

    Args:
        T: Height parameter
        use_optimal: Use optimal or PRZZ polynomials
        target_c: Expected c value (defaults to 1.0 for optimal, 2.137 for PRZZ)
        rtol: Relative tolerance for validation
        **config_kwargs: Additional config parameters

    Returns:
        (passed, diagonal, error)
    """
    if target_c is None:
        target_c = 1.0 if use_optimal else 2.137

    # Delta doesn't matter since we use diagonal directly
    config = VZetaMomentConfig(T=T, Delta=1.0, **config_kwargs)
    result = compute_vzeta_psi_moment(config, use_optimal=use_optimal)

    # diagonal IS the Δ→0 limit
    diagonal = result.diagonal

    error = abs(diagonal - target_c) / target_c
    passed = error < rtol

    return passed, diagonal, error


# Mesoscopic delta grid for testing wide-window behavior
MESOSCOPIC_DELTAS = np.array([0.005, 0.01, 0.02, 0.05, 0.1])
STANDARD_DELTAS = np.array([0.5, 1.0, 2.0, 3.0, 5.0])


def mesoscopic_sweep(
    T: float,
    use_optimal: bool = True,
    include_standard: bool = True,
    **config_kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sweep mesoscopic delta values where window width ~ 1/Δ is large.

    This tests Hypothesis 4(a): does cancellation emerge when the
    averaging length is large (small Δ)?

    Args:
        T: Height parameter
        use_optimal: Use optimal or PRZZ polynomials
        include_standard: Include standard delta values [0.5, 1, 2, 3, 5]
        **config_kwargs: Additional config parameters

    Returns:
        (deltas, moments, diagonals, off_over_diags) arrays
    """
    if include_standard:
        all_deltas = np.sort(np.concatenate([MESOSCOPIC_DELTAS, STANDARD_DELTAS]))
    else:
        all_deltas = MESOSCOPIC_DELTAS

    moments = []
    diagonals = []
    off_over_diags = []

    # Set appropriate R for the polynomial set
    if 'R' not in config_kwargs:
        config_kwargs['R'] = 1.14976 if use_optimal else PRZZ_R

    for Delta in all_deltas:
        config = VZetaMomentConfig(T=T, Delta=Delta, **config_kwargs)
        result = compute_vzeta_psi_moment(config, use_optimal=use_optimal)
        moments.append(result.moment)
        diagonals.append(result.diagonal)
        off_over_diags.append(result.off_over_diag)

    return all_deltas, np.array(moments), np.array(diagonals), np.array(off_over_diags)


def adaptive_delta_sweep(
    T: float,
    alphas: Optional[list] = None,
    use_optimal: bool = True,
    **config_kwargs,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use Δ = T^{-α} to scale window width with T.

    This allows the averaging window to grow with T, testing whether
    cancellation emerges as both T and window width increase together.

    Args:
        T: Height parameter
        alphas: Exponents for Δ = T^{-α} (default: [0.2, 0.3, 0.4, 0.5])
        use_optimal: Use optimal or PRZZ polynomials
        **config_kwargs: Additional config parameters

    Returns:
        (deltas, moments, off_over_diags) arrays
    """
    if alphas is None:
        alphas = [0.2, 0.3, 0.4, 0.5]

    deltas = np.array([T ** (-alpha) for alpha in alphas])
    moments = []
    off_over_diags = []

    if 'R' not in config_kwargs:
        config_kwargs['R'] = 1.14976 if use_optimal else PRZZ_R

    for Delta in deltas:
        config = VZetaMomentConfig(T=T, Delta=Delta, **config_kwargs)
        result = compute_vzeta_psi_moment(config, use_optimal=use_optimal)
        moments.append(result.moment)
        off_over_diags.append(result.off_over_diag)

    return deltas, np.array(moments), np.array(off_over_diags)


def off_diag_comparison_grid(
    T_values: list,
    Delta_values: list,
    sigma: Optional[float] = None,
) -> Dict[str, Any]:
    """Systematic comparison of off/diag across T and Δ grid.

    This is the key Phase 4 test: does optimal have systematically
    more negative off/diag than PRZZ?

    Args:
        T_values: List of T values to test
        Delta_values: List of Delta values to test
        sigma: If set, use fixed σ; otherwise use each set's Levinson line

    Returns:
        Dictionary with full comparison grid
    """
    results = {
        'T_values': T_values,
        'Delta_values': Delta_values,
        'sigma': sigma,
        'grid': [],
    }

    for T in T_values:
        for Delta in Delta_values:
            if sigma is not None:
                comp = compare_same_sigma(T, Delta, sigma)
            else:
                comp = compare_optimal_vs_przz(T, Delta)

            results['grid'].append({
                'T': T,
                'Delta': Delta,
                'opt_off_over_diag': comp['optimal']['off_over_diag'] if 'off_over_diag' in comp['optimal'] else None,
                'przz_off_over_diag': comp['przz']['off_over_diag'] if 'off_over_diag' in comp['przz'] else None,
                'moment_ratio': comp.get('moment_ratio', comp.get('ratio')),
            })

    return results

"""
Phase 5: Actual ζ Sanity Test

This module implements Track C from Phase 5: testing whether ANY local signal
exists when using the actual Riemann zeta function (via mpmath) instead of
a truncated Dirichlet polynomial model.

The key question: If we compute |ζ(s) · ψ(s)|² using actual ζ, do we see
any difference between optimal (c=1) and PRZZ mollifiers?

If actual_ratio differs from dirichlet_ratio → Track A (AFE+mirror) is justified
If actual_ratio ≈ dirichlet_ratio ≈ 1.2 → Track B (Jensen/zero counts) is better

Usage:
    from src.local.actual_zeta_probe import run_sanity_test_grid
    results = run_sanity_test_grid()
    print(results)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

try:
    import mpmath
    MPMATH_AVAILABLE = True
except ImportError:
    MPMATH_AVAILABLE = False

from src.local.vzeta_moment import (
    VZetaMomentConfig,
    compute_vzeta_psi_moment,
    compute_full_mollifier_coeffs,
)
from src.local.v_operator import load_optimal_Q, load_przz_Q, compute_vzeta_coeffs
from src.local.fejer import FejerKernel


@dataclass
class ActualZetaResult:
    """Result of computing moment with actual ζ function.

    Attributes:
        T: Height parameter
        Delta: Bandwidth parameter
        sigma: Real part of s
        actual_moment: Moment computed with actual ζ
        actual_diagonal: Diagonal term (ζ contribution at exact point)
        computation_time: Time taken (seconds)
    """
    T: float
    Delta: float
    sigma: float
    actual_moment: float
    actual_diagonal: float
    computation_time: float


@dataclass
class ComparisonResult:
    """Result of comparing actual ζ vs Dirichlet polynomial model.

    Attributes:
        T: Height
        Delta: Bandwidth
        sigma: Real part
        actual_opt: Moment using actual ζ with optimal mollifier
        actual_przz: Moment using actual ζ with PRZZ mollifier
        dirichlet_opt: Moment using Dirichlet model with optimal
        dirichlet_przz: Moment using Dirichlet model with PRZZ
        actual_ratio: actual_opt / actual_przz
        dirichlet_ratio: dirichlet_opt / dirichlet_przz
        ratio_difference: actual_ratio - dirichlet_ratio
    """
    T: float
    Delta: float
    sigma: float
    actual_opt: float
    actual_przz: float
    dirichlet_opt: float
    dirichlet_przz: float
    actual_ratio: float
    dirichlet_ratio: float
    ratio_difference: float


def check_mpmath_available():
    """Check if mpmath is available."""
    if not MPMATH_AVAILABLE:
        raise ImportError(
            "mpmath is required for actual_zeta_probe. "
            "Install with: pip install mpmath"
        )


def compute_zeta_psi_actual(
    t: float,
    sigma: float,
    psi_coeffs: np.ndarray,
) -> complex:
    """
    Compute ζ(s) · ψ(s) using actual zeta function.

    This computes the product of:
    - ζ(s) via mpmath.zeta (the actual Riemann zeta function)
    - ψ(s) = Σ_n a_n n^{-s} (Dirichlet polynomial mollifier)

    Args:
        t: Imaginary part of s
        sigma: Real part of s
        psi_coeffs: Mollifier coefficients a_n, shape (N+1,)

    Returns:
        Complex value ζ(σ+it) · ψ(σ+it)
    """
    check_mpmath_available()

    s = complex(sigma, t)

    # Compute actual zeta
    mpmath.mp.dps = 30  # 30 digits of precision
    zeta_s = complex(mpmath.zeta(s))

    # Compute mollifier ψ(s) = Σ_n a_n n^{-s}
    N = len(psi_coeffs) - 1
    psi_s = sum(
        psi_coeffs[n] * (n ** (-s))
        for n in range(1, N + 1)
        if psi_coeffs[n] != 0  # Skip zero coefficients for speed
    )

    return zeta_s * psi_s


def compute_vzeta_psi_actual(
    t: float,
    sigma: float,
    psi_coeffs: np.ndarray,
    Q_coeffs: np.ndarray,
    M: int,
) -> complex:
    """
    Compute V[ζ](s) · ψ(s) using actual zeta function for V[ζ].

    Two interpretations of "V applied to actual ζ":

    1. Simple: V[ζ](s) = (Σ_m Q_m m^{-s}) · ζ(s)
       This treats V as a multiplicative operator with actual ζ.

    2. Convolution model: Use our existing Dirichlet polynomial model.
       This is what Phase 3/4 computed.

    This function implements interpretation (1) for comparison.

    Args:
        t: Imaginary part
        sigma: Real part
        psi_coeffs: Mollifier coefficients
        Q_coeffs: Q polynomial coefficients (from evaluating Q at log(m)/log(M))
        M: Length of V[ζ] polynomial

    Returns:
        Complex value
    """
    check_mpmath_available()

    s = complex(sigma, t)

    # Actual zeta
    mpmath.mp.dps = 30
    zeta_s = complex(mpmath.zeta(s))

    # V[ζ](s) as simple product: (Σ_m Q_m m^{-s}) · ζ(s)
    # Q_coeffs should be b_m = Q(log(m)/log(M)) for m = 1..M
    v_zeta = sum(
        Q_coeffs[m] * (m ** (-s))
        for m in range(1, min(M + 1, len(Q_coeffs)))
        if Q_coeffs[m] != 0
    ) * zeta_s

    # Mollifier ψ(s)
    N = len(psi_coeffs) - 1
    psi_s = sum(
        psi_coeffs[n] * (n ** (-s))
        for n in range(1, N + 1)
        if psi_coeffs[n] != 0
    )

    return v_zeta * psi_s


def fejer_windowed_moment_actual_zeta(
    T: float,
    Delta: float,
    sigma: float,
    psi_coeffs: np.ndarray,
    n_points: int = 100,
    use_v_operator: bool = False,
    Q_coeffs: Optional[np.ndarray] = None,
    M: Optional[int] = None,
    n_halfwidth: float = 4.0,
) -> Tuple[float, float]:
    """
    Compute ∫ |ζ(σ+it) · ψ(σ+it)|² w_Δ(t-T) dt using actual ζ.

    This is the Fejér-windowed second moment using mpmath.zeta.

    Args:
        T: Center of window
        Delta: Bandwidth parameter
        sigma: Real part of s
        psi_coeffs: Mollifier coefficients
        n_points: Number of quadrature points
        use_v_operator: If True, compute |Vζ·ψ|² instead of |ζ·ψ|²
        Q_coeffs: V[ζ] coefficients (required if use_v_operator=True)
        M: V[ζ] length (required if use_v_operator=True)
        n_halfwidth: Number of first-zero widths for integration range

    Returns:
        (moment, diagonal_estimate) tuple
    """
    check_mpmath_available()

    # Fejér kernel parameters
    kernel = FejerKernel(Delta)
    # Half-width in time: n_halfwidth times the first zero location
    half_width = n_halfwidth * kernel.first_zero

    # Quadrature grid
    t_grid = np.linspace(T - half_width, T + half_width, n_points)
    dt = t_grid[1] - t_grid[0]

    # Compute moment via quadrature
    moment = 0.0
    center_value = 0.0

    for i, t in enumerate(t_grid):
        if use_v_operator and Q_coeffs is not None and M is not None:
            val = compute_vzeta_psi_actual(t, sigma, psi_coeffs, Q_coeffs, M)
        else:
            val = compute_zeta_psi_actual(t, sigma, psi_coeffs)

        weight = kernel.w_time(t - T)
        moment += abs(val) ** 2 * weight * dt

        # Track center value for diagonal estimate
        if i == n_points // 2:
            center_value = abs(val) ** 2

    # Normalize (already normalized in kernel, but convention is moment / (2*pi))
    # Actually the integral ∫ w(t) dt = 1 by normalization, so just return moment
    return moment, center_value


def compare_actual_vs_dirichlet(
    T: float,
    Delta: float = 1.0,
    sigma: Optional[float] = None,
    N: int = 40,
    n_points: int = 100,
    use_v_operator: bool = False,
) -> ComparisonResult:
    """
    Compare actual ζ moment vs Dirichlet polynomial model.

    This is the key diagnostic: do we see different behavior between
    optimal and PRZZ when using actual ζ vs the Dirichlet model?

    Args:
        T: Height parameter
        Delta: Bandwidth
        sigma: Real part (defaults to 0.35)
        N: Mollifier length
        n_points: Quadrature points for actual ζ
        use_v_operator: If True, include V operator in actual computation

    Returns:
        ComparisonResult with all moment values and ratios
    """
    if sigma is None:
        sigma = 0.35

    # === Dirichlet model (Phase 3/4 approach) ===
    config_opt = VZetaMomentConfig(
        T=T, Delta=Delta, N=N, sigma_override=sigma
    )
    config_przz = VZetaMomentConfig(
        T=T, Delta=Delta, N=N, sigma_override=sigma
    )

    dirichlet_opt_result = compute_vzeta_psi_moment(config_opt, use_optimal=True)
    dirichlet_przz_result = compute_vzeta_psi_moment(config_przz, use_optimal=False)

    dirichlet_opt = dirichlet_opt_result.moment
    dirichlet_przz = dirichlet_przz_result.moment

    # === Actual ζ computation ===
    # Load mollifier coefficients
    psi_opt = compute_full_mollifier_coeffs(N, use_optimal=True)
    psi_przz = compute_full_mollifier_coeffs(N, use_optimal=False)

    if use_v_operator:
        # Load Q coefficients
        Q_opt, _ = load_optimal_Q()
        Q_przz, _ = load_przz_Q()

        vzeta_opt = compute_vzeta_coeffs(N, Q_opt)
        vzeta_przz = compute_vzeta_coeffs(N, Q_przz)

        actual_opt, _ = fejer_windowed_moment_actual_zeta(
            T, Delta, sigma, psi_opt, n_points,
            use_v_operator=True, Q_coeffs=vzeta_opt.b, M=N
        )
        actual_przz, _ = fejer_windowed_moment_actual_zeta(
            T, Delta, sigma, psi_przz, n_points,
            use_v_operator=True, Q_coeffs=vzeta_przz.b, M=N
        )
    else:
        # Simple |ζ·ψ|² comparison (no V operator)
        actual_opt, _ = fejer_windowed_moment_actual_zeta(
            T, Delta, sigma, psi_opt, n_points
        )
        actual_przz, _ = fejer_windowed_moment_actual_zeta(
            T, Delta, sigma, psi_przz, n_points
        )

    # Compute ratios
    actual_ratio = actual_opt / actual_przz if actual_przz != 0 else float('inf')
    dirichlet_ratio = dirichlet_opt / dirichlet_przz if dirichlet_przz != 0 else float('inf')

    return ComparisonResult(
        T=T,
        Delta=Delta,
        sigma=sigma,
        actual_opt=actual_opt,
        actual_przz=actual_przz,
        dirichlet_opt=dirichlet_opt,
        dirichlet_przz=dirichlet_przz,
        actual_ratio=actual_ratio,
        dirichlet_ratio=dirichlet_ratio,
        ratio_difference=actual_ratio - dirichlet_ratio,
    )


def run_sanity_test_grid(
    T_values: Optional[List[float]] = None,
    Delta_values: Optional[List[float]] = None,
    sigma_values: Optional[List[float]] = None,
    N: int = 40,
    n_points: int = 80,
    use_v_operator: bool = False,
    verbose: bool = True,
) -> List[ComparisonResult]:
    """
    Run sanity test grid comparing actual ζ vs Dirichlet model.

    This sweeps over T, Δ, σ to see if there's any regime where
    actual ζ shows different optimal/PRZZ behavior than the Dirichlet model.

    Args:
        T_values: Heights to test (default: [100, 300, 500, 1000])
        Delta_values: Bandwidths (default: [0.5, 1.0, 2.0])
        sigma_values: Real parts (default: [0.3, 0.35, 0.4])
        N: Mollifier length
        n_points: Quadrature points
        use_v_operator: Include V operator in actual computation
        verbose: Print progress

    Returns:
        List of ComparisonResult for each parameter combination
    """
    check_mpmath_available()

    if T_values is None:
        T_values = [100, 300, 500, 1000]
    if Delta_values is None:
        Delta_values = [0.5, 1.0, 2.0]
    if sigma_values is None:
        sigma_values = [0.30, 0.35, 0.40]

    results = []
    total = len(T_values) * len(Delta_values) * len(sigma_values)
    count = 0

    for T in T_values:
        for Delta in Delta_values:
            for sigma in sigma_values:
                count += 1
                if verbose:
                    print(f"[{count}/{total}] T={T}, Δ={Delta}, σ={sigma}...", end=" ", flush=True)

                try:
                    result = compare_actual_vs_dirichlet(
                        T=T,
                        Delta=Delta,
                        sigma=sigma,
                        N=N,
                        n_points=n_points,
                        use_v_operator=use_v_operator,
                    )
                    results.append(result)

                    if verbose:
                        print(f"actual_ratio={result.actual_ratio:.4f}, "
                              f"dirichlet_ratio={result.dirichlet_ratio:.4f}, "
                              f"diff={result.ratio_difference:+.4f}")
                except Exception as e:
                    if verbose:
                        print(f"ERROR: {e}")

    return results


def summarize_results(results: List[ComparisonResult]) -> Dict:
    """
    Summarize sanity test results.

    Args:
        results: List of ComparisonResult from run_sanity_test_grid

    Returns:
        Dict with summary statistics
    """
    if not results:
        return {"error": "No results to summarize"}

    actual_ratios = [r.actual_ratio for r in results]
    dirichlet_ratios = [r.dirichlet_ratio for r in results]
    differences = [r.ratio_difference for r in results]

    return {
        "n_tests": len(results),
        "actual_ratio": {
            "mean": np.mean(actual_ratios),
            "std": np.std(actual_ratios),
            "min": np.min(actual_ratios),
            "max": np.max(actual_ratios),
        },
        "dirichlet_ratio": {
            "mean": np.mean(dirichlet_ratios),
            "std": np.std(dirichlet_ratios),
            "min": np.min(dirichlet_ratios),
            "max": np.max(dirichlet_ratios),
        },
        "difference": {
            "mean": np.mean(differences),
            "std": np.std(differences),
            "min": np.min(differences),
            "max": np.max(differences),
        },
        "conclusion": _determine_conclusion(differences),
    }


def _determine_conclusion(differences: List[float]) -> str:
    """Determine conclusion based on ratio differences."""
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)

    # If mean difference is significant (> 2 std from zero), there's a signal
    if abs(mean_diff) > 0.1 and abs(mean_diff) > 2 * std_diff:
        return "SIGNAL: Actual ζ shows different behavior. Pursue Track A (AFE+mirror)."
    elif std_diff > 0.2:
        return "UNCLEAR: High variance in differences. May need more data or larger T."
    else:
        return "NO SIGNAL: Actual ζ matches Dirichlet model. Pursue Track B (Jensen/zero counts)."


def format_results_table(results: List[ComparisonResult]) -> str:
    """Format results as a readable table."""
    lines = [
        "=" * 100,
        "Phase 5 Sanity Test: Actual ζ vs Dirichlet Model",
        "=" * 100,
        "",
        f"{'T':>6} | {'Δ':>4} | {'σ':>5} | {'Actual Opt':>11} | {'Actual PRZZ':>11} | "
        f"{'Dir. Opt':>9} | {'Dir. PRZZ':>9} | {'Act Ratio':>9} | {'Dir Ratio':>9} | {'Diff':>7}",
        "-" * 100,
    ]

    for r in results:
        lines.append(
            f"{r.T:>6.0f} | {r.Delta:>4.1f} | {r.sigma:>5.2f} | "
            f"{r.actual_opt:>11.4f} | {r.actual_przz:>11.4f} | "
            f"{r.dirichlet_opt:>9.4f} | {r.dirichlet_przz:>9.4f} | "
            f"{r.actual_ratio:>9.4f} | {r.dirichlet_ratio:>9.4f} | "
            f"{r.ratio_difference:>+7.4f}"
        )

    lines.extend([
        "-" * 100,
        "",
    ])

    # Add summary
    summary = summarize_results(results)
    lines.extend([
        "SUMMARY:",
        f"  Actual ratio:    mean={summary['actual_ratio']['mean']:.4f}, "
        f"std={summary['actual_ratio']['std']:.4f}, "
        f"range=[{summary['actual_ratio']['min']:.4f}, {summary['actual_ratio']['max']:.4f}]",
        f"  Dirichlet ratio: mean={summary['dirichlet_ratio']['mean']:.4f}, "
        f"std={summary['dirichlet_ratio']['std']:.4f}, "
        f"range=[{summary['dirichlet_ratio']['min']:.4f}, {summary['dirichlet_ratio']['max']:.4f}]",
        f"  Difference:      mean={summary['difference']['mean']:+.4f}, "
        f"std={summary['difference']['std']:.4f}",
        "",
        f"CONCLUSION: {summary['conclusion']}",
        "=" * 100,
    ])

    return "\n".join(lines)


# Quick test function
def quick_test(T: float = 500, sigma: float = 0.35) -> ComparisonResult:
    """Run a quick single-point test."""
    check_mpmath_available()
    print(f"Quick test at T={T}, σ={sigma}...")
    result = compare_actual_vs_dirichlet(T=T, Delta=1.0, sigma=sigma, N=40, n_points=60)
    print(f"  Actual ratio:    {result.actual_ratio:.4f}")
    print(f"  Dirichlet ratio: {result.dirichlet_ratio:.4f}")
    print(f"  Difference:      {result.ratio_difference:+.4f}")
    return result

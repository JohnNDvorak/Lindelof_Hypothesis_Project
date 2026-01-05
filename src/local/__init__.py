"""
Localized/Band-Limited Moment Engine

This package provides tools for computing localized moments of Dirichlet
polynomials using Fejer band-limited windows.

Main Components:
- FejerKernel: Band-limited window kernel
- SieveArrays: Fast arithmetic function arrays
- MollifierCoeffs: Coefficient generators for psi_1, psi_2, psi_3
- DirichletPolyResult: Dirichlet polynomial evaluator
- LocalMomentResult: Localized moment computation
- LocalEngine: Main engine class (follows KappaEngine pattern)

Usage:
    from src.local import LocalEngine

    engine = LocalEngine.from_config(N=1000, theta=4/7)
    result = engine.compute_moment(T=1000, Delta=1.0)
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from src.local.fejer import FejerKernel, delta_from_first_zero
from src.local.sieve import SieveArrays, compute_sieve_arrays
from src.local.mollifier_coeffs import (
    MollifierCoeffs,
    compute_mollifier_coeffs,
    compute_psi1_coeffs,
    load_optimal_polynomials,
    load_przz_polynomials,
)
from src.local.dirichlet_poly import (
    DirichletPolyResult,
    evaluate_dirichlet_poly,
)
from src.local.local_moment import (
    LocalMomentConfig,
    LocalMomentResult,
    RatioMomentDecomposition,
    compute_local_moment,
    compute_ratio_domain_moment,
    compute_ratio_domain_decomposed,
    verify_moment_consistency,
)
from src.local.ratio_classes import (
    RatioClassContribution,
    RatioClassDecomposition,
    compute_ratio_class_contribution,
    compute_ratio_classes,
    print_ratio_class_table,
)
from src.local.ratio_atoms import (
    TaylorCoefficient,
    RatioAtom,
    compute_taylor_coefficient,
    compute_ratio_atom,
    compute_prime_atoms,
    print_atom_table,
    compare_atom_polynomials,
    endpoint_derivative_analysis,
)
from src.local.v_operator import (
    VZetaCoeffs,
    compute_vzeta_coeffs,
    dirichlet_convolve,
    compute_vzeta_psi_coeffs,
    load_optimal_Q,
    load_przz_Q,
    load_optimal_polynomials_full,
)
from src.local.vzeta_moment import (
    VZetaMomentConfig,
    VZetaMomentResult,
    compute_vzeta_psi_moment,
    compute_full_mollifier_coeffs,
    delta_sweep,
    validate_global_limit,
    compare_optimal_vs_przz,
    # Phase 4 additions
    compare_same_sigma,
    validate_global_limit_v2,
    mesoscopic_sweep,
    adaptive_delta_sweep,
    off_diag_comparison_grid,
    MESOSCOPIC_DELTAS,
    STANDARD_DELTAS,
)
from src.local.actual_zeta_probe import (
    ActualZetaResult,
    ComparisonResult,
    compute_zeta_psi_actual,
    compare_actual_vs_dirichlet,
    run_sanity_test_grid,
    summarize_results,
    format_results_table,
    quick_test,
)


@dataclass
class LocalEngineConfig:
    """Configuration for LocalEngine.

    Attributes:
        N: Mollifier length
        theta: Mollifier exponent (typically 4/7)
        sigma: Real part of s (typically 0.5 for critical line)
        which_psi: Which mollifiers to compute (psi_1, psi_2, psi_3)
        use_optimal: If True, use optimal polynomials (c=1); if False, use PRZZ baseline
    """
    N: int
    theta: float = 4/7
    sigma: float = 0.5
    which_psi: Tuple[bool, bool, bool] = (True, False, False)
    use_optimal: bool = True


class LocalEngine:
    """
    Engine for computing localized moments of mollified Dirichlet polynomials.

    Follows the pattern of KappaEngine with:
    - Lazy loading of heavy computations
    - Factory methods for common configurations
    - Structured output dataclasses

    Example:
        engine = LocalEngine.from_config(N=1000, theta=4/7)
        result = engine.compute_moment(T=1000, Delta=1.0)
        print(f"Localized moment: {result.moment}")
    """

    def __init__(self, config: LocalEngineConfig):
        """Initialize engine with configuration."""
        self.config = config
        self._coeffs: Optional[MollifierCoeffs] = None
        self._sieve: Optional[SieveArrays] = None

    @classmethod
    def from_config(
        cls,
        N: int,
        theta: float = 4/7,
        sigma: float = 0.5,
        which_psi: Tuple[bool, bool, bool] = (True, False, False),
        use_optimal: bool = True,
    ) -> "LocalEngine":
        """Factory method for creating engine from parameters.

        Args:
            N: Mollifier length
            theta: Mollifier exponent
            sigma: Real part of s
            which_psi: (compute_psi1, compute_psi2, compute_psi3)
            use_optimal: Use optimal polynomials (c=1) vs PRZZ baseline

        Returns:
            LocalEngine instance
        """
        return cls(LocalEngineConfig(
            N=N,
            theta=theta,
            sigma=sigma,
            which_psi=which_psi,
            use_optimal=use_optimal,
        ))

    @property
    def coeffs(self) -> MollifierCoeffs:
        """Lazy-load mollifier coefficients."""
        if self._coeffs is None:
            self._coeffs = compute_mollifier_coeffs(
                self.config.N,
                which=self.config.which_psi,
                use_optimal=self.config.use_optimal,
            )
        return self._coeffs

    def get_active_coeffs(self) -> np.ndarray:
        """Get the currently active coefficient array.

        Returns the first available coefficient array in order: a1, a2, a3.
        For MVP, this is typically psi_1 coefficients (a1).

        Returns:
            Coefficient array of shape (N+1,)
        """
        if self.coeffs.a1 is not None:
            return self.coeffs.a1
        elif self.coeffs.a2 is not None:
            return self.coeffs.a2
        elif self.coeffs.a3 is not None:
            return self.coeffs.a3
        else:
            raise ValueError("No mollifier coefficients computed")

    def get_combined_coeffs(self) -> np.ndarray:
        """Get combined coefficients a = a1 + a2 + a3.

        Returns:
            Combined coefficient array
        """
        result = np.zeros(self.config.N + 1, dtype=np.float64)
        if self.coeffs.a1 is not None:
            result += self.coeffs.a1
        if self.coeffs.a2 is not None:
            result += self.coeffs.a2
        if self.coeffs.a3 is not None:
            result += self.coeffs.a3
        return result

    def compute_moment(
        self,
        T: float,
        Delta: float,
        n_halfwidth: float = 4.0,
        n_points_per_zero: int = 20,
        use_combined: bool = False,
    ) -> LocalMomentResult:
        """Compute localized moment at center T with bandwidth Delta.

        Args:
            T: Center of localization window
            Delta: Bandwidth parameter
            n_halfwidth: Number of first-zero widths for truncation
            n_points_per_zero: Quadrature resolution
            use_combined: If True, use combined a1+a2+a3; otherwise use first available

        Returns:
            LocalMomentResult with moment value and diagnostics
        """
        config = LocalMomentConfig(
            T=T,
            Delta=Delta,
            sigma=self.config.sigma,
            n_halfwidth=n_halfwidth,
            n_points_per_zero=n_points_per_zero,
        )
        coeffs = self.get_combined_coeffs() if use_combined else self.get_active_coeffs()
        return compute_local_moment(coeffs, config)

    def verify_consistency(
        self,
        T: float,
        Delta: float,
        rtol: float = 1e-3,
        use_combined: bool = False,
    ) -> Tuple[float, float, bool]:
        """Verify time-domain vs ratio-domain moment consistency.

        Args:
            T: Center time
            Delta: Bandwidth
            rtol: Relative tolerance
            use_combined: If True, use combined coefficients

        Returns:
            (time_domain_moment, ratio_domain_moment, passed)
        """
        config = LocalMomentConfig(T=T, Delta=Delta, sigma=self.config.sigma)
        coeffs = self.get_combined_coeffs() if use_combined else self.get_active_coeffs()
        return verify_moment_consistency(coeffs, config, rtol)

    def compute_decomposed(
        self,
        T: float,
        Delta: float,
        use_combined: bool = False,
    ) -> RatioMomentDecomposition:
        """Compute ratio-domain moment with diagonal/off-diagonal decomposition.

        This separates the moment into:
        - Diagonal: Σ_n |a_n|² n^{-2σ}  (always positive)
        - Off-diagonal: Everything else (oscillatory in T)

        Args:
            T: Center time
            Delta: Bandwidth
            use_combined: If True, use combined coefficients

        Returns:
            RatioMomentDecomposition with total, diagonal, off_diagonal, and ratio
        """
        config = LocalMomentConfig(T=T, Delta=Delta, sigma=self.config.sigma)
        coeffs = self.get_combined_coeffs() if use_combined else self.get_active_coeffs()
        return compute_ratio_domain_decomposed(coeffs, config)

    def compute_ratio_classes(
        self,
        T: float,
        Delta: float,
        A_max: int = 50,
        use_combined: bool = False,
    ) -> RatioClassDecomposition:
        """Decompose moment by ratio classes (A, B).

        Identifies which specific coprime pairs (A, B) drive the off-diagonal
        contribution to the localized moment.

        Args:
            T: Center time
            Delta: Bandwidth
            A_max: Maximum value for A and B in the search
            use_combined: If True, use combined coefficients

        Returns:
            RatioClassDecomposition with sorted list of contributions
        """
        config = LocalMomentConfig(T=T, Delta=Delta, sigma=self.config.sigma)
        coeffs = self.get_combined_coeffs() if use_combined else self.get_active_coeffs()
        return compute_ratio_classes(coeffs, config, A_max=A_max)

    def evaluate_dirichlet(
        self,
        t_grid: np.ndarray,
        use_combined: bool = False,
    ) -> DirichletPolyResult:
        """Evaluate Dirichlet polynomial D(sigma + it) on given time grid.

        Args:
            t_grid: Time points
            use_combined: If True, use combined coefficients

        Returns:
            DirichletPolyResult with values and |D|^2
        """
        coeffs = self.get_combined_coeffs() if use_combined else self.get_active_coeffs()
        return evaluate_dirichlet_poly(coeffs, t_grid=t_grid, sigma=self.config.sigma)


__all__ = [
    # Fejer
    'FejerKernel',
    'delta_from_first_zero',
    # Sieve
    'SieveArrays',
    'compute_sieve_arrays',
    # Mollifier
    'MollifierCoeffs',
    'compute_mollifier_coeffs',
    'load_optimal_polynomials',
    'load_przz_polynomials',
    # Dirichlet
    'DirichletPolyResult',
    'evaluate_dirichlet_poly',
    # Moment
    'LocalMomentConfig',
    'LocalMomentResult',
    'RatioMomentDecomposition',
    'compute_local_moment',
    'compute_ratio_domain_moment',
    'compute_ratio_domain_decomposed',
    'verify_moment_consistency',
    # Ratio classes
    'RatioClassContribution',
    'RatioClassDecomposition',
    'compute_ratio_class_contribution',
    'compute_ratio_classes',
    'print_ratio_class_table',
    # Ratio atoms
    'TaylorCoefficient',
    'RatioAtom',
    'compute_taylor_coefficient',
    'compute_ratio_atom',
    'compute_prime_atoms',
    'print_atom_table',
    'compare_atom_polynomials',
    'endpoint_derivative_analysis',
    # V-operator (Phase 3)
    'VZetaCoeffs',
    'compute_vzeta_coeffs',
    'dirichlet_convolve',
    'compute_vzeta_psi_coeffs',
    'load_optimal_Q',
    'load_przz_Q',
    'load_optimal_polynomials_full',
    # VZeta moment (Phase 3)
    'VZetaMomentConfig',
    'VZetaMomentResult',
    'compute_vzeta_psi_moment',
    'compute_full_mollifier_coeffs',
    'delta_sweep',
    'validate_global_limit',
    'compare_optimal_vs_przz',
    # Phase 4: Apples-to-apples diagnostics
    'compare_same_sigma',
    'validate_global_limit_v2',
    'mesoscopic_sweep',
    'adaptive_delta_sweep',
    'off_diag_comparison_grid',
    'MESOSCOPIC_DELTAS',
    'STANDARD_DELTAS',
    # Phase 5: Actual zeta sanity test
    'ActualZetaResult',
    'ComparisonResult',
    'compute_zeta_psi_actual',
    'compare_actual_vs_dirichlet',
    'run_sanity_test_grid',
    'summarize_results',
    'format_results_table',
    'quick_test',
    # Engine
    'LocalEngineConfig',
    'LocalEngine',
]

"""
Tests for localized |V[ζ]·ψ|² moment computation.

Tests cover:
- VZetaMomentConfig properties
- Full mollifier coefficient computation
- Moment computation positivity
- Delta sweep and global limit validation
- Optimal vs PRZZ comparison
"""

import numpy as np
import pytest

from src.local.vzeta_moment import (
    VZetaMomentConfig,
    VZetaMomentResult,
    compute_vzeta_psi_moment,
    compute_full_mollifier_coeffs,
    delta_sweep,
    validate_global_limit,
    compare_optimal_vs_przz,
    load_c1_optimal_mollifier_polynomials,
    load_przz_mollifier_polynomials,
)


class TestVZetaMomentConfig:
    """Tests for VZetaMomentConfig dataclass."""

    def test_sigma_levinson_line(self):
        """Sigma = 0.5 - R/log(T) on Levinson line."""
        config = VZetaMomentConfig(T=1000, Delta=1.0, R=1.14976)

        expected_sigma = 0.5 - 1.14976 / np.log(1000)
        assert abs(config.sigma - expected_sigma) < 1e-10

    def test_sigma_critical_line(self):
        """Sigma = 0.5 when not using Levinson line."""
        config = VZetaMomentConfig(T=1000, Delta=1.0, use_levinson_line=False)

        assert config.sigma == 0.5

    def test_mollifier_length(self):
        """N = T^θ with θ = 4/7."""
        config = VZetaMomentConfig(T=1000, Delta=1.0)

        expected_N = int(1000 ** (4 / 7))
        assert config.mollifier_length == expected_N

    def test_mollifier_length_override(self):
        """N can be overridden explicitly."""
        config = VZetaMomentConfig(T=1000, Delta=1.0, N=100)

        assert config.mollifier_length == 100


class TestMollifierCoeffs:
    """Tests for full mollifier coefficient computation."""

    def test_optimal_coeffs_shape(self):
        """Optimal coefficients have correct shape."""
        N = 100
        coeffs = compute_full_mollifier_coeffs(N, use_optimal=True)

        assert len(coeffs) == N + 1

    def test_przz_coeffs_shape(self):
        """PRZZ coefficients have correct shape."""
        N = 100
        coeffs = compute_full_mollifier_coeffs(N, use_optimal=False)

        assert len(coeffs) == N + 1

    def test_coeffs_nonzero(self):
        """Some coefficients should be nonzero."""
        N = 100
        coeffs = compute_full_mollifier_coeffs(N, use_optimal=True)

        # a[1] should be nonzero (ψ₁ contributes)
        assert abs(coeffs[1]) > 0.1

    def test_load_optimal_polynomials(self):
        """Optimal mollifier polynomials load correctly."""
        P1, P2, P3 = load_c1_optimal_mollifier_polynomials()

        # P1 tilde should have 4 coefficients
        assert len(P1.tilde_coeffs) == 4

    def test_load_przz_polynomials(self):
        """PRZZ mollifier polynomials load correctly."""
        P1, P2, P3 = load_przz_mollifier_polynomials()

        # Should have some tilde coefficients
        assert len(P1.tilde_coeffs) >= 2


class TestVZetaMoment:
    """Tests for the main moment computation."""

    def test_moment_positive(self):
        """Moment |Vζψ|² should be positive."""
        config = VZetaMomentConfig(T=500, Delta=1.0, N=50)
        result = compute_vzeta_psi_moment(config, use_optimal=True)

        assert result.moment > 0

    def test_moment_with_przz(self):
        """PRZZ moment computation works."""
        config = VZetaMomentConfig(T=500, Delta=1.0, N=50, R=1.3036)
        result = compute_vzeta_psi_moment(config, use_optimal=False)

        assert result.moment > 0

    def test_result_has_diagnostics(self):
        """Result includes diagnostics."""
        config = VZetaMomentConfig(T=500, Delta=1.0, N=50)
        result = compute_vzeta_psi_moment(config, use_optimal=True)

        assert result.diagnostics is not None
        assert 'Q_type' in result.diagnostics
        assert result.diagnostics['Q_type'] == 'optimal'

    def test_result_has_coefficients(self):
        """Result includes coefficient arrays."""
        config = VZetaMomentConfig(T=500, Delta=1.0, N=50)
        result = compute_vzeta_psi_moment(config, use_optimal=True)

        # V[ζ] coefficients
        assert result.vzeta_coeffs is not None
        assert result.vzeta_coeffs.b[1] == 1.0  # Q(0) = 1

        # ψ coefficients
        assert result.psi_coeffs is not None
        assert len(result.psi_coeffs) == 51  # N + 1

        # Convolved coefficients
        assert result.convolved_coeffs is not None

    def test_decomposition_optional(self):
        """Decomposition is optional."""
        config = VZetaMomentConfig(T=500, Delta=1.0, N=50)

        # Without decomposition
        result1 = compute_vzeta_psi_moment(config, include_decomposition=False)
        assert result1.ratio_decomposition is None

        # With decomposition
        result2 = compute_vzeta_psi_moment(config, include_decomposition=True)
        assert result2.ratio_decomposition is not None


class TestDeltaSweep:
    """Tests for Delta parameter sweep."""

    def test_sweep_returns_arrays(self):
        """Delta sweep returns arrays."""
        T = 500
        deltas = np.array([0.5, 1.0, 2.0])

        out_deltas, moments = delta_sweep(T, deltas, use_optimal=True, N=30)

        assert len(out_deltas) == 3
        assert len(moments) == 3

    def test_sweep_moments_positive(self):
        """All swept moments are positive."""
        T = 500
        deltas = np.array([0.5, 1.0, 2.0])

        _, moments = delta_sweep(T, deltas, use_optimal=True, N=30)

        assert all(m > 0 for m in moments)

    def test_wider_bandwidth_changes_moment(self):
        """Wider bandwidth (smaller Delta) should change the moment."""
        T = 500
        deltas = np.array([0.5, 2.0])

        _, moments = delta_sweep(T, deltas, use_optimal=True, N=30)

        # Moments should differ for different bandwidths
        assert moments[0] != moments[1]


class TestGlobalLimitValidation:
    """Tests for validating the Delta->0 global limit."""

    @pytest.mark.slow
    def test_validate_optimal_runs(self):
        """Validation runs without error for optimal."""
        # Use small T and few deltas for speed
        passed, extrapolated, error = validate_global_limit(
            T=200, use_optimal=True, n_deltas=5, delta_max=5.0, rtol=0.5
        )

        assert passed in (True, False)  # Works with numpy bool
        assert extrapolated > 0
        assert error >= 0

    @pytest.mark.slow
    def test_validate_przz_runs(self):
        """Validation runs without error for PRZZ."""
        passed, extrapolated, error = validate_global_limit(
            T=200, use_optimal=False, n_deltas=5, delta_max=5.0, rtol=0.5
        )

        assert passed in (True, False)
        assert extrapolated > 0


class TestOptimalVsPRZZComparison:
    """Tests for comparing optimal and PRZZ moments."""

    def test_comparison_structure(self):
        """Comparison returns expected structure."""
        result = compare_optimal_vs_przz(T=500, Delta=1.0, N=50)

        assert 'T' in result
        assert 'Delta' in result
        assert 'optimal' in result
        assert 'przz' in result
        assert 'ratio' in result

        assert 'moment' in result['optimal']
        assert 'moment' in result['przz']

    def test_optimal_uses_correct_R(self):
        """Optimal uses R ≈ 1.15."""
        result = compare_optimal_vs_przz(T=500, Delta=1.0, N=50)

        assert abs(result['optimal']['R'] - 1.14976) < 0.01

    def test_przz_uses_correct_R(self):
        """PRZZ uses R ≈ 1.30."""
        result = compare_optimal_vs_przz(T=500, Delta=1.0, N=50)

        assert abs(result['przz']['R'] - 1.3036) < 0.01

    def test_ratio_computed(self):
        """Ratio is computed as optimal/przz."""
        result = compare_optimal_vs_przz(T=500, Delta=1.0, N=50)

        expected_ratio = result['optimal']['moment'] / result['przz']['moment']
        assert abs(result['ratio'] - expected_ratio) < 1e-10


class TestConvolvedCoefficients:
    """Tests for the convolved (Vζ · ψ) coefficients."""

    def test_convolved_c1_involves_b1_a1(self):
        """c[1] = b[1]·a[1] = 1·a[1] = a[1]."""
        config = VZetaMomentConfig(T=500, Delta=1.0, N=50)
        result = compute_vzeta_psi_moment(config, use_optimal=True)

        # b[1] = Q(0) = 1, so c[1] = a[1]
        assert abs(result.convolved_coeffs[1] - result.psi_coeffs[1]) < 1e-10

    def test_convolution_length(self):
        """Convolved length = M * N."""
        config = VZetaMomentConfig(T=500, Delta=1.0, N=50)
        result = compute_vzeta_psi_moment(config, use_optimal=True)

        M = result.vzeta_coeffs.M
        N = len(result.psi_coeffs) - 1
        expected_length = M * N + 1

        assert len(result.convolved_coeffs) == expected_length

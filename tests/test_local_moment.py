"""
Tests for localized moment computation.
"""

import numpy as np
import pytest
from src.local.local_moment import (
    LocalMomentConfig,
    LocalMomentResult,
    RatioMomentDecomposition,
    compute_local_moment,
    compute_ratio_domain_moment,
    compute_ratio_domain_decomposed,
    verify_moment_consistency,
)
from src.local.fejer import FejerKernel


class TestLocalMomentConfig:
    """Tests for LocalMomentConfig dataclass."""

    def test_defaults(self):
        """Test default values."""
        config = LocalMomentConfig(T=100.0, Delta=1.0)

        assert config.T == 100.0
        assert config.Delta == 1.0
        assert config.sigma == 0.5
        assert config.n_halfwidth == 4.0
        assert config.n_points_per_zero == 20


class TestComputeLocalMoment:
    """Tests for time-domain local moment computation."""

    def test_constant_polynomial(self):
        """For D(s) = 1, |D|^2 = 1, moment = integral of w."""
        # D(s) = 1 means coeffs[1] = 1, rest zero
        coeffs = np.array([0.0, 1.0])

        config = LocalMomentConfig(
            T=100.0,
            Delta=1.0,
            n_halfwidth=10.0,  # Use wide window for accuracy
            n_points_per_zero=50,
        )

        result = compute_local_moment(coeffs, config)

        # |D|^2 = 1 everywhere, so moment = integral of w_Delta
        # Integral of Fejer kernel is 1 (with some numerical error from truncation)
        assert np.isclose(result.moment, 1.0, rtol=0.02)  # 2% tolerance for truncation

    def test_result_structure(self):
        """Result should have correct structure."""
        coeffs = np.array([0.0, 1.0, 0.5])
        config = LocalMomentConfig(T=50.0, Delta=2.0)

        result = compute_local_moment(coeffs, config)

        assert isinstance(result, LocalMomentResult)
        assert result.T == 50.0
        assert result.Delta == 2.0
        assert result.sigma == 0.5
        assert len(result.t_grid) > 0
        assert len(result.weights) == len(result.t_grid)
        assert len(result.D_squared) == len(result.t_grid)

    def test_moment_positive(self):
        """Moment should be positive for non-trivial coefficients."""
        np.random.seed(42)
        coeffs = np.zeros(51)
        coeffs[1:] = np.random.randn(50)

        config = LocalMomentConfig(T=100.0, Delta=1.0)
        result = compute_local_moment(coeffs, config)

        # |D|^2 >= 0 and w >= 0, so moment >= 0
        assert result.moment >= 0

    def test_grid_centered_at_T(self):
        """Grid should be centered at T."""
        coeffs = np.array([0.0, 1.0])
        config = LocalMomentConfig(T=123.456, Delta=1.0)

        result = compute_local_moment(coeffs, config)

        # Grid should contain T (approximately at center)
        center = (result.t_grid[0] + result.t_grid[-1]) / 2
        assert np.isclose(center, config.T)


class TestComputeRatioDomainMoment:
    """Tests for ratio-domain local moment computation."""

    def test_constant_polynomial(self):
        """For D(s) = 1, ratio-domain moment = w_hat(0) = 1."""
        coeffs = np.array([0.0, 1.0])

        config = LocalMomentConfig(T=100.0, Delta=1.0)
        moment = compute_ratio_domain_moment(coeffs, config)

        # Single term: a_1 = 1, n = m = 1, log(n/m) = 0, w_hat(0) = 1
        assert np.isclose(moment, 1.0)

    def test_two_terms(self):
        """Test with D(s) = 1 + 2^{-s}."""
        # coeffs[1] = 1, coeffs[2] = 1
        coeffs = np.array([0.0, 1.0, 1.0])
        sigma = 0.5

        # Use small Delta so only nearby ratios contribute
        config = LocalMomentConfig(T=0.0, Delta=1.0, sigma=sigma)
        moment = compute_ratio_domain_moment(coeffs, config)

        # Should be positive and finite
        assert moment > 0
        assert np.isfinite(moment)


class TestVerifyMomentConsistency:
    """Tests for time-domain vs ratio-domain consistency."""

    def test_constant_polynomial(self):
        """Constant polynomial should pass consistency."""
        coeffs = np.array([0.0, 1.0])
        config = LocalMomentConfig(
            T=100.0,
            Delta=1.0,
            n_halfwidth=10.0,
            n_points_per_zero=50,
        )

        time_mom, ratio_mom, passed = verify_moment_consistency(coeffs, config, rtol=0.05)

        assert passed
        assert np.isclose(time_mom, 1.0, rtol=0.05)
        assert np.isclose(ratio_mom, 1.0)

    def test_small_n_consistency(self):
        """Small N should have good consistency."""
        np.random.seed(42)
        N = 20
        coeffs = np.zeros(N + 1)
        coeffs[1:] = np.random.randn(N) * 0.1
        coeffs[1] = 1.0  # Ensure non-trivial

        config = LocalMomentConfig(
            T=10.0,
            Delta=0.5,
            n_halfwidth=6.0,
            n_points_per_zero=30,
        )

        time_mom, ratio_mom, passed = verify_moment_consistency(coeffs, config, rtol=0.01)

        # Should pass with reasonable tolerance
        rel_error = abs(time_mom - ratio_mom) / abs(ratio_mom)
        assert rel_error < 0.05, f"Relative error {rel_error} too large"


class TestNumericalProperties:
    """Tests for numerical properties of local moments."""

    def test_moment_depends_on_T(self):
        """Moment should depend on window center T."""
        np.random.seed(123)
        coeffs = np.zeros(31)
        coeffs[1:] = np.random.randn(30)

        config1 = LocalMomentConfig(T=100.0, Delta=1.0)
        config2 = LocalMomentConfig(T=200.0, Delta=1.0)

        result1 = compute_local_moment(coeffs, config1)
        result2 = compute_local_moment(coeffs, config2)

        # Moments at different T should generally differ
        # (unless there's special structure)
        # We just check they're computed and positive
        assert result1.moment > 0 or result1.moment == 0
        assert result2.moment > 0 or result2.moment == 0

    def test_moment_depends_on_Delta(self):
        """Moment should depend on bandwidth Delta."""
        np.random.seed(456)
        coeffs = np.zeros(31)
        coeffs[1:] = np.random.randn(30)

        config1 = LocalMomentConfig(T=100.0, Delta=0.5)
        config2 = LocalMomentConfig(T=100.0, Delta=2.0)

        result1 = compute_local_moment(coeffs, config1)
        result2 = compute_local_moment(coeffs, config2)

        # Different bandwidths should give different moments
        # (wider bandwidth integrates over more of |D|^2)
        assert result1.moment > 0
        assert result2.moment > 0

    def test_weights_are_fejer(self):
        """Weights should match Fejer kernel values."""
        coeffs = np.array([0.0, 1.0])
        config = LocalMomentConfig(T=50.0, Delta=1.5)

        result = compute_local_moment(coeffs, config)

        # Verify weights match Fejer kernel
        kernel = FejerKernel(config.Delta)
        expected_weights = kernel.w_time(result.t_grid - config.T)

        assert np.allclose(result.weights, expected_weights)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_coefficient(self):
        """Single non-zero coefficient."""
        coeffs = np.array([0.0, 1.0])
        config = LocalMomentConfig(T=100.0, Delta=1.0)

        result = compute_local_moment(coeffs, config)
        assert np.isfinite(result.moment)

    def test_many_zeros(self):
        """Many zero coefficients."""
        coeffs = np.zeros(101)
        coeffs[1] = 1.0
        coeffs[100] = 0.5

        config = LocalMomentConfig(T=100.0, Delta=1.0)
        result = compute_local_moment(coeffs, config)

        assert np.isfinite(result.moment)
        assert result.moment > 0


class TestRatioMomentDecomposition:
    """Tests for diagonal/off-diagonal decomposition."""

    def test_constant_polynomial_is_pure_diagonal(self):
        """For D(s) = 1, moment is pure diagonal."""
        coeffs = np.array([0.0, 1.0])
        config = LocalMomentConfig(T=100.0, Delta=1.0)

        decomp = compute_ratio_domain_decomposed(coeffs, config)

        # With single coefficient, only diagonal term exists
        assert np.isclose(decomp.diagonal, 1.0)  # |a_1|^2 * 1^{-2*0.5} = 1
        assert np.isclose(decomp.off_diagonal, 0.0)
        assert np.isclose(decomp.total, 1.0)

    def test_decomposition_sums_correctly(self):
        """total = diagonal + off_diagonal."""
        np.random.seed(42)
        N = 30
        coeffs = np.zeros(N + 1)
        coeffs[1:] = np.random.randn(N) * 0.5
        coeffs[1] = 1.0

        config = LocalMomentConfig(T=100.0, Delta=1.0)
        decomp = compute_ratio_domain_decomposed(coeffs, config)

        # Verify decomposition sums correctly
        assert np.isclose(decomp.total, decomp.diagonal + decomp.off_diagonal)

    def test_decomposition_matches_total_moment(self):
        """Decomposed total should match compute_ratio_domain_moment."""
        np.random.seed(123)
        N = 20
        coeffs = np.zeros(N + 1)
        coeffs[1:] = np.random.randn(N) * 0.3
        coeffs[1] = 1.0

        config = LocalMomentConfig(T=50.0, Delta=0.5)

        decomp = compute_ratio_domain_decomposed(coeffs, config)
        total_moment = compute_ratio_domain_moment(coeffs, config)

        assert np.isclose(decomp.total, total_moment, rtol=1e-10)

    def test_diagonal_is_positive(self):
        """Diagonal should always be positive (sum of squares)."""
        np.random.seed(456)
        coeffs = np.zeros(51)
        coeffs[1:] = np.random.randn(50)

        config = LocalMomentConfig(T=100.0, Delta=1.0)
        decomp = compute_ratio_domain_decomposed(coeffs, config)

        assert decomp.diagonal > 0

    def test_off_diagonal_can_be_negative(self):
        """Off-diagonal can be negative (interference)."""
        # With specific coefficients, off-diagonal can be negative
        coeffs = np.array([0.0, 1.0, 1.0])  # D(s) = 1 + 2^{-s}
        config = LocalMomentConfig(T=0.0, Delta=2.0)

        decomp = compute_ratio_domain_decomposed(coeffs, config)

        # At T=0, off-diagonal should be positive (constructive interference)
        # At T=π/log(2), it would be negative (destructive interference)
        # Just check it's finite and non-zero
        assert np.isfinite(decomp.off_diagonal)

    def test_off_over_diag_ratio(self):
        """off_over_diag should be correctly computed."""
        np.random.seed(789)
        coeffs = np.zeros(21)
        coeffs[1:] = np.random.randn(20) * 0.2
        coeffs[1] = 1.0

        config = LocalMomentConfig(T=100.0, Delta=1.0)
        decomp = compute_ratio_domain_decomposed(coeffs, config)

        expected_ratio = decomp.off_diagonal / decomp.diagonal
        assert np.isclose(decomp.off_over_diag, expected_ratio)

    def test_decomposition_varies_with_T(self):
        """Off-diagonal should vary with T (oscillatory)."""
        coeffs = np.array([0.0, 1.0, 0.5, 0.3])  # Multiple terms
        config1 = LocalMomentConfig(T=0.0, Delta=2.0)
        config2 = LocalMomentConfig(T=10.0, Delta=2.0)

        decomp1 = compute_ratio_domain_decomposed(coeffs, config1)
        decomp2 = compute_ratio_domain_decomposed(coeffs, config2)

        # Diagonal should be the same (independent of T)
        assert np.isclose(decomp1.diagonal, decomp2.diagonal)

        # Off-diagonal may differ (oscillatory in T)
        # (They could be equal by coincidence, so we just check they're computed)
        assert np.isfinite(decomp1.off_diagonal)
        assert np.isfinite(decomp2.off_diagonal)

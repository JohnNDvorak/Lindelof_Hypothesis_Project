"""
Tests for Dirichlet polynomial evaluator.
"""

import numpy as np
import pytest
from src.local.dirichlet_poly import (
    evaluate_brute,
    evaluate_incremental,
    evaluate_dirichlet_poly,
    DirichletPolyResult,
)


class TestEvaluateBrute:
    """Tests for brute-force Dirichlet polynomial evaluation."""

    def test_simple_polynomial(self):
        """Test D(s) = 1 + 2^{-s}."""
        # coeffs[1] = 1, coeffs[2] = 1
        coeffs = np.array([0.0, 1.0, 1.0])
        t_grid = np.array([0.0])
        sigma = 0.5

        result = evaluate_brute(coeffs, t_grid, sigma)

        # D(0.5) = 1 * 1^{-0.5} + 1 * 2^{-0.5} = 1 + 1/sqrt(2)
        expected = 1.0 + 2**(-0.5)
        assert np.isclose(result.values[0], expected)

    def test_single_term(self):
        """Test D(s) = n^{-s} for single n."""
        n = 5
        coeffs = np.zeros(n + 1)
        coeffs[n] = 1.0

        t = 10.0
        sigma = 0.5
        s = sigma + 1j * t

        result = evaluate_brute(coeffs, np.array([t]), sigma)

        # D(s) = n^{-s} = n^{-sigma} * exp(-i*t*log(n))
        expected = n**(-sigma) * np.exp(-1j * t * np.log(n))
        assert np.isclose(result.values[0], expected)

    def test_abs_squared(self):
        """abs_squared should be |values|^2."""
        coeffs = np.array([0.0, 1.0, 0.5, 0.25])
        t_grid = np.linspace(-10, 10, 21)

        result = evaluate_brute(coeffs, t_grid, sigma=0.5)

        expected_squared = np.abs(result.values) ** 2
        assert np.allclose(result.abs_squared, expected_squared)

    def test_t_grid_shape(self):
        """Output should match t_grid shape."""
        coeffs = np.array([0.0, 1.0, 2.0, 3.0])
        t_grid = np.linspace(0, 100, 51)

        result = evaluate_brute(coeffs, t_grid, sigma=0.5)

        assert result.t_grid.shape == t_grid.shape
        assert result.values.shape == t_grid.shape
        assert result.abs_squared.shape == t_grid.shape


class TestEvaluateIncremental:
    """Tests for incremental phase Dirichlet polynomial evaluation."""

    def test_matches_brute(self):
        """Incremental should match brute-force exactly."""
        np.random.seed(42)
        N = 50
        coeffs = np.zeros(N + 1)
        coeffs[1:] = np.random.randn(N)

        T = 100.0
        dt = 0.1
        K = 20

        # Brute evaluation
        t_grid = np.arange(-K, K + 1) * dt + T
        brute_result = evaluate_brute(coeffs, t_grid, sigma=0.5)

        # Incremental evaluation
        incr_result = evaluate_incremental(coeffs, T, dt, K, sigma=0.5)

        # Should match
        assert np.allclose(incr_result.t_grid, brute_result.t_grid)
        assert np.allclose(incr_result.values, brute_result.values, rtol=1e-10)
        assert np.allclose(incr_result.abs_squared, brute_result.abs_squared, rtol=1e-10)

    def test_grid_structure(self):
        """Grid should be centered at T with 2K+1 points."""
        coeffs = np.array([0.0, 1.0, 1.0])
        T = 50.0
        dt = 0.5
        K = 10

        result = evaluate_incremental(coeffs, T, dt, K)

        # Grid should have 2K+1 = 21 points
        assert len(result.t_grid) == 21

        # Center should be at T
        assert np.isclose(result.t_grid[K], T)

        # Grid spacing
        assert np.allclose(np.diff(result.t_grid), dt)


class TestEvaluateDirichletPoly:
    """Tests for unified interface."""

    def test_auto_mode_small_n(self):
        """Auto mode should use brute for small N."""
        coeffs = np.array([0.0, 1.0, 1.0, 1.0])  # N = 3
        result = evaluate_dirichlet_poly(
            coeffs, T=10.0, dt=0.1, K=5, mode='auto'
        )
        assert result is not None

    def test_auto_mode_large_n(self):
        """Auto mode should use incremental for large N."""
        coeffs = np.zeros(2001)  # N = 2000
        coeffs[1:] = 1.0
        result = evaluate_dirichlet_poly(
            coeffs, T=10.0, dt=0.1, K=5, mode='auto'
        )
        assert result is not None

    def test_explicit_t_grid(self):
        """Should work with explicit t_grid."""
        coeffs = np.array([0.0, 1.0, 2.0])
        t_grid = np.array([0.0, 1.0, 2.0])

        result = evaluate_dirichlet_poly(coeffs, t_grid=t_grid)
        assert np.allclose(result.t_grid, t_grid)

    def test_mode_brute(self):
        """Explicit brute mode."""
        coeffs = np.zeros(101)
        coeffs[1:] = 1.0

        result = evaluate_dirichlet_poly(
            coeffs, T=10.0, dt=0.1, K=5, mode='brute'
        )
        assert result is not None

    def test_mode_incremental(self):
        """Explicit incremental mode."""
        coeffs = np.zeros(101)
        coeffs[1:] = 1.0

        result = evaluate_dirichlet_poly(
            coeffs, T=10.0, dt=0.1, K=5, mode='incremental'
        )
        assert result is not None

    def test_incremental_requires_params(self):
        """Incremental mode should require T, dt, K."""
        coeffs = np.array([0.0, 1.0])

        with pytest.raises(ValueError):
            evaluate_dirichlet_poly(coeffs, mode='incremental')

    def test_brute_requires_grid(self):
        """Brute mode should require t_grid or (T, dt, K)."""
        coeffs = np.array([0.0, 1.0])

        with pytest.raises(ValueError):
            evaluate_dirichlet_poly(coeffs, mode='brute')


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_incremental_long_sweep(self):
        """Incremental should remain stable over long sweeps."""
        np.random.seed(123)
        N = 100
        coeffs = np.zeros(N + 1)
        coeffs[1:] = np.random.randn(N)

        T = 1000.0
        dt = 0.01
        K = 500  # 1001 points

        # Both methods
        t_grid = np.arange(-K, K + 1) * dt + T
        brute = evaluate_brute(coeffs, t_grid, sigma=0.5)
        incr = evaluate_incremental(coeffs, T, dt, K, sigma=0.5)

        # Should match to good precision even for 1001 points
        assert np.allclose(incr.values, brute.values, rtol=1e-8)

    def test_different_sigma(self):
        """Should work with different sigma values."""
        coeffs = np.array([0.0, 1.0, 1.0, 1.0])

        for sigma in [0.0, 0.25, 0.5, 0.75, 1.0]:
            result = evaluate_dirichlet_poly(
                coeffs, t_grid=np.array([0.0, 1.0]), sigma=sigma
            )
            assert np.all(np.isfinite(result.values))

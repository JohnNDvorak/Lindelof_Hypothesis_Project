"""
Tests for the Fejer band-limited window kernel.
"""

import numpy as np
import pytest
from src.local.fejer import FejerKernel, delta_from_first_zero


class TestFejerKernel:
    """Tests for FejerKernel class."""

    def test_first_zero_location(self):
        """First zero of w(t) should be at t = 2*pi/Delta."""
        Delta = 1.0
        kernel = FejerKernel(Delta)

        # First zero at 2*pi/Delta
        first_zero = kernel.first_zero
        assert np.isclose(first_zero, 2 * np.pi / Delta)

        # w(first_zero) should be zero (sinc^2 zero)
        w_at_zero = kernel.w_time(first_zero)
        assert np.isclose(w_at_zero, 0.0, atol=1e-15)

    def test_normalization_at_origin(self):
        """w(0) = Delta / (2*pi)."""
        Delta = 2.5
        kernel = FejerKernel(Delta)

        w_at_0 = kernel.w_time(0.0)
        expected = Delta / (2 * np.pi)
        assert np.isclose(w_at_0, expected)

    def test_frequency_at_origin(self):
        """w_hat(0) = 1."""
        kernel = FejerKernel(Delta=3.0)
        assert np.isclose(kernel.w_freq(0.0), 1.0)

    def test_frequency_triangle(self):
        """w_hat(xi) = max(0, 1 - |xi|/Delta)."""
        Delta = 2.0
        kernel = FejerKernel(Delta)

        # At xi = Delta/2, w_hat = 0.5
        assert np.isclose(kernel.w_freq(Delta / 2), 0.5)
        assert np.isclose(kernel.w_freq(-Delta / 2), 0.5)

        # At xi = Delta, w_hat = 0
        assert np.isclose(kernel.w_freq(Delta), 0.0)
        assert np.isclose(kernel.w_freq(-Delta), 0.0)

        # Beyond Delta, w_hat = 0
        assert kernel.w_freq(Delta * 1.5) == 0.0
        assert kernel.w_freq(-Delta * 2.0) == 0.0

    def test_positivity(self):
        """w(t) >= 0 for all t (Fejer kernel is non-negative)."""
        kernel = FejerKernel(Delta=1.0)

        # Test many random points
        t = np.linspace(-100, 100, 10000)
        w = kernel.w_time(t)
        assert np.all(w >= -1e-15)  # Allow tiny numerical errors

    def test_from_first_zero(self):
        """Create kernel with specified first zero location."""
        L = 5.0
        kernel = FejerKernel.from_first_zero(L)

        # First zero should be at L
        assert np.isclose(kernel.first_zero, L)

        # Delta = 2*pi/L
        assert np.isclose(kernel.Delta, 2 * np.pi / L)

    def test_from_first_zero_invalid(self):
        """from_first_zero should reject non-positive L."""
        with pytest.raises(ValueError):
            FejerKernel.from_first_zero(0.0)
        with pytest.raises(ValueError):
            FejerKernel.from_first_zero(-1.0)

    def test_array_input(self):
        """w_time and w_freq should handle array inputs."""
        kernel = FejerKernel(Delta=1.0)

        t = np.array([0.0, 1.0, 2.0, 3.0])
        w = kernel.w_time(t)
        assert w.shape == t.shape

        xi = np.array([-0.5, 0.0, 0.5, 1.0])
        w_hat = kernel.w_freq(xi)
        assert w_hat.shape == xi.shape

    def test_decay_rate(self):
        """w(t) ~ t^{-2} for large |t|."""
        kernel = FejerKernel(Delta=1.0)

        # At large t, w(t) should decay like 1/t^2
        # Need to sample at exact zeros + small offset to avoid zero crossings
        # w(t) = (Delta/2pi) * sinc^2(Delta*t/2pi)
        # The envelope decays like 1/t^2

        # Check that w decreases as t increases (roughly as 1/t^2)
        t_values = np.array([50.0, 100.0, 200.0, 400.0])
        w_values = kernel.w_time(t_values)

        # Envelope should roughly follow C/t^2, so w*t^2 should be roughly constant
        # But sinc oscillates, so we check the envelope bound instead
        # w(t) <= (Delta/2pi) * 1/(Delta*t/2pi)^2 = 4*pi / (Delta * t^2)
        envelope = 4 * np.pi / (kernel.Delta * t_values**2)
        assert np.all(w_values <= envelope * 1.1)  # Allow 10% margin

    def test_effective_support(self):
        """effective_support returns reasonable truncation point."""
        kernel = FejerKernel(Delta=1.0)

        T = kernel.effective_support(tolerance=1e-4)

        # w(T) / w(0) should be approximately tolerance
        ratio = kernel.w_time(T) / kernel.w_time(0.0)
        assert ratio < 1e-3  # Should be small


class TestDeltaFromFirstZero:
    """Tests for delta_from_first_zero function."""

    def test_basic(self):
        """Delta = 2*pi/L."""
        L = 3.0
        Delta = delta_from_first_zero(L)
        assert np.isclose(Delta, 2 * np.pi / L)

    def test_invalid(self):
        """Should reject non-positive L."""
        with pytest.raises(ValueError):
            delta_from_first_zero(0.0)
        with pytest.raises(ValueError):
            delta_from_first_zero(-5.0)


class TestNormalization:
    """Test normalization properties of the Fejer kernel."""

    def test_integral_approximation(self):
        """Integral of w(t) over large interval should approximate 1."""
        kernel = FejerKernel(Delta=1.0)

        # Numerical integration over large interval
        t = np.linspace(-1000, 1000, 100001)
        dt = t[1] - t[0]
        w = kernel.w_time(t)
        integral = np.sum(w) * dt

        # Should be close to 1 (Parseval: integral of w = w_hat(0) = 1)
        assert np.isclose(integral, 1.0, rtol=0.01)

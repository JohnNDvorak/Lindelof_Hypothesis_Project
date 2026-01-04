"""
Tests for mollifier coefficient generators.
"""

import numpy as np
import pytest
from src.local.mollifier_coeffs import (
    compute_u_array,
    compute_mollifier_coeffs,
    load_optimal_polynomials,
    load_przz_polynomials,
    MollifierCoeffs,
)
from src.local.sieve import compute_sieve_arrays


class TestComputeUArray:
    """Tests for u_n = log(N/n) / log(N)."""

    def test_boundary_values(self):
        """u[1] = 1, u[N] = 0."""
        N = 100
        u = compute_u_array(N)

        assert np.isclose(u[1], 1.0)
        assert np.isclose(u[N], 0.0)

    def test_monotonicity(self):
        """u should be monotonically decreasing."""
        N = 100
        u = compute_u_array(N)

        # u[1:] should be strictly decreasing
        assert np.all(np.diff(u[1:]) < 0)

    def test_range(self):
        """All u values should be in [0, 1]."""
        N = 1000
        u = compute_u_array(N)

        assert np.all(u[1:] >= 0)
        assert np.all(u[1:] <= 1)

    def test_specific_value(self):
        """Test a specific u value."""
        N = 100
        u = compute_u_array(N)

        # u[10] = log(100/10) / log(100) = log(10) / log(100) = 0.5
        assert np.isclose(u[10], 0.5)


class TestLoadPolynomials:
    """Tests for polynomial loading functions."""

    def test_load_optimal(self):
        """load_optimal_polynomials returns valid polynomials."""
        P1, P2, P3 = load_optimal_polynomials()

        # Check P1 constraints: P1(0) = 0, P1(1) = 1
        assert np.isclose(P1.eval(np.array([0.0]))[0], 0.0)
        assert np.isclose(P1.eval(np.array([1.0]))[0], 1.0)

        # Check P2, P3 constraints: P(0) = 0
        assert np.isclose(P2.eval(np.array([0.0]))[0], 0.0)
        assert np.isclose(P3.eval(np.array([0.0]))[0], 0.0)

    def test_load_przz(self):
        """load_przz_polynomials returns valid polynomials."""
        P1, P2, P3 = load_przz_polynomials()

        # Check P1 constraints
        assert np.isclose(P1.eval(np.array([0.0]))[0], 0.0)
        assert np.isclose(P1.eval(np.array([1.0]))[0], 1.0)

        # Check P2, P3 constraints
        assert np.isclose(P2.eval(np.array([0.0]))[0], 0.0)
        assert np.isclose(P3.eval(np.array([0.0]))[0], 0.0)


class TestComputeMollifierCoeffs:
    """Tests for compute_mollifier_coeffs function."""

    def test_basic_psi1(self):
        """Compute psi_1 coefficients only."""
        coeffs = compute_mollifier_coeffs(N=100, which=(True, False, False))

        assert isinstance(coeffs, MollifierCoeffs)
        assert coeffs.N == 100
        assert coeffs.a1 is not None
        assert coeffs.a2 is None
        assert coeffs.a3 is None
        assert len(coeffs.a1) == 101

    def test_a1_at_one(self):
        """a1[1] = mu(1) * P1(u[1]) = 1 * P1(1) = 1."""
        coeffs = compute_mollifier_coeffs(N=100, which=(True, False, False))

        # a1[1] = mu(1) * P1(1) = 1 * 1 = 1
        assert np.isclose(coeffs.a1[1], 1.0)

    def test_a1_at_prime(self):
        """a1[p] = mu(p) * P1(u_p) = -P1(u_p)."""
        N = 100
        coeffs = compute_mollifier_coeffs(N=N, which=(True, False, False))
        P1, _, _ = load_optimal_polynomials()

        # For p = 2
        u_2 = np.log(N / 2) / np.log(N)
        expected = -P1.eval(np.array([u_2]))[0]
        assert np.isclose(coeffs.a1[2], expected)

        # For p = 7
        u_7 = np.log(N / 7) / np.log(N)
        expected = -P1.eval(np.array([u_7]))[0]
        assert np.isclose(coeffs.a1[7], expected)

    def test_a1_sparsity(self):
        """a1[n] = 0 when mu(n) = 0."""
        coeffs = compute_mollifier_coeffs(N=100, which=(True, False, False))

        # n = 4, 8, 9, 12, ... have mu(n) = 0
        assert coeffs.a1[4] == 0.0
        assert coeffs.a1[8] == 0.0
        assert coeffs.a1[9] == 0.0
        assert coeffs.a1[12] == 0.0

    def test_use_optimal_vs_przz(self):
        """use_optimal flag should affect results."""
        coeffs_opt = compute_mollifier_coeffs(N=100, which=(True, False, False), use_optimal=True)
        coeffs_przz = compute_mollifier_coeffs(N=100, which=(True, False, False), use_optimal=False)

        # Coefficients should differ (different polynomials)
        assert not np.allclose(coeffs_opt.a1, coeffs_przz.a1)

    def test_psi2_coefficients(self):
        """Compute psi_2 coefficients."""
        coeffs = compute_mollifier_coeffs(N=100, which=(False, True, False))

        assert coeffs.a1 is None
        assert coeffs.a2 is not None
        assert coeffs.a3 is None

    def test_psi3_coefficients(self):
        """Compute psi_3 coefficients."""
        coeffs = compute_mollifier_coeffs(N=100, which=(False, False, True))

        assert coeffs.a1 is None
        assert coeffs.a2 is None
        assert coeffs.a3 is not None

    def test_all_coefficients(self):
        """Compute all psi coefficients."""
        coeffs = compute_mollifier_coeffs(N=100, which=(True, True, True))

        assert coeffs.a1 is not None
        assert coeffs.a2 is not None
        assert coeffs.a3 is not None


class TestCoefficientsShape:
    """Tests for coefficient array shapes."""

    def test_shapes(self):
        """All arrays should have length N+1."""
        N = 50
        coeffs = compute_mollifier_coeffs(N=N, which=(True, True, True))

        assert len(coeffs.u) == N + 1
        assert len(coeffs.a1) == N + 1
        assert len(coeffs.a2) == N + 1
        assert len(coeffs.a3) == N + 1

    def test_index_zero_unused(self):
        """Index 0 should be unused (zero)."""
        coeffs = compute_mollifier_coeffs(N=50, which=(True, True, True))

        # u[0] and a[0] should be 0 (unused)
        assert coeffs.u[0] == 0.0
        assert coeffs.a1[0] == 0.0
        assert coeffs.a2[0] == 0.0
        assert coeffs.a3[0] == 0.0

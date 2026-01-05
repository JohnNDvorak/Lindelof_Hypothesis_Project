"""
Tests for V-operator implementation.

Tests cover:
- V[ζ] coefficient computation: b_m = Q(log(m)/log(M))
- PRZZ constraint: Q(0) = 1 implies b_1 = 1
- Dirichlet convolution: c_k = Σ_{mn=k} b_m · a_n
- Loading optimal and PRZZ Q polynomials
"""

import numpy as np
import pytest

from src.local.v_operator import (
    VZetaCoeffs,
    compute_vzeta_coeffs,
    dirichlet_convolve,
    compute_vzeta_psi_coeffs,
    load_optimal_Q,
    load_przz_Q,
    load_optimal_polynomials_full,
)
from src.polynomials import Polynomial


class TestVZetaCoeffs:
    """Tests for V[ζ] coefficient computation."""

    def test_b1_equals_Q0(self):
        """b[1] = Q(log(1)/log(M)) = Q(0) should equal Q(0)."""
        # Simple Q(x) = 1 - x
        Q = Polynomial(coeffs=np.array([1.0, -1.0]))
        M = 100
        vzeta = compute_vzeta_coeffs(M, Q)

        # b[1] = Q(0) = 1
        assert abs(vzeta.b[1] - 1.0) < 1e-10

    def test_bM_equals_Q1(self):
        """b[M] = Q(log(M)/log(M)) = Q(1)."""
        # Q(x) = 1 - x  =>  Q(1) = 0
        Q = Polynomial(coeffs=np.array([1.0, -1.0]))
        M = 100
        vzeta = compute_vzeta_coeffs(M, Q)

        # b[M] = Q(1) = 0
        assert abs(vzeta.b[M] - 0.0) < 1e-10

    def test_coefficient_shape(self):
        """V[ζ] coefficient array has correct shape."""
        Q = Polynomial(coeffs=np.array([1.0]))
        M = 50
        vzeta = compute_vzeta_coeffs(M, Q)

        assert len(vzeta.b) == M + 1
        assert vzeta.b[0] == 0  # Unused slot
        assert vzeta.M == M

    def test_monotonic_for_decreasing_Q(self):
        """For Q(x) = 1 - x, coefficients b[m] should decrease with m."""
        Q = Polynomial(coeffs=np.array([1.0, -1.0]))
        M = 100
        vzeta = compute_vzeta_coeffs(M, Q)

        # b[m] = Q(log(m)/log(M)) = 1 - log(m)/log(M)
        # This decreases as m increases
        for m in range(2, M):
            assert vzeta.b[m] > vzeta.b[m + 1]


class TestDirichletConvolve:
    """Tests for Dirichlet convolution."""

    def test_identity_convolution(self):
        """Convolution with delta at 1 is identity."""
        # b = [0, 1, 0, 0, ...]  (delta at n=1)
        b = np.zeros(5)
        b[1] = 1.0

        # a = [0, 1, 2, 3, 4]
        a = np.arange(5, dtype=np.float64)

        c = dirichlet_convolve(b, a)

        # c[k] = Σ_{mn=k} b[m]·a[n] = a[k] since b[m]=0 except b[1]=1
        for k in range(1, 5):
            assert abs(c[k] - a[k]) < 1e-10

    def test_convolution_at_1(self):
        """c[1] = b[1]·a[1]."""
        b = np.array([0.0, 2.0, 3.0, 4.0])
        a = np.array([0.0, 5.0, 6.0, 7.0])

        c = dirichlet_convolve(b, a)

        assert abs(c[1] - 2.0 * 5.0) < 1e-10

    def test_convolution_at_prime(self):
        """c[p] = b[1]·a[p] + b[p]·a[1] for prime p."""
        b = np.array([0.0, 2.0, 3.0, 4.0])
        a = np.array([0.0, 5.0, 6.0, 7.0])

        c = dirichlet_convolve(b, a)

        # c[2] = b[1]·a[2] + b[2]·a[1] = 2·6 + 3·5 = 27
        assert abs(c[2] - 27.0) < 1e-10

        # c[3] = b[1]·a[3] + b[3]·a[1] = 2·7 + 4·5 = 34
        assert abs(c[3] - 34.0) < 1e-10

    def test_convolution_at_composite(self):
        """c[6] = Σ_{d|6} b[d]·a[6/d]."""
        b = np.zeros(7)
        b[1] = 1.0
        b[2] = 2.0
        b[3] = 3.0
        b[6] = 6.0

        a = np.zeros(7)
        a[1] = 1.0
        a[2] = 2.0
        a[3] = 3.0
        a[6] = 6.0

        c = dirichlet_convolve(b, a)

        # c[6] = b[1]·a[6] + b[2]·a[3] + b[3]·a[2] + b[6]·a[1]
        #      = 1·6 + 2·3 + 3·2 + 6·1 = 6 + 6 + 6 + 6 = 24
        assert abs(c[6] - 24.0) < 1e-10

    def test_convolution_length(self):
        """Output length is M*N + 1."""
        M, N = 10, 20
        b = np.zeros(M + 1)
        a = np.zeros(N + 1)
        b[1] = 1.0
        a[1] = 1.0

        c = dirichlet_convolve(b, a)

        assert len(c) == M * N + 1


class TestLoadPolynomials:
    """Tests for loading Q polynomials."""

    def test_optimal_Q_loads(self):
        """Optimal Q polynomial loads successfully."""
        Q, R = load_optimal_Q()

        assert isinstance(Q, Polynomial)
        assert isinstance(R, float)
        # Optimal R should be around 1.15
        assert 1.0 < R < 1.3

    def test_optimal_Q_at_zero_is_one(self):
        """Q(0) = 1 (PRZZ constraint)."""
        Q, _ = load_optimal_Q()

        # Q(0) should be 1 (first coefficient in monomial form)
        assert abs(Q.eval(0.0) - 1.0) < 1e-10

    def test_przz_Q_loads(self):
        """PRZZ Q polynomial loads successfully."""
        Q, R = load_przz_Q()

        assert isinstance(Q, Polynomial)
        assert isinstance(R, float)
        # PRZZ R = 1.3036
        assert abs(R - 1.3036) < 0.01

    def test_przz_Q_at_zero_is_one(self):
        """PRZZ Q(0) = 1."""
        Q, _ = load_przz_Q()

        assert abs(Q.eval(0.0) - 1.0) < 1e-10

    def test_optimal_polynomials_full(self):
        """Full optimal polynomial set loads correctly."""
        data = load_optimal_polynomials_full()

        assert 'P1_tilde' in data
        assert 'P2_tilde' in data
        assert 'P3_tilde' in data
        assert 'Q_monomial' in data
        assert 'R' in data

        # Check shapes
        assert len(data['P1_tilde']) == 4
        assert len(data['Q_monomial']) == 6

        # Check R value
        assert abs(data['R'] - 1.14976) < 0.001


class TestVZetaPsiCoeffs:
    """Tests for the combined V[ζ]·ψ convolution."""

    def test_output_shape(self):
        """Convolved coefficients have correct shape."""
        Q = Polynomial(coeffs=np.array([1.0, -1.0]))
        M = 10
        N = 20

        vzeta = compute_vzeta_coeffs(M, Q)
        psi = np.zeros(N + 1)
        psi[1] = 1.0

        c = compute_vzeta_psi_coeffs(vzeta, psi)

        assert len(c) == M * N + 1

    def test_with_delta_psi(self):
        """Convolution with delta psi gives V[ζ] coefficients."""
        Q = Polynomial(coeffs=np.array([1.0, -0.5]))
        M = 10

        vzeta = compute_vzeta_coeffs(M, Q)

        # psi = delta at 1
        psi = np.zeros(M + 1)
        psi[1] = 1.0

        c = compute_vzeta_psi_coeffs(vzeta, psi)

        # c[k] = b[k] for k ≤ M
        for k in range(1, M + 1):
            assert abs(c[k] - vzeta.b[k]) < 1e-10


class TestPRZZConstraint:
    """Tests verifying the PRZZ Q(0)=1 constraint."""

    def test_optimal_b1_is_one(self):
        """Optimal V[ζ] has b[1] = Q(0) = 1."""
        Q, _ = load_optimal_Q()
        M = 100
        vzeta = compute_vzeta_coeffs(M, Q)

        assert abs(vzeta.b[1] - 1.0) < 1e-10

    def test_przz_b1_is_one(self):
        """PRZZ V[ζ] has b[1] = Q(0) = 1."""
        Q, _ = load_przz_Q()
        M = 100
        vzeta = compute_vzeta_coeffs(M, Q)

        assert abs(vzeta.b[1] - 1.0) < 1e-10

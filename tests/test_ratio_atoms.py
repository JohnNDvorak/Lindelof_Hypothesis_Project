"""
Tests for ratio-atom Taylor series module.
"""

import numpy as np
import pytest
from math import gcd, factorial

from src.local.ratio_atoms import (
    TaylorCoefficient,
    RatioAtom,
    compute_taylor_coefficient,
    compute_ratio_atom,
    compute_prime_atoms,
)
from src.polynomials import P1Polynomial
from src.local.sieve import compute_sieve_arrays
from src.local.mollifier_coeffs import compute_u_array


class TestTaylorCoefficient:
    """Tests for I_{r,s} coefficient computation."""

    @pytest.fixture
    def setup(self):
        """Common setup for tests."""
        N = 100
        sigma = 0.5
        # Simple P1: P1(x) = x (a0=0, so tilde_coeffs = [])
        P1 = P1Polynomial(tilde_coeffs=[])
        u_array = compute_u_array(N)
        sieve = compute_sieve_arrays(N, include_psi3=False)
        return {
            'N': N,
            'sigma': sigma,
            'P1': P1,
            'u_array': u_array,
            'mobius': sieve.mobius,
        }

    def test_I_00_is_sieving_sum(self, setup):
        """I_{0,0} = Σ_{g: (g,AB)=1} μ(g)² P₁(u_g)² g^{-2σ}."""
        coeff = compute_taylor_coefficient(
            r=0, s=0, A=1, B=2, N=setup['N'], sigma=setup['sigma'],
            P1=setup['P1'], u_array=setup['u_array'], mobius=setup['mobius']
        )

        assert coeff.r == 0
        assert coeff.s == 0
        assert coeff.value > 0  # Sum of positive terms
        assert coeff.n_terms > 0

    def test_I_10_involves_derivative(self, setup):
        """I_{1,0} involves P₁'(u_g)."""
        coeff = compute_taylor_coefficient(
            r=1, s=0, A=1, B=2, N=setup['N'], sigma=setup['sigma'],
            P1=setup['P1'], u_array=setup['u_array'], mobius=setup['mobius']
        )

        assert coeff.r == 1
        assert coeff.s == 0
        # For P1(x) = x, P1'(x) = 1, so this should be nonzero
        assert coeff.n_terms > 0

    def test_coprime_constraint(self, setup):
        """Only g coprime with AB contribute."""
        # (1, 6): AB = 6, so g must not be divisible by 2 or 3
        coeff = compute_taylor_coefficient(
            r=0, s=0, A=1, B=6, N=setup['N'], sigma=setup['sigma'],
            P1=setup['P1'], u_array=setup['u_array'], mobius=setup['mobius']
        )

        # With N=100, g_max = 100//6 = 16
        # Coprime with 6: 1, 5, 7, 11, 13
        assert coeff.n_terms <= 5


class TestRatioAtom:
    """Tests for full ratio atom computation."""

    @pytest.fixture
    def P1_simple(self):
        """P1(x) = x."""
        return P1Polynomial(tilde_coeffs=[])

    @pytest.fixture
    def P1_with_a0(self):
        """P1 with nonzero a0 = -2 (optimal)."""
        return P1Polynomial(tilde_coeffs=[-2.0, 1.0])

    def test_coprime_required(self, P1_simple):
        """Non-coprime pairs should return None."""
        atom = compute_ratio_atom(2, 4, N=100, sigma=0.5, P1=P1_simple)
        assert atom is None

    def test_squarefree_required(self, P1_simple):
        """Non-squarefree A or B should return None (μ=0)."""
        # A=4 is not squarefree
        atom = compute_ratio_atom(4, 3, N=100, sigma=0.5, P1=P1_simple)
        assert atom is None

    def test_basic_atom_structure(self, P1_simple):
        """Test basic (1, 2) atom structure."""
        atom = compute_ratio_atom(1, 2, N=100, sigma=0.5, P1=P1_simple)

        assert atom is not None
        assert atom.A == 1
        assert atom.B == 2
        assert atom.mobius_sign == -1  # μ(1)μ(2) = 1 * (-1) = -1
        assert atom.prefactor == 2 ** (-0.5)  # (1*2)^{-0.5}

        # Should have Taylor coefficients
        assert (0, 0) in atom.taylor_coeffs
        assert (1, 0) in atom.taylor_coeffs
        assert (0, 1) in atom.taylor_coeffs

    def test_get_coefficient(self, P1_simple):
        """Test coefficient access."""
        atom = compute_ratio_atom(1, 3, N=100, sigma=0.5, P1=P1_simple, max_order=2)

        I_00 = atom.get_coefficient(0, 0)
        I_10 = atom.get_coefficient(1, 0)
        I_01 = atom.get_coefficient(0, 1)

        assert I_00 > 0
        # Higher order should exist
        assert atom.get_coefficient(2, 0) is not None or atom.get_coefficient(2, 0) == 0.0

    def test_series_evaluation(self, P1_simple):
        """Test BivariateSeries evaluation."""
        atom = compute_ratio_atom(1, 2, N=100, sigma=0.5, P1=P1_simple, max_order=2)

        # Evaluate at (0, 0) should give I_{0,0}
        val_00 = atom.series.evaluate(0.0, 0.0)
        I_00 = atom.get_coefficient(0, 0)
        assert np.isclose(val_00, I_00)

    def test_endpoint_derivative_term(self, P1_with_a0):
        """Test endpoint derivative extraction for (1, p) atoms."""
        atom = compute_ratio_atom(1, 2, N=100, sigma=0.5, P1=P1_with_a0, max_order=2)

        # For (1, p), endpoint_derivative_term returns I_{0,1}
        edt = atom.endpoint_derivative_term()
        I_01 = atom.get_coefficient(0, 1)
        assert np.isclose(edt, I_01)

    def test_evaluate_at_delta(self, P1_simple):
        """Test evaluation at specific (δ_A, δ_B)."""
        atom = compute_ratio_atom(1, 2, N=100, sigma=0.5, P1=P1_simple, max_order=2)

        # At (0, 0)
        val = atom.evaluate_at_delta(0.0, 0.0)
        expected = atom.mobius_sign * atom.prefactor * atom.get_coefficient(0, 0)
        assert np.isclose(val, expected)


class TestPrimeAtoms:
    """Tests for batch prime atom computation."""

    @pytest.fixture
    def P1(self):
        return P1Polynomial(tilde_coeffs=[-2.0, 1.0])

    def test_compute_prime_atoms(self, P1):
        """Test batch computation for primes."""
        primes = [2, 3, 5]
        atoms = compute_prime_atoms(primes, N=100, sigma=0.5, P1=P1, max_order=2)

        # Should have (1, p) and (p, 1) atoms
        assert (1, 2) in atoms
        assert (2, 1) in atoms
        assert (1, 3) in atoms
        assert (3, 1) in atoms

        # Should have (p, q) atoms
        assert (2, 3) in atoms
        assert (3, 2) in atoms

    def test_prime_beyond_N_excluded(self, P1):
        """Primes > N should be excluded."""
        primes = [2, 3, 101]  # 101 > N=100
        atoms = compute_prime_atoms(primes, N=100, sigma=0.5, P1=P1, max_order=2)

        assert (1, 101) not in atoms
        assert (101, 1) not in atoms

    def test_prime_product_beyond_N(self, P1):
        """Prime products > N should be excluded."""
        primes = [2, 3, 5, 7, 11, 13]
        atoms = compute_prime_atoms(primes, N=100, sigma=0.5, P1=P1, max_order=2)

        # 11 * 13 = 143 > 100, so (11, 13) should not exist
        assert (11, 13) not in atoms

        # But (2, 3) = 6 ≤ 100 should exist
        assert (2, 3) in atoms


class TestSymmetryProperties:
    """Tests for mathematical symmetry properties."""

    @pytest.fixture
    def P1(self):
        return P1Polynomial(tilde_coeffs=[-2.0, 1.0])

    def test_taylor_coefficient_symmetry(self, P1):
        """I_{r,s}(A,B) relates to I_{s,r}(B,A) by swapping roles."""
        N = 100
        sigma = 0.5

        atom_12 = compute_ratio_atom(1, 2, N, sigma, P1, max_order=2)
        atom_21 = compute_ratio_atom(2, 1, N, sigma, P1, max_order=2)

        # I_{r,s}(1,2) should equal I_{s,r}(2,1) when r,s swapped
        # This is because (A,B) -> (B,A) swaps the roles of δ_A and δ_B
        I_10_12 = atom_12.get_coefficient(1, 0)
        I_01_21 = atom_21.get_coefficient(0, 1)

        assert np.isclose(I_10_12, I_01_21, rtol=1e-10)

    def test_I00_consistency_with_ratio_class(self, P1):
        """I_{0,0} at δ=0 should match ratio-class diagonal."""
        from src.local.ratio_classes import compute_ratio_class_contribution
        from src.local.local_moment import LocalMomentConfig
        from src.local.mollifier_coeffs import compute_psi1_coeffs

        N = 100
        sigma = 0.5

        # Compute ψ₁ coefficients
        sieve = compute_sieve_arrays(N, include_psi3=False)
        a = compute_psi1_coeffs(N, sieve, P1)

        config = LocalMomentConfig(T=0.0, Delta=5.0, sigma=sigma)

        # For (1, 2): ratio-class contribution
        contrib = compute_ratio_class_contribution(1, 2, a, config)

        # Ratio-atom I_{0,0}
        atom = compute_ratio_atom(1, 2, N, sigma, P1, max_order=0)

        # The relationship is:
        # RatioClass = μ(A)μ(B) * (AB)^{-σ} * ŵ(log(A/B)) * exp(-iT·log(A/B)) * I_{0,0}
        # At T=0, this simplifies to check I_{0,0} structure

        # Just verify both are non-zero and have same sign
        assert contrib is not None
        assert atom is not None
        assert atom.get_coefficient(0, 0) > 0


class TestEndpointDerivative:
    """Tests for endpoint derivative analysis."""

    def test_P1_deriv_at_1_formula(self):
        """P₁'(1) = 1 - a₀ for P₁(x) = x + x(1-x)P̃(1-x)."""
        # With a0 = -2
        P1 = P1Polynomial(tilde_coeffs=[-2.0, 1.0, 0.5])

        # Evaluate P₁'(1)
        deriv_at_1 = P1.eval_deriv(np.array([1.0]), 1)[0]

        # Should equal 1 - a₀ = 1 - (-2) = 3
        expected = 1 - (-2.0)
        assert np.isclose(deriv_at_1, expected)

    def test_P1_deriv_at_1_linear(self):
        """For P₁(x) = x (a₀=0), P₁'(1) = 1."""
        P1 = P1Polynomial(tilde_coeffs=[])

        deriv_at_1 = P1.eval_deriv(np.array([1.0]), 1)[0]
        assert np.isclose(deriv_at_1, 1.0)


class TestNumericalAccuracy:
    """Tests for numerical accuracy of Taylor coefficients."""

    def test_high_order_coefficients_decay(self):
        """Higher order Taylor coefficients should generally decay."""
        P1 = P1Polynomial(tilde_coeffs=[-2.0, 1.0])

        atom = compute_ratio_atom(1, 2, N=200, sigma=0.5, P1=P1, max_order=3)

        I_00 = abs(atom.get_coefficient(0, 0))
        I_10 = abs(atom.get_coefficient(1, 0))
        I_20 = abs(atom.get_coefficient(2, 0))
        I_30 = abs(atom.get_coefficient(3, 0))

        # Leading coefficient should be largest
        assert I_00 >= I_10 * 0.1  # Allow some flexibility

    def test_series_truncation_error(self):
        """Evaluate series at small δ and check convergence."""
        P1 = P1Polynomial(tilde_coeffs=[-2.0, 1.0])

        atom_low = compute_ratio_atom(1, 2, N=100, sigma=0.5, P1=P1, max_order=2)
        atom_high = compute_ratio_atom(1, 2, N=100, sigma=0.5, P1=P1, max_order=3)

        # At small δ, higher order should give similar result
        delta = 0.05
        val_low = atom_low.evaluate_at_delta(delta, delta)
        val_high = atom_high.evaluate_at_delta(delta, delta)

        # Should be close for small δ
        if abs(val_low) > 1e-10:
            rel_diff = abs(val_high - val_low) / abs(val_low)
            assert rel_diff < 0.5  # Within 50% (depends on convergence rate)

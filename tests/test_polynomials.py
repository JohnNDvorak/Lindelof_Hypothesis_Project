"""
Unit tests for polynomials module.

Tests cover:
- Constraint enforcement (automatic by parameterization)
- PRZZ polynomial reproduction
- Derivative computation via monomial conversion
- JSON self-consistency
- Vectorization
"""

import json
import numpy as np
import pytest
from pathlib import Path

from src.polynomials import (
    falling_factorial,
    Polynomial,
    P1Polynomial,
    PellPolynomial,
    QPolynomial,
    make_P1_from_tilde,
    make_Pell_from_tilde,
    make_Q_from_basis,
    load_przz_polynomials,
)


# =============================================================================
# JSON Self-Consistency Test
# =============================================================================

class TestJSONSelfConsistency:
    """Verify przz_parameters.json internal consistency."""

    def test_kappa_c_R_relationship(self):
        """Verify kappa = 1 - log(c)/R from JSON values."""
        json_path = Path(__file__).parent.parent / "data" / "przz_parameters.json"
        with open(json_path) as f:
            data = json.load(f)

        R = data["configuration"]["R"]
        targets = data["targets"]

        # targets["c"] is a display-rounded value; prefer c_precise for exact identities
        c_display = targets["c"]
        c = targets.get("c_precise", c_display)
        kappa = targets["kappa"]

        # Check kappa = 1 - log(c)/R
        computed_kappa = 1 - np.log(c) / R
        assert abs(computed_kappa - kappa) < 1e-8, (
            f"kappa mismatch: computed {computed_kappa} vs stored {kappa}"
        )

        # Check c = exp(R*(1-kappa))
        computed_c = np.exp(R * (1 - kappa))
        assert abs(computed_c - c) < 1e-10, (
            f"c mismatch: computed {computed_c} vs stored {c}"
        )


# =============================================================================
# Falling Factorial Tests
# =============================================================================

class TestFallingFactorial:
    """Tests for falling_factorial helper function."""

    def test_basic_values(self):
        """Test standard falling factorial values."""
        assert falling_factorial(5, 0) == 1
        assert falling_factorial(5, 1) == 5
        assert falling_factorial(5, 2) == 20  # 5*4
        assert falling_factorial(5, 3) == 60  # 5*4*3
        assert falling_factorial(5, 4) == 120  # 5*4*3*2
        assert falling_factorial(5, 5) == 120  # 5!

    def test_k_greater_than_n(self):
        """falling_factorial returns 0 when k > n."""
        assert falling_factorial(3, 4) == 0
        assert falling_factorial(0, 1) == 0
        assert falling_factorial(5, 10) == 0

    def test_k_equals_zero(self):
        """falling_factorial(n, 0) = 1 for any n."""
        for n in range(10):
            assert falling_factorial(n, 0) == 1

    def test_negative_k_raises(self):
        """Negative k should raise ValueError."""
        with pytest.raises(ValueError):
            falling_factorial(5, -1)


# =============================================================================
# Base Polynomial Tests
# =============================================================================

class TestPolynomial:
    """Tests for base Polynomial class."""

    def test_degree(self):
        """Test polynomial degree computation."""
        assert Polynomial(np.array([1.0, 2.0, 3.0])).degree == 2
        assert Polynomial(np.array([1.0])).degree == 0
        assert Polynomial(np.array([0.0])).degree == 0
        assert Polynomial(np.array([])).degree == -1
        # Test with trailing zeros
        assert Polynomial(np.array([1.0, 2.0, 0.0])).degree == 1

    def test_eval_constant(self):
        """Test evaluation of constant polynomial."""
        p = Polynomial(np.array([5.0]))
        x = np.linspace(0, 1, 10)
        assert np.allclose(p.eval(x), 5.0)

    def test_eval_linear(self):
        """Test evaluation of linear polynomial."""
        p = Polynomial(np.array([2.0, 3.0]))  # 2 + 3x
        x = np.array([0.0, 1.0, 2.0])
        expected = np.array([2.0, 5.0, 8.0])
        assert np.allclose(p.eval(x), expected)

    def test_eval_quadratic(self):
        """Test evaluation of quadratic polynomial."""
        p = Polynomial(np.array([1.0, 2.0, 3.0]))  # 1 + 2x + 3x^2
        x = np.array([0.0, 1.0, 2.0])
        expected = np.array([1.0, 6.0, 17.0])
        assert np.allclose(p.eval(x), expected)

    def test_derivative_analytic(self):
        """Test derivative matches analytic formula."""
        # P(x) = 1 + 2x + 3x^2 + 4x^3
        p = Polynomial(np.array([1.0, 2.0, 3.0, 4.0]))
        x = np.linspace(0.1, 0.9, 20)

        # P'(x) = 2 + 6x + 12x^2
        assert np.allclose(p.eval_deriv(x, 1), 2 + 6*x + 12*x**2, rtol=1e-12)

        # P''(x) = 6 + 24x
        assert np.allclose(p.eval_deriv(x, 2), 6 + 24*x, rtol=1e-12)

        # P'''(x) = 24
        assert np.allclose(p.eval_deriv(x, 3), np.full_like(x, 24.0), rtol=1e-12)

        # P''''(x) = 0
        assert np.allclose(p.eval_deriv(x, 4), np.zeros_like(x), rtol=1e-12)

    def test_derivative_higher_than_degree(self):
        """Derivative order > degree returns zero."""
        p = Polynomial(np.array([1.0, 2.0, 3.0]))  # degree 2
        x = np.linspace(0, 1, 10)
        assert np.allclose(p.eval_deriv(x, 3), 0.0)
        assert np.allclose(p.eval_deriv(x, 10), 0.0)


# =============================================================================
# Constraint Enforcement Tests
# =============================================================================

class TestConstraintEnforcement:
    """Tests that constraints are automatically enforced by parameterization."""

    def test_P1_constraints_automatic(self):
        """P1(0)=0 and P1(1)=1 for ANY P_tilde coefficients."""
        rng = np.random.default_rng(42)
        x = np.array([0.0, 1.0])

        for _ in range(20):
            random_tilde = rng.standard_normal(5).tolist()
            p1 = make_P1_from_tilde(random_tilde)
            vals = p1.eval(x)
            assert abs(vals[0]) < 1e-14, "P1(0) should be 0"
            assert abs(vals[1] - 1.0) < 1e-14, "P1(1) should be 1"

    def test_Pell_zero_at_zero_automatic(self):
        """P_ell(0)=0 for ANY P_tilde coefficients."""
        rng = np.random.default_rng(123)

        for _ in range(20):
            random_tilde = rng.standard_normal(4).tolist()
            p = make_Pell_from_tilde(random_tilde)
            assert abs(p.eval(np.array([0.0]))[0]) < 1e-14

    def test_Q_one_at_zero_enforced(self):
        """Q(0)=1 exactly when enforce_Q0=True."""
        # Use coefficients that don't naturally sum to 1
        q = make_Q_from_basis({1: 0.5, 3: 0.3, 5: -0.1}, enforce_Q0=True)
        assert abs(q.eval(np.array([0.0]))[0] - 1.0) < 1e-14

    def test_Q_paper_literal_mode(self):
        """Q(0) uses stored c0 when enforce_Q0=False."""
        coeffs = {0: 0.490464, 1: 0.636851, 3: -0.159327, 5: 0.032011}
        q = make_Q_from_basis(coeffs, enforce_Q0=False)
        expected_Q0 = sum(coeffs.values())  # approx 0.999999
        assert abs(q.eval(np.array([0.0]))[0] - expected_Q0) < 1e-10


# =============================================================================
# PRZZ Polynomial Reproduction Tests
# =============================================================================

class TestPRZZReproduction:
    """Tests that PRZZ polynomials are correctly loaded and evaluated."""

    def test_przz_P1_grid(self):
        """P1 matches expected values across [0,1]."""
        P1, _, _, _ = load_przz_polynomials()
        x = np.linspace(0, 1, 21)

        # Compute expected from explicit formula
        tilde_coeffs = [0.261076, -1.071007, -0.236840, 0.260233]

        def p_tilde(x_val):
            y = 1 - x_val
            return sum(c * y**i for i, c in enumerate(tilde_coeffs))

        expected = x + x * (1 - x) * np.array([p_tilde(xi) for xi in x])
        assert np.allclose(P1.eval(x), expected, rtol=1e-12)

    def test_przz_P2_grid(self):
        """P2 matches expected values across [0,1]."""
        _, P2, _, _ = load_przz_polynomials()
        x = np.linspace(0, 1, 21)
        expected = 1.048274*x + 1.319912*x**2 - 0.940058*x**3
        assert np.allclose(P2.eval(x), expected, rtol=1e-12)

    def test_przz_P3_grid(self):
        """P3 matches expected values across [0,1]."""
        _, _, P3, _ = load_przz_polynomials()
        x = np.linspace(0, 1, 21)
        expected = 0.522811*x - 0.686510*x**2 - 0.049923*x**3
        assert np.allclose(P3.eval(x), expected, rtol=1e-12)

    def test_przz_Q_grid(self):
        """Q matches expected values across [0,1]."""
        _, _, _, Q = load_przz_polynomials(enforce_Q0=False)
        x = np.linspace(0, 1, 21)

        u = 1 - 2*x  # basis variable
        expected = 0.490464 + 0.636851*u - 0.159327*u**3 + 0.032011*u**5
        assert np.allclose(Q.eval(x), expected, rtol=1e-10)


# =============================================================================
# Endpoint Behavior Tests
# =============================================================================

class TestEndpointBehavior:
    """Test key endpoint values for all polynomials."""

    def test_P1_endpoints(self):
        """P1(0)=0, P1(1)=1."""
        P1, _, _, _ = load_przz_polynomials()
        assert abs(P1.eval(np.array([0.0]))[0]) < 1e-14
        assert abs(P1.eval(np.array([1.0]))[0] - 1.0) < 1e-14

    def test_P2_P3_zero_at_zero(self):
        """P2(0)=0, P3(0)=0."""
        _, P2, P3, _ = load_przz_polynomials()
        assert abs(P2.eval(np.array([0.0]))[0]) < 1e-14
        assert abs(P3.eval(np.array([0.0]))[0]) < 1e-14

    def test_Q_at_half(self):
        """Q(1/2): since (1-2x)=0 at x=1/2, Q(1/2) = c0."""
        _, _, _, Q = load_przz_polynomials(enforce_Q0=False)
        # c0 = 0.490464
        assert abs(Q.eval(np.array([0.5]))[0] - 0.490464) < 1e-10


# =============================================================================
# Derivative Tests (via Monomial Conversion)
# =============================================================================

class TestDerivativesViaMonomial:
    """Test that derivatives are correct via monomial conversion."""

    def test_P1_derivative_via_monomial(self):
        """P1 derivative matches monomial conversion."""
        P1, _, _, _ = load_przz_polynomials()
        x = np.linspace(0.05, 0.95, 20)

        p1_mono = P1.to_monomial()
        for k in range(1, 5):
            assert np.allclose(
                P1.eval_deriv(x, k),
                p1_mono.eval_deriv(x, k),
                rtol=1e-12
            )

    def test_P2_derivative_via_monomial(self):
        """P2 derivative matches monomial conversion."""
        _, P2, _, _ = load_przz_polynomials()
        x = np.linspace(0.05, 0.95, 20)

        p2_mono = P2.to_monomial()
        for k in range(1, 4):
            assert np.allclose(
                P2.eval_deriv(x, k),
                p2_mono.eval_deriv(x, k),
                rtol=1e-12
            )

    def test_P3_derivative_via_monomial(self):
        """P3 derivative matches monomial conversion."""
        _, _, P3, _ = load_przz_polynomials()
        x = np.linspace(0.05, 0.95, 20)

        p3_mono = P3.to_monomial()
        for k in range(1, 4):
            assert np.allclose(
                P3.eval_deriv(x, k),
                p3_mono.eval_deriv(x, k),
                rtol=1e-12
            )

    def test_Q_derivative_via_monomial(self):
        """Q derivative matches monomial conversion."""
        _, _, _, Q = load_przz_polynomials()
        x = np.linspace(0.05, 0.95, 20)

        q_mono = Q.to_monomial()
        for k in range(1, 4):
            assert np.allclose(
                Q.eval_deriv(x, k),
                q_mono.eval_deriv(x, k),
                rtol=1e-10
            )


# =============================================================================
# Vectorization Tests
# =============================================================================

class TestVectorization:
    """Test that evaluation works on arrays of various shapes."""

    def test_vectorized_1d(self):
        """Evaluation works on 1D arrays."""
        p = Polynomial(np.array([1.0, 2.0, 3.0]))
        x = np.linspace(0, 1, 100)
        assert p.eval(x).shape == (100,)
        assert p.eval_deriv(x, 1).shape == (100,)

    def test_vectorized_2d(self):
        """Evaluation works on 2D arrays (quadrature grids)."""
        p = Polynomial(np.array([1.0, 2.0, 3.0]))
        rng = np.random.default_rng(999)
        x = rng.random((50, 50))
        assert p.eval(x).shape == (50, 50)
        assert p.eval_deriv(x, 1).shape == (50, 50)

    def test_constrained_poly_vectorized(self):
        """Constrained polynomials also vectorize correctly."""
        P1, P2, _, Q = load_przz_polynomials()
        rng = np.random.default_rng(888)
        x = rng.random((30, 30))

        assert P1.eval(x).shape == (30, 30)
        assert P2.eval(x).shape == (30, 30)
        assert Q.eval(x).shape == (30, 30)


# =============================================================================
# Monomial Conversion Roundtrip Tests
# =============================================================================

class TestMonomialRoundtrip:
    """Test that monomial conversion and evaluation are consistent."""

    def test_P1_to_monomial_roundtrip(self):
        """P1 -> monomial -> evaluate gives same result."""
        P1, _, _, _ = load_przz_polynomials()
        x = np.linspace(0, 1, 50)

        direct = P1.eval(x)
        via_monomial = P1.to_monomial().eval(x)
        assert np.allclose(direct, via_monomial, rtol=1e-12)

    def test_Pell_to_monomial_roundtrip(self):
        """P_ell -> monomial -> evaluate gives same result."""
        _, P2, P3, _ = load_przz_polynomials()
        x = np.linspace(0, 1, 50)

        for P in [P2, P3]:
            direct = P.eval(x)
            via_monomial = P.to_monomial().eval(x)
            assert np.allclose(direct, via_monomial, rtol=1e-12)

    def test_Q_to_monomial_roundtrip(self):
        """Q -> monomial -> evaluate gives same result."""
        _, _, _, Q = load_przz_polynomials()
        x = np.linspace(0, 1, 50)

        direct = Q.eval(x)
        via_monomial = Q.to_monomial().eval(x)
        assert np.allclose(direct, via_monomial, rtol=1e-10)

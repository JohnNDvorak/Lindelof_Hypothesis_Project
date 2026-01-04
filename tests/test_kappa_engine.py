"""
tests/test_kappa_engine.py
Tests for the production KappaEngine.

Validates that:
1. Engine loads PRZZ polynomials correctly
2. First-principles formulas are applied correctly
3. Both benchmarks pass within tolerance

Created: 2025-12-27
"""

import pytest
import math

from src.kappa_engine import (
    KappaEngine,
    compute_g_I1,
    compute_g_I2,
    compute_base,
    compute_przz_kappa,
    validate_przz_benchmarks,
)


# Constants
THETA = 4 / 7
K = 3
R_KAPPA = 1.3036
R_KAPPA_STAR = 1.1167


class TestFirstPrinciplesFormulas:
    """Test the first-principles g correction formulas."""

    def test_g_I1_formula(self):
        """
        g_I1 = 1 + θ(1-θ)(2(K-1)+θ) / (8K(2K+1)²)

        For K=3, θ=4/7:
        numerator = (4/7) × (3/7) × (32/7) = 384/343
        denominator = 8 × 3 × 49 = 1176
        ε_I1 = 384/343 / 1176 = 0.0009519843...
        """
        g = compute_g_I1(THETA, K)

        # Direct formula check
        numerator = THETA * (1 - THETA) * (2 * (K - 1) + THETA)
        denominator = 8 * K * (2 * K + 1) ** 2
        expected = 1 + numerator / denominator

        assert g == pytest.approx(expected, rel=1e-12)
        assert g == pytest.approx(1.0009519843, rel=1e-6)

    def test_g_I2_formula(self):
        """
        g_I2 = 1 + θ(2-θ) / (2K(2K+1))

        For K=3, θ=4/7:
        θ(2-θ) = (4/7)(10/7) = 40/49
        2K(2K+1) = 42
        ε_I2 = 40/49 / 42 = 0.0194363...
        """
        g = compute_g_I2(THETA, K)

        # Direct formula check
        expected = 1 + THETA * (2 - THETA) / (2 * K * (2 * K + 1))

        assert g == pytest.approx(expected, rel=1e-12)
        assert g == pytest.approx(1.0194363460, rel=1e-6)

    def test_base_formula(self):
        """
        base = exp(R) + (2K-1)

        For R=1.3036, K=3:
        base = exp(1.3036) + 5 ≈ 8.6825
        """
        base = compute_base(R_KAPPA, K)

        expected = math.exp(R_KAPPA) + (2 * K - 1)
        assert base == pytest.approx(expected, rel=1e-12)
        assert base == pytest.approx(8.6825299412, rel=1e-6)

    def test_3_over_28_derivation(self):
        """
        The (3/28) coefficient in the simplified θ³ form is NOT empirical.

        For K=3, θ=4/7:
        (3/28) = (1-θ)(2(K-1)+θ) / (8(2K+1)θ²)
        """
        # The coefficient
        coeff = (1 - THETA) * (2 * (K - 1) + THETA) / (8 * (2 * K + 1) * THETA**2)
        assert coeff == pytest.approx(3 / 28, rel=1e-10)

        # And the simplified form equals the unified form
        g_I1_unified = compute_g_I1(THETA, K)
        g_I1_simplified = 1 + (3 / 28) * THETA**3 / (K * (2 * K + 1))
        assert g_I1_unified == pytest.approx(g_I1_simplified, rel=1e-12)


class TestKappaEngineBenchmarks:
    """Test the KappaEngine against PRZZ benchmarks."""

    def test_kappa_benchmark(self):
        """κ benchmark should match within 0.001%."""
        engine = KappaEngine.from_przz_kappa(n_quad=60)
        result = engine.compute_kappa()

        kappa_target = 0.417293962
        gap_pct = abs(result.kappa / kappa_target - 1) * 100

        assert gap_pct < 0.001, f"κ gap {gap_pct:.4f}% exceeds 0.001%"

    def test_kappa_star_benchmark(self):
        """κ* benchmark should match within 0.001%."""
        engine = KappaEngine.from_przz_kappa_star(n_quad=60)
        result = engine.compute_kappa()

        kappa_star_target = 0.407511457
        gap_pct = abs(result.kappa / kappa_star_target - 1) * 100

        assert gap_pct < 0.001, f"κ* gap {gap_pct:.4f}% exceeds 0.001%"

    def test_c_target_kappa(self):
        """c value for κ benchmark should match within 0.001%."""
        engine = KappaEngine.from_przz_kappa(n_quad=60)
        result = engine.compute_kappa()

        c_target = 2.13745440613217263636
        gap_pct = abs(result.c / c_target - 1) * 100

        assert gap_pct < 0.001, f"c gap {gap_pct:.4f}% exceeds 0.001%"

    def test_c_target_kappa_star(self):
        """c value for κ* benchmark should match within 0.001%."""
        engine = KappaEngine.from_przz_kappa_star(n_quad=60)
        result = engine.compute_kappa()

        c_target = 1.9379524112
        gap_pct = abs(result.c / c_target - 1) * 100

        assert gap_pct < 0.001, f"c gap {gap_pct:.4f}% exceeds 0.001%"


class TestKappaEngineIntegration:
    """Integration tests for convenience functions."""

    def test_compute_przz_kappa(self):
        """Test the compute_przz_kappa convenience function."""
        result = compute_przz_kappa(n_quad=60)

        assert result.kappa == pytest.approx(0.417294, rel=1e-4)
        assert result.c == pytest.approx(2.1374544, rel=1e-5)

    def test_validate_przz_benchmarks_passes(self):
        """Both benchmarks should pass validation."""
        validation = validate_przz_benchmarks(tolerance_pct=0.01, n_quad=60)

        assert validation["kappa"]["passed"]
        assert validation["kappa_star"]["passed"]
        assert abs(validation["kappa"]["gap_pct"]) < 0.01
        assert abs(validation["kappa_star"]["gap_pct"]) < 0.01


class TestKappaResult:
    """Test the KappaResult dataclass."""

    def test_result_has_all_fields(self):
        """Result should contain all expected fields."""
        engine = KappaEngine.from_przz_kappa(n_quad=60)
        result = engine.compute_kappa()

        # Main values
        assert hasattr(result, 'kappa')
        assert hasattr(result, 'c')

        # Integrals
        assert hasattr(result, 'integrals')
        assert result.integrals.S12_plus > 0
        assert result.integrals.S12_minus > 0

        # Corrections
        assert hasattr(result, 'corrections')
        assert result.corrections.g_I1 > 1
        assert result.corrections.g_I2 > 1
        assert result.corrections.m > 0

    def test_result_str_formatting(self):
        """Result should have nice string representation."""
        engine = KappaEngine.from_przz_kappa(n_quad=60)
        result = engine.compute_kappa()

        s = str(result)
        assert 'κ =' in s
        assert 'c =' in s
        assert 'g_I1' in s
        assert 'S12' in s


class TestNoCalibration:
    """Verify no calibrated parameters are used."""

    def test_g_I1_differs_from_calibrated(self):
        """g_I1 from formula should differ slightly from calibrated value."""
        g_I1_formula = compute_g_I1(THETA, K)
        g_I1_calibrated = 1.00091428  # From Phase 45

        # They should be close but not identical
        assert g_I1_formula != pytest.approx(g_I1_calibrated, rel=1e-10)
        # But within 0.01% of each other
        gap = abs(g_I1_formula / g_I1_calibrated - 1) * 100
        assert gap < 0.01

    def test_g_I2_differs_from_calibrated(self):
        """g_I2 from formula should differ slightly from calibrated value."""
        g_I2_formula = compute_g_I2(THETA, K)
        g_I2_calibrated = 1.01945154  # From Phase 45

        # They should be close but not identical
        assert g_I2_formula != pytest.approx(g_I2_calibrated, rel=1e-10)
        # But within 0.01% of each other
        gap = abs(g_I2_formula / g_I2_calibrated - 1) * 100
        assert gap < 0.01

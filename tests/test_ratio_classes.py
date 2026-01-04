"""
Tests for ratio-class decomposition.
"""

import numpy as np
import pytest
from math import gcd
from src.local.ratio_classes import (
    RatioClassContribution,
    RatioClassDecomposition,
    compute_ratio_class_contribution,
    compute_ratio_classes,
)
from src.local.local_moment import LocalMomentConfig


class TestRatioClassContribution:
    """Tests for individual ratio-class computation."""

    def test_single_pair_contribution(self):
        """Test computation for a simple pair (1, 2)."""
        # D(s) = 1 + 2^{-s}: coeffs[1] = 1, coeffs[2] = 1
        coeffs = np.array([0.0, 1.0, 1.0])
        config = LocalMomentConfig(T=0.0, Delta=2.0)

        contrib = compute_ratio_class_contribution(1, 2, coeffs, config)

        assert contrib is not None
        assert contrib.A == 1
        assert contrib.B == 2
        assert np.isclose(contrib.log_ratio, np.log(0.5))
        assert contrib.window_weight > 0
        assert contrib.n_terms == 1  # Only g=1 contributes

    def test_coprime_required(self):
        """Non-coprime pairs should return None."""
        coeffs = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        config = LocalMomentConfig(T=0.0, Delta=2.0)

        # (2, 4) are not coprime
        contrib = compute_ratio_class_contribution(2, 4, coeffs, config)
        assert contrib is None

        # (3, 6) are not coprime
        contrib = compute_ratio_class_contribution(3, 6, coeffs, config)
        assert contrib is None

    def test_outside_window(self):
        """Pairs outside Fejér window should return None."""
        coeffs = np.array([0.0, 1.0, 1.0, 1.0])
        config = LocalMomentConfig(T=0.0, Delta=0.5)  # Small bandwidth

        # log(3/1) ≈ 1.1 > 0.5 = Delta
        contrib = compute_ratio_class_contribution(3, 1, coeffs, config)
        assert contrib is None

    def test_symmetric_pair(self):
        """(A, B) and (B, A) should give conjugate contributions at T=0."""
        coeffs = np.zeros(11)
        coeffs[1:11] = 1.0  # Uniform coefficients

        config = LocalMomentConfig(T=0.0, Delta=2.0, sigma=0.5)

        contrib_12 = compute_ratio_class_contribution(1, 2, coeffs, config)
        contrib_21 = compute_ratio_class_contribution(2, 1, coeffs, config)

        assert contrib_12 is not None
        assert contrib_21 is not None

        # At T=0, (1,2) and (2,1) should be complex conjugates
        # because log(1/2) = -log(2/1) and exp(0) = 1
        assert np.isclose(contrib_12.contribution.real, contrib_21.contribution.real)
        assert np.isclose(contrib_12.contribution.imag, -contrib_21.contribution.imag)


class TestRatioClassDecomposition:
    """Tests for full ratio-class decomposition."""

    def test_single_coefficient_no_classes(self):
        """With D(s) = 1, there are no off-diagonal classes."""
        coeffs = np.array([0.0, 1.0])
        config = LocalMomentConfig(T=0.0, Delta=1.0)

        decomp = compute_ratio_classes(coeffs, config, A_max=10)

        assert decomp.diagonal == 1.0
        assert decomp.total_classes == 0
        assert decomp.off_diagonal_total == 0.0

    def test_two_terms_has_classes(self):
        """D(s) = 1 + 2^{-s} should have ratio classes (1,2) and (2,1)."""
        coeffs = np.array([0.0, 1.0, 1.0])
        config = LocalMomentConfig(T=0.0, Delta=2.0)

        decomp = compute_ratio_classes(coeffs, config, A_max=10)

        assert decomp.total_classes >= 2
        # Should have (1,2) and (2,1)
        pairs = {(c.A, c.B) for c in decomp.classes}
        assert (1, 2) in pairs
        assert (2, 1) in pairs

    def test_diagonal_matches_decomposed(self):
        """Diagonal should match compute_ratio_domain_decomposed."""
        from src.local.local_moment import compute_ratio_domain_decomposed

        np.random.seed(42)
        coeffs = np.zeros(51)
        coeffs[1:51] = np.random.randn(50) * 0.3
        coeffs[1] = 1.0

        config = LocalMomentConfig(T=100.0, Delta=1.0)

        decomp = compute_ratio_classes(coeffs, config, A_max=30)
        decomp2 = compute_ratio_domain_decomposed(coeffs, config)

        assert np.isclose(decomp.diagonal, decomp2.diagonal)

    def test_sorted_by_contribution(self):
        """Classes should be sorted by absolute contribution (descending)."""
        np.random.seed(123)
        coeffs = np.zeros(101)
        coeffs[1:101] = np.random.randn(100) * 0.2
        coeffs[1] = 1.0

        config = LocalMomentConfig(T=50.0, Delta=1.0)

        decomp = compute_ratio_classes(coeffs, config, A_max=20)

        if len(decomp.classes) > 1:
            for i in range(len(decomp.classes) - 1):
                assert decomp.classes[i].abs_contribution >= decomp.classes[i+1].abs_contribution

    def test_top_contributors(self):
        """top_contributors should return requested number."""
        np.random.seed(456)
        coeffs = np.zeros(51)
        coeffs[1:51] = np.random.randn(50) * 0.5

        config = LocalMomentConfig(T=0.0, Delta=1.0)

        decomp = compute_ratio_classes(coeffs, config, A_max=20)

        top5 = decomp.top_contributors(5)
        assert len(top5) <= 5

        # If there are at least 5 classes
        if decomp.total_classes >= 5:
            assert len(top5) == 5


class TestRatioClassProperties:
    """Tests for mathematical properties of ratio-class decomposition."""

    def test_only_coprime_pairs(self):
        """All returned pairs should be coprime."""
        np.random.seed(789)
        coeffs = np.zeros(101)
        coeffs[1:101] = np.random.randn(100)

        config = LocalMomentConfig(T=0.0, Delta=2.0)

        decomp = compute_ratio_classes(coeffs, config, A_max=30)

        for c in decomp.classes:
            assert gcd(c.A, c.B) == 1, f"({c.A}, {c.B}) not coprime"

    def test_window_weight_in_range(self):
        """Window weights should be in [0, 1]."""
        np.random.seed(101)
        coeffs = np.zeros(51)
        coeffs[1:51] = np.random.randn(50)

        config = LocalMomentConfig(T=0.0, Delta=1.0)

        decomp = compute_ratio_classes(coeffs, config, A_max=20)

        for c in decomp.classes:
            assert 0 < c.window_weight <= 1

    def test_no_diagonal_in_classes(self):
        """Classes should not include diagonal (A=B)."""
        np.random.seed(202)
        coeffs = np.zeros(31)
        coeffs[1:31] = np.random.randn(30)

        config = LocalMomentConfig(T=0.0, Delta=1.0)

        decomp = compute_ratio_classes(coeffs, config, A_max=20)

        for c in decomp.classes:
            assert c.A != c.B


class TestPrimeAtoms:
    """Tests specifically for prime ratio classes."""

    def test_prime_pair_1_p(self):
        """Test (1, p) pairs for small primes."""
        # Create coefficients where primes have known values
        coeffs = np.zeros(11)
        coeffs[1] = 1.0
        coeffs[2] = 0.5
        coeffs[3] = 0.3
        coeffs[5] = 0.2
        coeffs[7] = 0.1

        config = LocalMomentConfig(T=0.0, Delta=3.0, sigma=0.5)

        # Test (1, 2)
        contrib = compute_ratio_class_contribution(1, 2, coeffs, config)
        assert contrib is not None
        assert contrib.A == 1
        assert contrib.B == 2

        # (1, 3)
        contrib = compute_ratio_class_contribution(1, 3, coeffs, config)
        assert contrib is not None

        # (1, 5)
        contrib = compute_ratio_class_contribution(1, 5, coeffs, config)
        assert contrib is not None

    def test_two_prime_pair(self):
        """Test (p, q) pairs for distinct primes."""
        coeffs = np.zeros(11)
        coeffs[1:11] = 1.0

        config = LocalMomentConfig(T=0.0, Delta=2.0, sigma=0.5)

        # (2, 3)
        contrib = compute_ratio_class_contribution(2, 3, coeffs, config)
        assert contrib is not None
        assert gcd(2, 3) == 1

        # (3, 5)
        contrib = compute_ratio_class_contribution(3, 5, coeffs, config)
        assert contrib is not None


class TestNumericalAccuracy:
    """Tests for numerical accuracy of decomposition."""

    def test_total_matches_sum(self):
        """Off-diagonal total should match sum of class contributions."""
        np.random.seed(303)
        coeffs = np.zeros(51)
        coeffs[1:51] = np.random.randn(50) * 0.3
        coeffs[1] = 1.0

        config = LocalMomentConfig(T=0.0, Delta=1.0)

        decomp = compute_ratio_classes(coeffs, config, A_max=30)

        sum_contributions = sum(c.contribution.real for c in decomp.classes)
        assert np.isclose(decomp.off_diagonal_total, sum_contributions)

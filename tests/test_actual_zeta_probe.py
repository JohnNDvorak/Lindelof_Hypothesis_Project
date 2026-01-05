"""
Tests for Phase 5 actual zeta sanity test.

Tests cover:
- mpmath availability checking
- compute_zeta_psi_actual function
- compare_actual_vs_dirichlet function
- ComparisonResult structure
- Result summarization

Note: Some tests are marked as slow and may require mpmath.
"""

import numpy as np
import pytest

# Check mpmath availability
try:
    import mpmath
    MPMATH_AVAILABLE = True
except ImportError:
    MPMATH_AVAILABLE = False


# Only import if mpmath is available (to avoid import errors)
if MPMATH_AVAILABLE:
    from src.local.actual_zeta_probe import (
        ActualZetaResult,
        ComparisonResult,
        compute_zeta_psi_actual,
        compare_actual_vs_dirichlet,
        run_sanity_test_grid,
        summarize_results,
        format_results_table,
        quick_test,
        check_mpmath_available,
        fejer_windowed_moment_actual_zeta,
    )
    from src.local.vzeta_moment import compute_full_mollifier_coeffs


# Skip all tests if mpmath not available
pytestmark = pytest.mark.skipif(
    not MPMATH_AVAILABLE,
    reason="mpmath not installed"
)


class TestMpmathAvailability:
    """Tests for mpmath availability checking."""

    def test_mpmath_available(self):
        """mpmath should be available for these tests."""
        assert MPMATH_AVAILABLE

    def test_check_mpmath_available_passes(self):
        """check_mpmath_available should not raise when mpmath is installed."""
        # Should not raise
        check_mpmath_available()


class TestComputeZetaPsiActual:
    """Tests for compute_zeta_psi_actual function."""

    def test_returns_complex(self):
        """Function returns complex value."""
        psi_coeffs = np.zeros(11, dtype=np.float64)
        psi_coeffs[1] = 1.0  # Simple ψ(s) = 1^{-s} = 1

        result = compute_zeta_psi_actual(t=100.0, sigma=0.5, psi_coeffs=psi_coeffs)

        assert isinstance(result, complex)

    def test_nonzero_result(self):
        """Result should be non-zero for typical inputs."""
        psi_coeffs = np.zeros(11, dtype=np.float64)
        psi_coeffs[1] = 1.0

        result = compute_zeta_psi_actual(t=100.0, sigma=0.5, psi_coeffs=psi_coeffs)

        assert abs(result) > 0

    def test_different_sigma_different_result(self):
        """Different sigma should give different results."""
        psi_coeffs = np.zeros(11, dtype=np.float64)
        psi_coeffs[1] = 1.0

        result1 = compute_zeta_psi_actual(t=100.0, sigma=0.4, psi_coeffs=psi_coeffs)
        result2 = compute_zeta_psi_actual(t=100.0, sigma=0.5, psi_coeffs=psi_coeffs)

        assert abs(result1 - result2) > 0


class TestFejerWindowedMoment:
    """Tests for Fejér windowed moment with actual ζ."""

    def test_returns_tuple(self):
        """Function returns (moment, diagonal) tuple."""
        psi_coeffs = compute_full_mollifier_coeffs(20, use_optimal=True)

        result = fejer_windowed_moment_actual_zeta(
            T=100.0, Delta=1.0, sigma=0.35, psi_coeffs=psi_coeffs, n_points=30
        )

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_moment_is_positive(self):
        """Moment should be positive (it's an L² integral)."""
        psi_coeffs = compute_full_mollifier_coeffs(20, use_optimal=True)

        moment, _ = fejer_windowed_moment_actual_zeta(
            T=100.0, Delta=1.0, sigma=0.35, psi_coeffs=psi_coeffs, n_points=30
        )

        assert moment > 0


class TestCompareActualVsDirichlet:
    """Tests for compare_actual_vs_dirichlet function."""

    def test_returns_comparison_result(self):
        """Function returns ComparisonResult."""
        result = compare_actual_vs_dirichlet(
            T=100.0, Delta=1.0, sigma=0.35, N=20, n_points=30
        )

        assert isinstance(result, ComparisonResult)

    def test_result_has_all_fields(self):
        """ComparisonResult has all expected fields."""
        result = compare_actual_vs_dirichlet(
            T=100.0, Delta=1.0, sigma=0.35, N=20, n_points=30
        )

        assert hasattr(result, 'T')
        assert hasattr(result, 'Delta')
        assert hasattr(result, 'sigma')
        assert hasattr(result, 'actual_opt')
        assert hasattr(result, 'actual_przz')
        assert hasattr(result, 'dirichlet_opt')
        assert hasattr(result, 'dirichlet_przz')
        assert hasattr(result, 'actual_ratio')
        assert hasattr(result, 'dirichlet_ratio')
        assert hasattr(result, 'ratio_difference')

    def test_all_moments_positive(self):
        """All moment values should be positive."""
        result = compare_actual_vs_dirichlet(
            T=100.0, Delta=1.0, sigma=0.35, N=20, n_points=30
        )

        assert result.actual_opt > 0
        assert result.actual_przz > 0
        assert result.dirichlet_opt > 0
        assert result.dirichlet_przz > 0

    def test_ratios_reasonable(self):
        """Ratios should be finite and positive."""
        result = compare_actual_vs_dirichlet(
            T=100.0, Delta=1.0, sigma=0.35, N=20, n_points=30
        )

        assert 0 < result.actual_ratio < float('inf')
        assert 0 < result.dirichlet_ratio < float('inf')

    def test_ratio_difference_computed(self):
        """ratio_difference = actual_ratio - dirichlet_ratio."""
        result = compare_actual_vs_dirichlet(
            T=100.0, Delta=1.0, sigma=0.35, N=20, n_points=30
        )

        expected_diff = result.actual_ratio - result.dirichlet_ratio
        assert abs(result.ratio_difference - expected_diff) < 1e-10


class TestSummarizeResults:
    """Tests for result summarization."""

    def test_summarize_single_result(self):
        """summarize_results handles single result."""
        result = compare_actual_vs_dirichlet(
            T=100.0, Delta=1.0, sigma=0.35, N=20, n_points=30
        )

        summary = summarize_results([result])

        assert 'n_tests' in summary
        assert summary['n_tests'] == 1
        assert 'actual_ratio' in summary
        assert 'dirichlet_ratio' in summary
        assert 'difference' in summary
        assert 'conclusion' in summary

    def test_summarize_empty_list(self):
        """summarize_results handles empty list."""
        summary = summarize_results([])

        assert 'error' in summary


class TestFormatResultsTable:
    """Tests for results table formatting."""

    def test_format_returns_string(self):
        """format_results_table returns a string."""
        result = compare_actual_vs_dirichlet(
            T=100.0, Delta=1.0, sigma=0.35, N=20, n_points=30
        )

        table = format_results_table([result])

        assert isinstance(table, str)
        assert len(table) > 0

    def test_format_includes_header(self):
        """Table includes header."""
        result = compare_actual_vs_dirichlet(
            T=100.0, Delta=1.0, sigma=0.35, N=20, n_points=30
        )

        table = format_results_table([result])

        assert "Phase 5" in table or "Sanity Test" in table

    def test_format_includes_conclusion(self):
        """Table includes conclusion."""
        result = compare_actual_vs_dirichlet(
            T=100.0, Delta=1.0, sigma=0.35, N=20, n_points=30
        )

        table = format_results_table([result])

        assert "CONCLUSION" in table


class TestQuickTest:
    """Tests for quick_test function."""

    def test_quick_test_returns_result(self):
        """quick_test returns ComparisonResult."""
        result = quick_test(T=100.0, sigma=0.35)

        assert isinstance(result, ComparisonResult)


@pytest.mark.slow
class TestSanityTestGrid:
    """Tests for full sanity test grid (slow)."""

    def test_run_minimal_grid(self):
        """Run minimal grid test."""
        results = run_sanity_test_grid(
            T_values=[100],
            Delta_values=[1.0],
            sigma_values=[0.35],
            N=20,
            n_points=30,
            verbose=False,
        )

        assert len(results) == 1
        assert isinstance(results[0], ComparisonResult)


class TestDirichletRatioConsistency:
    """Tests that Dirichlet ratios match Phase 4 expectations."""

    def test_dirichlet_ratio_near_phase4(self):
        """Dirichlet ratio should be near 1.2 as found in Phase 4."""
        result = compare_actual_vs_dirichlet(
            T=500.0, Delta=1.0, sigma=0.35, N=40, n_points=50
        )

        # Phase 4 found Dirichlet ratios around 1.2
        assert 0.9 < result.dirichlet_ratio < 1.5

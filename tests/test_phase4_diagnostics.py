"""
Tests for Phase 4 apples-to-apples diagnostics.

Tests cover:
- sigma_override parameter
- compare_same_sigma function
- validate_global_limit_v2 (diagonal as Δ→0 limit)
- mesoscopic_sweep and adaptive_delta_sweep
- off_diag_comparison_grid
- VZetaMomentResult diagonal/off-diagonal fields
"""

import numpy as np
import pytest

from src.local.vzeta_moment import (
    VZetaMomentConfig,
    VZetaMomentResult,
    compute_vzeta_psi_moment,
    compare_same_sigma,
    validate_global_limit_v2,
    mesoscopic_sweep,
    adaptive_delta_sweep,
    off_diag_comparison_grid,
    MESOSCOPIC_DELTAS,
    STANDARD_DELTAS,
)


class TestSigmaOverride:
    """Tests for sigma_override parameter."""

    def test_sigma_override_bypasses_levinson(self):
        """sigma_override takes precedence over Levinson line calculation."""
        config = VZetaMomentConfig(
            T=1000, Delta=1.0, R=1.14976, sigma_override=0.4
        )
        assert config.sigma == 0.4

    def test_sigma_override_none_uses_levinson(self):
        """Without sigma_override, uses Levinson line."""
        config = VZetaMomentConfig(T=1000, Delta=1.0, R=1.14976)
        expected = 0.5 - 1.14976 / np.log(1000)
        assert abs(config.sigma - expected) < 1e-10

    def test_sigma_override_with_use_levinson_false(self):
        """sigma_override takes precedence even if use_levinson_line=False."""
        config = VZetaMomentConfig(
            T=1000, Delta=1.0, use_levinson_line=False, sigma_override=0.35
        )
        assert config.sigma == 0.35


class TestDiagonalOffDiagonal:
    """Tests for diagonal/off-diagonal fields in VZetaMomentResult."""

    def test_result_has_diagonal_field(self):
        """VZetaMomentResult includes diagonal field."""
        config = VZetaMomentConfig(T=500, Delta=1.0, N=30)
        result = compute_vzeta_psi_moment(config)

        assert hasattr(result, 'diagonal')
        assert result.diagonal > 0

    def test_result_has_off_diagonal_field(self):
        """VZetaMomentResult includes off_diagonal field."""
        config = VZetaMomentConfig(T=500, Delta=1.0, N=30)
        result = compute_vzeta_psi_moment(config)

        assert hasattr(result, 'off_diagonal')

    def test_result_has_off_over_diag_field(self):
        """VZetaMomentResult includes off_over_diag field."""
        config = VZetaMomentConfig(T=500, Delta=1.0, N=30)
        result = compute_vzeta_psi_moment(config)

        assert hasattr(result, 'off_over_diag')

    def test_moment_equals_diag_plus_off(self):
        """moment = diagonal + off_diagonal."""
        config = VZetaMomentConfig(T=500, Delta=1.0, N=30)
        result = compute_vzeta_psi_moment(config)

        assert abs(result.moment - (result.diagonal + result.off_diagonal)) < 1e-10

    def test_off_over_diag_computation(self):
        """off_over_diag = off_diagonal / diagonal."""
        config = VZetaMomentConfig(T=500, Delta=1.0, N=30)
        result = compute_vzeta_psi_moment(config)

        expected = result.off_diagonal / result.diagonal
        assert abs(result.off_over_diag - expected) < 1e-10

    def test_diagnostics_includes_diag_off(self):
        """Diagnostics dict includes diagonal/off-diagonal."""
        config = VZetaMomentConfig(T=500, Delta=1.0, N=30)
        result = compute_vzeta_psi_moment(config)

        assert 'diagonal' in result.diagnostics
        assert 'off_diagonal' in result.diagnostics
        assert 'off_over_diag' in result.diagnostics


class TestCompareSameSigma:
    """Tests for compare_same_sigma function."""

    def test_returns_expected_structure(self):
        """compare_same_sigma returns expected dictionary structure."""
        result = compare_same_sigma(T=500, Delta=1.0, sigma=0.32, N=30)

        assert 'T' in result
        assert 'Delta' in result
        assert 'sigma' in result
        assert 'optimal' in result
        assert 'przz' in result
        assert 'moment_ratio' in result

    def test_both_use_same_sigma(self):
        """Both optimal and PRZZ are evaluated at the same σ."""
        sigma = 0.35
        result = compare_same_sigma(T=500, Delta=1.0, sigma=sigma, N=30)

        # The sigma used should be the one we specified
        assert result['sigma'] == sigma

    def test_optimal_has_diag_off_fields(self):
        """Optimal result includes diagonal and off-diagonal."""
        result = compare_same_sigma(T=500, Delta=1.0, sigma=0.32, N=30)

        assert 'diagonal' in result['optimal']
        assert 'off_diagonal' in result['optimal']
        assert 'off_over_diag' in result['optimal']

    def test_przz_has_diag_off_fields(self):
        """PRZZ result includes diagonal and off-diagonal."""
        result = compare_same_sigma(T=500, Delta=1.0, sigma=0.32, N=30)

        assert 'diagonal' in result['przz']
        assert 'off_diagonal' in result['przz']
        assert 'off_over_diag' in result['przz']


class TestValidateGlobalLimitV2:
    """Tests for validate_global_limit_v2 (uses diagonal)."""

    def test_returns_tuple(self):
        """validate_global_limit_v2 returns (passed, diagonal, error)."""
        passed, diagonal, error = validate_global_limit_v2(T=500, N=30)

        assert passed in (True, False)
        assert diagonal > 0
        assert error >= 0

    def test_diagonal_is_positive(self):
        """Diagonal is always positive."""
        _, diagonal, _ = validate_global_limit_v2(T=500, use_optimal=True, N=30)
        assert diagonal > 0

        _, diagonal, _ = validate_global_limit_v2(T=500, use_optimal=False, N=30)
        assert diagonal > 0

    def test_error_computation(self):
        """Error is |diagonal - target| / target."""
        target = 1.5
        _, diagonal, error = validate_global_limit_v2(
            T=500, use_optimal=True, target_c=target, N=30
        )

        expected_error = abs(diagonal - target) / target
        assert abs(error - expected_error) < 1e-10


class TestMesoscopicSweep:
    """Tests for mesoscopic_sweep function."""

    def test_returns_four_arrays(self):
        """mesoscopic_sweep returns (deltas, moments, diagonals, off_over_diags)."""
        deltas, moments, diagonals, off_over_diags = mesoscopic_sweep(
            T=300, use_optimal=True, include_standard=False, N=30
        )

        assert len(deltas) == len(MESOSCOPIC_DELTAS)
        assert len(moments) == len(MESOSCOPIC_DELTAS)
        assert len(diagonals) == len(MESOSCOPIC_DELTAS)
        assert len(off_over_diags) == len(MESOSCOPIC_DELTAS)

    def test_includes_standard_when_requested(self):
        """include_standard=True adds standard delta values."""
        deltas, _, _, _ = mesoscopic_sweep(
            T=300, use_optimal=True, include_standard=True, N=30
        )

        expected_len = len(MESOSCOPIC_DELTAS) + len(STANDARD_DELTAS)
        assert len(deltas) == expected_len

    def test_deltas_are_sorted(self):
        """Returned deltas are sorted ascending."""
        deltas, _, _, _ = mesoscopic_sweep(
            T=300, use_optimal=True, include_standard=True, N=30
        )

        assert all(deltas[i] <= deltas[i + 1] for i in range(len(deltas) - 1))

    def test_moments_are_positive(self):
        """All moments are positive."""
        _, moments, _, _ = mesoscopic_sweep(
            T=300, use_optimal=True, include_standard=False, N=30
        )

        assert all(m > 0 for m in moments)


class TestAdaptiveDeltaSweep:
    """Tests for adaptive_delta_sweep function."""

    def test_returns_three_arrays(self):
        """adaptive_delta_sweep returns (deltas, moments, off_over_diags)."""
        deltas, moments, off_over_diags = adaptive_delta_sweep(
            T=500, use_optimal=True, N=30
        )

        assert len(deltas) == 4  # Default alphas = [0.2, 0.3, 0.4, 0.5]
        assert len(moments) == 4
        assert len(off_over_diags) == 4

    def test_custom_alphas(self):
        """Custom alphas are used correctly."""
        alphas = [0.1, 0.2, 0.3]
        deltas, _, _ = adaptive_delta_sweep(
            T=500, alphas=alphas, use_optimal=True, N=30
        )

        assert len(deltas) == 3
        for i, alpha in enumerate(alphas):
            expected_delta = 500 ** (-alpha)
            assert abs(deltas[i] - expected_delta) < 1e-10

    def test_deltas_scale_with_T(self):
        """Δ = T^{-α} scales correctly with T."""
        T = 1000
        alphas = [0.3]
        deltas, _, _ = adaptive_delta_sweep(T=T, alphas=alphas, use_optimal=True, N=30)

        expected = T ** (-0.3)
        assert abs(deltas[0] - expected) < 1e-10


class TestOffDiagComparisonGrid:
    """Tests for off_diag_comparison_grid function."""

    def test_returns_expected_structure(self):
        """off_diag_comparison_grid returns expected dictionary structure."""
        T_values = [300, 500]
        Delta_values = [1.0, 2.0]

        result = off_diag_comparison_grid(T_values, Delta_values, sigma=0.32)

        assert 'T_values' in result
        assert 'Delta_values' in result
        assert 'sigma' in result
        assert 'grid' in result

    def test_grid_has_correct_size(self):
        """Grid has T × Δ entries."""
        T_values = [300, 500]
        Delta_values = [1.0, 2.0]

        result = off_diag_comparison_grid(T_values, Delta_values, sigma=0.32)

        assert len(result['grid']) == 4  # 2 × 2

    def test_grid_entries_have_off_over_diag(self):
        """Each grid entry has off_over_diag for both polynomial sets."""
        T_values = [300]
        Delta_values = [1.0]

        result = off_diag_comparison_grid(T_values, Delta_values, sigma=0.32)

        entry = result['grid'][0]
        assert 'opt_off_over_diag' in entry
        assert 'przz_off_over_diag' in entry


class TestDeltaConstants:
    """Tests for delta grid constants."""

    def test_mesoscopic_deltas_are_small(self):
        """MESOSCOPIC_DELTAS are all < 0.5."""
        assert all(d < 0.5 for d in MESOSCOPIC_DELTAS)

    def test_standard_deltas_are_larger(self):
        """STANDARD_DELTAS are all >= 0.5."""
        assert all(d >= 0.5 for d in STANDARD_DELTAS)

    def test_deltas_are_sorted(self):
        """Delta arrays are sorted."""
        assert all(MESOSCOPIC_DELTAS[i] <= MESOSCOPIC_DELTAS[i + 1]
                   for i in range(len(MESOSCOPIC_DELTAS) - 1))
        assert all(STANDARD_DELTAS[i] <= STANDARD_DELTAS[i + 1]
                   for i in range(len(STANDARD_DELTAS) - 1))

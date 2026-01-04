"""
Ratio-class decomposition for localized moments.

Decomposes the ratio-domain moment by reduced ratio (A, B) with gcd(A,B) = 1.
This identifies which specific ratio classes drive the off-diagonal contribution.

Mathematical framework:
For each coprime pair (A, B), the contribution is:
    C_{A,B} = ŵ(log(A/B)) · exp(-iT·log(A/B)) · Σ_g a[Ag]·ā[Bg]·(ABg²)^{-σ}

where the sum is over g such that:
- Ag ≤ N and Bg ≤ N
- (g, AB) = 1 (coprimality with A and B)

The Fejér window cutoff means only |log(A/B)| ≤ Δ contributes.
"""

from dataclasses import dataclass
from math import gcd
from typing import List, Optional
import numpy as np

from src.local.fejer import FejerKernel
from src.local.local_moment import LocalMomentConfig


@dataclass
class RatioClassContribution:
    """Contribution from a single ratio class (A, B).

    Attributes:
        A: Numerator (coprime with B)
        B: Denominator (coprime with A)
        log_ratio: log(A/B)
        window_weight: ŵ(log(A/B))
        contribution: Complex contribution to moment
        abs_contribution: |contribution|
        n_terms: Number of g values contributing
    """
    A: int
    B: int
    log_ratio: float
    window_weight: float
    contribution: complex
    abs_contribution: float
    n_terms: int


@dataclass
class RatioClassDecomposition:
    """Full decomposition of moment by ratio classes.

    Attributes:
        config: The LocalMomentConfig used
        diagonal: Diagonal contribution (A=B=1 with g varying)
        off_diagonal_total: Sum of all off-diagonal contributions
        classes: List of RatioClassContribution, sorted by |contribution|
        total_classes: Number of non-zero ratio classes
    """
    config: LocalMomentConfig
    diagonal: float
    off_diagonal_total: float
    classes: List[RatioClassContribution]
    total_classes: int

    def top_contributors(self, n: int = 20) -> List[RatioClassContribution]:
        """Return top n contributors by absolute value."""
        return self.classes[:min(n, len(self.classes))]

    def summary_table(self, n: int = 20) -> str:
        """Return formatted summary table of top contributors."""
        lines = []
        lines.append(f"Ratio-class decomposition: T={self.config.T}, Δ={self.config.Delta}")
        lines.append(f"Diagonal: {self.diagonal:.6f}")
        lines.append(f"Off-diagonal total: {self.off_diagonal_total:.6f}")
        lines.append(f"Total non-zero classes: {self.total_classes}")
        lines.append("")
        lines.append(f"{'Rank':>4} | {'(A,B)':>10} | {'|C|':>12} | {'Re(C)':>12} | {'n_terms':>7}")
        lines.append("-" * 60)

        for i, c in enumerate(self.top_contributors(n)):
            lines.append(
                f"{i+1:>4} | ({c.A},{c.B}):>10 | {c.abs_contribution:>12.6f} | "
                f"{c.contribution.real:>12.6f} | {c.n_terms:>7}"
            )

        return "\n".join(lines)


def _coprime_with(g: int, AB: int) -> bool:
    """Check if g is coprime with AB."""
    return gcd(g, AB) == 1


def compute_ratio_class_contribution(
    A: int,
    B: int,
    coeffs: np.ndarray,
    config: LocalMomentConfig,
) -> Optional[RatioClassContribution]:
    """Compute contribution from ratio class (A, B).

    Args:
        A: Numerator (must be coprime with B)
        B: Denominator (must be coprime with A)
        coeffs: Dirichlet polynomial coefficients
        config: Localization configuration

    Returns:
        RatioClassContribution or None if contribution is zero
    """
    if gcd(A, B) != 1:
        return None

    N = len(coeffs) - 1
    kernel = FejerKernel(config.Delta)

    log_ratio = np.log(A / B)
    w_hat = kernel.w_freq(log_ratio)

    if w_hat == 0:
        return None  # Outside Fejér window

    sigma = config.sigma
    T = config.T
    AB = A * B

    # Phase factor from T
    phase = np.exp(-1j * T * log_ratio)

    # Sum over g with (g, AB) = 1 and Ag ≤ N and Bg ≤ N
    g_max = min(N // A, N // B)

    contribution = 0.0 + 0.0j
    n_terms = 0

    for g in range(1, g_max + 1):
        if not _coprime_with(g, AB):
            continue

        idx_Ag = A * g
        idx_Bg = B * g

        if idx_Ag > N or idx_Bg > N:
            continue

        a_Ag = coeffs[idx_Ag]
        a_Bg = coeffs[idx_Bg]

        if a_Ag == 0 or a_Bg == 0:
            continue

        # (ABg²)^{-σ}
        factor = (AB * g * g) ** (-sigma)

        contribution += a_Ag * np.conj(a_Bg) * factor
        n_terms += 1

    if n_terms == 0:
        return None

    # Multiply by window weight and phase
    contribution *= w_hat * phase

    return RatioClassContribution(
        A=A,
        B=B,
        log_ratio=log_ratio,
        window_weight=w_hat,
        contribution=contribution,
        abs_contribution=np.abs(contribution),
        n_terms=n_terms,
    )


def compute_ratio_classes(
    coeffs: np.ndarray,
    config: LocalMomentConfig,
    A_max: int = 50,
) -> RatioClassDecomposition:
    """Decompose moment by ratio classes.

    Iterates over all coprime pairs (A, B) with 1 ≤ A, B ≤ A_max
    and |log(A/B)| ≤ Δ.

    Args:
        coeffs: Dirichlet polynomial coefficients
        config: Localization configuration
        A_max: Maximum value for A and B

    Returns:
        RatioClassDecomposition with sorted list of contributions
    """
    N = len(coeffs) - 1
    sigma = config.sigma

    # Compute diagonal separately (A=B=1, sum over g)
    a = coeffs[1:N+1]
    n = np.arange(1, N + 1, dtype=np.float64)
    n_pow = n ** (-sigma)
    diagonal = np.sum(np.abs(a)**2 * n_pow**2)

    # Collect ratio-class contributions
    classes: List[RatioClassContribution] = []

    for A in range(1, A_max + 1):
        for B in range(1, A_max + 1):
            if A == B:
                continue  # Diagonal is handled separately

            if gcd(A, B) != 1:
                continue

            contrib = compute_ratio_class_contribution(A, B, coeffs, config)
            if contrib is not None:
                classes.append(contrib)

    # Sort by absolute contribution (descending)
    classes.sort(key=lambda c: c.abs_contribution, reverse=True)

    # Compute off-diagonal total
    off_diagonal_total = sum(c.contribution.real for c in classes)

    return RatioClassDecomposition(
        config=config,
        diagonal=diagonal,
        off_diagonal_total=off_diagonal_total,
        classes=classes,
        total_classes=len(classes),
    )


def print_ratio_class_table(
    decomp: RatioClassDecomposition,
    n: int = 20,
) -> None:
    """Print formatted ratio-class table."""
    print(f"\nRatio-class decomposition: T={decomp.config.T}, Δ={decomp.config.Delta}")
    print(f"Diagonal: {decomp.diagonal:.6f}")
    print(f"Off-diagonal total: {decomp.off_diagonal_total:.6f}")
    print(f"Total non-zero classes: {decomp.total_classes}")
    print()
    print(f"{'Rank':>4} | {'(A,B)':>10} | {'|C|':>12} | {'Re(C)':>12} | {'ŵ':>6} | {'n_g':>5}")
    print("-" * 65)

    for i, c in enumerate(decomp.top_contributors(n)):
        print(
            f"{i+1:>4} | ({c.A:>3},{c.B:>3}) | {c.abs_contribution:>12.6f} | "
            f"{c.contribution.real:>12.6f} | {c.window_weight:>6.3f} | {c.n_terms:>5}"
        )

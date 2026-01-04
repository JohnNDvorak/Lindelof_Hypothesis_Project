"""src/kernel_registry.py

Kernel registry / case-selection contract.

This module centralizes the mapping from PRZZ "piece indices" (ℓ, d, …) to
Case A/B/C selection and ω (omega) as described in the TeX.

Why this exists
---------------
The codebase now has multiple evaluators and oracles. To avoid accidental
"oracle drift" (mixing kernels or case rules), the *case selection* must be
owned by exactly one module.

We also support multiple kernel regimes:
- "raw"  : legacy / diagnostic kernel using raw P(u ± x) (no Case C transform)
- "paper": TeX-driven regime where ω drives Case B/C (for d=1: ω = ℓ - 1)

Nothing in this file evaluates integrals. It only answers:
- what case applies?
- what ω applies?

Consumers (Term DSL, series engines, etc.) can then request Taylor coefficients
for the appropriate profile function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

KernelRegime = Literal["raw", "paper"]
CaseTag = Literal["A", "B", "C"]


@dataclass(frozen=True)
class KernelSpec:
    """Case selection + omega for one polynomial factor."""

    regime: KernelRegime
    case: CaseTag
    omega: int


def omega_d1(ell: int) -> int:
    """PRZZ ω(d,l) specialized to d=1 for piece index ell.

    TeX (for d=1): ω = ℓ - 1.
    """

    if ell < 0:
        raise ValueError(f"ell must be non-negative, got {ell}")
    return ell - 1


def case_from_omega(omega: int) -> CaseTag:
    """Map ω to Case A/B/C per PRZZ."""

    if omega < -1:
        raise ValueError(f"omega must be >= -1 for PRZZ cases, got {omega}")
    if omega == -1:
        return "A"
    if omega == 0:
        return "B"
    return "C"


def kernel_spec_for_piece(
    ell: int,
    *,
    d: int = 1,
    regime: KernelRegime = "raw",
) -> KernelSpec:
    """Return KernelSpec for a piece index.

    Args:
        ell: piece index (1..K in this repo for K=3)
        d: mollifier depth (only d=1 is supported here)
        regime: "raw" (diagnostic legacy kernel) or "paper" (TeX-driven)

    Returns:
        KernelSpec(regime, case, omega)
    """

    if d != 1:
        raise NotImplementedError("kernel_spec_for_piece currently supports only d=1")

    if regime == "raw":
        return KernelSpec(regime=regime, case="B", omega=0)

    if regime != "paper":
        raise ValueError(f"Unknown kernel regime: {regime}")

    omega = omega_d1(ell)
    return KernelSpec(regime=regime, case=case_from_omega(omega), omega=omega)


def omega_for_poly_name(poly_name: str, *, regime: KernelRegime = "raw") -> Optional[int]:
    """Convenience: map poly name (P1/P2/P3/Q) to ω.

    - Pℓ: uses ell index
    - Q: not an omega-controlled profile (returns None)
    """

    name = poly_name.strip()
    if name == "Q":
        return None

    if name.startswith("P") and len(name) == 2 and name[1].isdigit():
        ell = int(name[1])
        return kernel_spec_for_piece(ell, regime=regime).omega

    raise ValueError(f"Unrecognized polynomial name: {poly_name!r}")

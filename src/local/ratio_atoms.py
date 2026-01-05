"""
Ratio-atom Taylor series for Backlund bridge analysis.

For a ratio class (A, B), the contribution to the localized moment expands as:

    S_{A,B} = μ(A)μ(B) (AB)^{-σ} Σ_{r,s≥0} (-δ_A)^r/r! (-δ_B)^s/s! · I_{r,s}

where:
    δ_A = log(A) / log(N)
    δ_B = log(B) / log(N)
    I_{r,s} = Σ_{g: (g,AB)=1} μ(g)² P₁^(r)(u_g) P₁^(s)(u_g) g^{-2σ}

This module computes:
1. The Taylor coefficients I_{r,s} (integrals of derivative products)
2. The full bivariate Taylor series as BivariateSeries objects
3. Comparison with PRZZ I₁...I₄ structure

Key insight: The coefficient of δ_A in the (1,p) atom involves P₁'(u_g),
and endpoint behavior P₁'(1) = 1 - a₀ is where the a₀=-2 geometry lives.
"""

from dataclasses import dataclass
from math import gcd, factorial
from typing import Dict, List, Optional, Tuple
import numpy as np

from src.local.sieve import compute_sieve_arrays
from src.local.mollifier_coeffs import compute_u_array
from src.polynomials import P1Polynomial, PellPolynomial
from src.series_bivariate import BivariateSeries


@dataclass
class TaylorCoefficient:
    """A single Taylor coefficient I_{r,s}.

    I_{r,s} = Σ_{g: (g,AB)=1} μ(g)² P₁^(r)(u_g) P₁^(s)(u_g) g^{-2σ}

    Attributes:
        r: Derivative order for first factor
        s: Derivative order for second factor
        value: The computed coefficient
        n_terms: Number of g values that contributed
    """
    r: int
    s: int
    value: float
    n_terms: int


@dataclass
class RatioAtom:
    """Taylor expansion for a ratio class (A, B).

    The ratio-class contribution expands as:
        S_{A,B} = μ(A)μ(B) (AB)^{-σ} Σ_{r,s} (-δ_A)^r/r! (-δ_B)^s/s! · I_{r,s}

    Stored as a BivariateSeries in (δ_A, δ_B).

    Attributes:
        A: Numerator of ratio (coprime with B)
        B: Denominator of ratio (coprime with A)
        N: Mollifier length used
        sigma: Real part of s
        max_order: Maximum Taylor order computed
        mobius_sign: μ(A)μ(B) = ±1 or 0
        prefactor: (AB)^{-σ}
        taylor_coeffs: Dictionary (r,s) -> I_{r,s}
        series: BivariateSeries representation
    """
    A: int
    B: int
    N: int
    sigma: float
    max_order: int
    mobius_sign: int
    prefactor: float
    taylor_coeffs: Dict[Tuple[int, int], TaylorCoefficient]
    series: BivariateSeries

    def get_coefficient(self, r: int, s: int) -> float:
        """Get Taylor coefficient I_{r,s}."""
        if (r, s) in self.taylor_coeffs:
            return self.taylor_coeffs[(r, s)].value
        return 0.0

    def evaluate_at_delta(self, delta_A: float, delta_B: float) -> float:
        """Evaluate the Taylor series at specific (δ_A, δ_B)."""
        return self.mobius_sign * self.prefactor * self.series.evaluate(delta_A, delta_B)

    def endpoint_derivative_term(self) -> float:
        """Return the coefficient of the linear δ term.

        For (1, p) atom: this is I_{0,1}, the term involving P₁'(u_g).
        For (p, 1) atom: this is I_{1,0}.

        This is where P₁'(1) = 1 - a₀ enters.
        """
        if self.A == 1:
            return self.get_coefficient(0, 1)
        elif self.B == 1:
            return self.get_coefficient(1, 0)
        else:
            # For (p, q), return the sum of both linear terms
            return self.get_coefficient(1, 0) + self.get_coefficient(0, 1)


def _is_coprime_with(g: int, AB: int) -> bool:
    """Check if g is coprime with AB."""
    return gcd(g, AB) == 1


def _mobius_product(A: int, B: int, mobius: np.ndarray) -> int:
    """Compute μ(A)·μ(B).

    Both A and B must be squarefree for this to be non-zero.
    """
    if A > len(mobius) - 1 or B > len(mobius) - 1:
        return 0
    return int(mobius[A]) * int(mobius[B])


def compute_taylor_coefficient(
    r: int,
    s: int,
    A: int,
    B: int,
    N: int,
    sigma: float,
    P1: P1Polynomial,
    u_array: np.ndarray,
    mobius: np.ndarray,
) -> TaylorCoefficient:
    """Compute Taylor coefficient I_{r,s}.

    I_{r,s} = Σ_{g: (g,AB)=1, Ag≤N, Bg≤N} μ(g)² P₁^(r)(u_g) P₁^(s)(u_g) g^{-2σ}

    Args:
        r: Derivative order for first P₁ factor
        s: Derivative order for second P₁ factor
        A, B: Ratio class numerator/denominator
        N: Mollifier length
        sigma: Real part of s
        P1: The P₁ polynomial
        u_array: Precomputed u[g] = log(N/g)/log(N)
        mobius: Precomputed Möbius function array

    Returns:
        TaylorCoefficient with computed value
    """
    AB = A * B
    g_max = min(N // A, N // B)

    total = 0.0
    n_terms = 0

    for g in range(1, g_max + 1):
        # Check coprimality with AB
        if not _is_coprime_with(g, AB):
            continue

        # Check μ(g)² = 1 (g squarefree)
        if mobius[g] == 0:
            continue

        # u_g = log(N/g) / log(N)
        u_g = u_array[g]

        # Evaluate derivatives P₁^(r)(u_g) and P₁^(s)(u_g)
        P1_r = P1.eval_deriv(np.array([u_g]), r)[0]
        P1_s = P1.eval_deriv(np.array([u_g]), s)[0]

        # Factor g^{-2σ}
        g_factor = g ** (-2 * sigma)

        total += P1_r * P1_s * g_factor
        n_terms += 1

    return TaylorCoefficient(r=r, s=s, value=total, n_terms=n_terms)


def compute_ratio_atom(
    A: int,
    B: int,
    N: int,
    sigma: float,
    P1: P1Polynomial,
    max_order: int = 3,
) -> Optional[RatioAtom]:
    """Compute the ratio atom Taylor expansion for class (A, B).

    The expansion is:
        S_{A,B}(δ_A, δ_B) = μ(A)μ(B) (AB)^{-σ} Σ_{r,s} (-δ_A)^r/r! (-δ_B)^s/s! · I_{r,s}

    Stored as a BivariateSeries where:
        series[(r, s)] = I_{r,s} / (r! s!)  (with signs absorbed)

    Args:
        A, B: Ratio class (must be coprime)
        N: Mollifier length
        sigma: Real part of s
        P1: The P₁ polynomial
        max_order: Maximum Taylor order (r + s ≤ max_order)

    Returns:
        RatioAtom or None if μ(A)μ(B) = 0
    """
    if gcd(A, B) != 1:
        return None

    # Compute sieve arrays
    sieve = compute_sieve_arrays(N, include_psi3=False)
    mobius = sieve.mobius

    # Check Möbius sign
    mobius_sign = _mobius_product(A, B, mobius)
    if mobius_sign == 0:
        return None

    # Precompute u array
    u_array = compute_u_array(N)

    # Prefactor (AB)^{-σ}
    prefactor = (A * B) ** (-sigma)

    # Compute Taylor coefficients I_{r,s}
    taylor_coeffs: Dict[Tuple[int, int], TaylorCoefficient] = {}
    series_coeffs: Dict[Tuple[int, int], float] = {}

    for r in range(max_order + 1):
        for s in range(max_order + 1):
            if r + s > max_order:
                continue

            coeff = compute_taylor_coefficient(
                r, s, A, B, N, sigma, P1, u_array, mobius
            )
            taylor_coeffs[(r, s)] = coeff

            # Build series coefficient: (-1)^(r+s) * I_{r,s} / (r! * s!)
            # The (-1)^(r+s) comes from (-δ_A)^r * (-δ_B)^s
            sign = (-1) ** (r + s)
            series_coeff = sign * coeff.value / (factorial(r) * factorial(s))
            if series_coeff != 0:
                series_coeffs[(r, s)] = series_coeff

    # Build BivariateSeries
    series = BivariateSeries(
        max_dx=max_order,
        max_dy=max_order,
        coeffs=series_coeffs
    )

    return RatioAtom(
        A=A,
        B=B,
        N=N,
        sigma=sigma,
        max_order=max_order,
        mobius_sign=mobius_sign,
        prefactor=prefactor,
        taylor_coeffs=taylor_coeffs,
        series=series,
    )


def compute_prime_atoms(
    primes: List[int],
    N: int,
    sigma: float,
    P1: P1Polynomial,
    max_order: int = 3,
) -> Dict[Tuple[int, int], RatioAtom]:
    """Compute ratio atoms for prime pairs.

    Atoms computed:
    - (1, p) for each prime p
    - (p, 1) for each prime p
    - (p, q) for each pair of distinct primes

    Args:
        primes: List of primes to include
        N: Mollifier length
        sigma: Real part of s
        P1: The P₁ polynomial
        max_order: Maximum Taylor order

    Returns:
        Dictionary mapping (A, B) -> RatioAtom
    """
    atoms: Dict[Tuple[int, int], RatioAtom] = {}

    # (1, p) atoms
    for p in primes:
        if p > N:
            continue
        atom = compute_ratio_atom(1, p, N, sigma, P1, max_order)
        if atom is not None:
            atoms[(1, p)] = atom

    # (p, 1) atoms
    for p in primes:
        if p > N:
            continue
        atom = compute_ratio_atom(p, 1, N, sigma, P1, max_order)
        if atom is not None:
            atoms[(p, 1)] = atom

    # (p, q) atoms for distinct primes
    for i, p in enumerate(primes):
        for q in primes[i+1:]:
            if p * q > N:
                continue
            atom_pq = compute_ratio_atom(p, q, N, sigma, P1, max_order)
            if atom_pq is not None:
                atoms[(p, q)] = atom_pq
            atom_qp = compute_ratio_atom(q, p, N, sigma, P1, max_order)
            if atom_qp is not None:
                atoms[(q, p)] = atom_qp

    return atoms


def print_atom_table(atoms: Dict[Tuple[int, int], RatioAtom]) -> None:
    """Print summary table of ratio atoms."""
    print(f"\nRatio Atom Summary (Taylor coefficients I_{{r,s}}):")
    print(f"{'(A,B)':>10} | {'μ':>3} | {'I_00':>12} | {'I_10':>12} | {'I_01':>12} | {'I_11':>12} | {'I_20':>12} | {'I_02':>12}")
    print("-" * 95)

    for (A, B), atom in sorted(atoms.items()):
        I_00 = atom.get_coefficient(0, 0)
        I_10 = atom.get_coefficient(1, 0)
        I_01 = atom.get_coefficient(0, 1)
        I_11 = atom.get_coefficient(1, 1)
        I_20 = atom.get_coefficient(2, 0)
        I_02 = atom.get_coefficient(0, 2)

        print(f"({A:>3},{B:>3}) | {atom.mobius_sign:>3} | {I_00:>12.6f} | {I_10:>12.6f} | "
              f"{I_01:>12.6f} | {I_11:>12.6f} | {I_20:>12.6f} | {I_02:>12.6f}")


def compare_atom_polynomials(
    A: int,
    B: int,
    N: int,
    sigma: float,
    P1_opt: P1Polynomial,
    P1_przz: P1Polynomial,
    max_order: int = 3,
) -> None:
    """Compare ratio atom between optimal and PRZZ polynomials."""
    atom_opt = compute_ratio_atom(A, B, N, sigma, P1_opt, max_order)
    atom_przz = compute_ratio_atom(A, B, N, sigma, P1_przz, max_order)

    if atom_opt is None or atom_przz is None:
        print(f"({A}, {B}): Cannot compute (μ=0)")
        return

    print(f"\nRatio Atom ({A}, {B}) Comparison:")
    print(f"  μ(A)μ(B) = {atom_opt.mobius_sign}, prefactor = {atom_opt.prefactor:.6f}")
    print()
    print(f"{'(r,s)':>8} | {'Optimal':>14} | {'PRZZ':>14} | {'Ratio':>10} | {'Diff %':>10}")
    print("-" * 65)

    for r in range(max_order + 1):
        for s in range(max_order + 1):
            if r + s > max_order:
                continue

            I_opt = atom_opt.get_coefficient(r, s)
            I_przz = atom_przz.get_coefficient(r, s)

            if abs(I_przz) > 1e-12:
                ratio = I_opt / I_przz
                diff_pct = (I_opt - I_przz) / abs(I_przz) * 100
            else:
                ratio = float('nan')
                diff_pct = float('nan')

            print(f"({r:>2},{s:>2}) | {I_opt:>14.6f} | {I_przz:>14.6f} | {ratio:>10.6f} | {diff_pct:>10.4f}")


def endpoint_derivative_analysis(
    P1: P1Polynomial,
    N: int,
    sigma: float = 0.5,
) -> None:
    """Analyze the endpoint derivative P₁'(1) contribution.

    P₁(x) = x + x(1-x)·P̃(1-x) implies P₁'(1) = 1 - a₀

    where a₀ = P̃(0) = tilde_coeffs[0].

    This is the key quantity where the a₀=-2 geometry matters.
    """
    # Get P₁'(1)
    P1_deriv_at_1 = P1.eval_deriv(np.array([1.0]), 1)[0]

    print(f"\nEndpoint Derivative Analysis:")
    print(f"  P₁'(1) = {P1_deriv_at_1:.6f}")

    # Compare with formula: P₁'(1) = 1 - a₀
    # where a₀ is the first tilde coefficient
    a0 = P1.tilde_coeffs[0] if P1.tilde_coeffs else 0
    expected = 1 - a0
    print(f"  Expected from formula 1 - a₀ = 1 - ({a0}) = {expected:.6f}")

    # Show contribution to (1, p) atom
    sieve = compute_sieve_arrays(N, include_psi3=False)
    primes = [p for p in range(2, min(20, N)) if sieve.mobius[p] == -1]

    print(f"\n  Impact on (1, p) atoms (I_{{0,1}} coefficient):")
    for p in primes[:5]:
        atom = compute_ratio_atom(1, p, N, sigma, P1, max_order=2)
        if atom:
            I_01 = atom.get_coefficient(0, 1)
            print(f"    (1, {p}): I_{{0,1}} = {I_01:.6f}")

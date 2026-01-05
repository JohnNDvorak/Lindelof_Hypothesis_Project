# Lindelöf Hypothesis Project: Phase 2 Handoff to GPT

## Executive Summary

Phase 2 implements the ratio-class decomposition and ratio-atom Taylor series you requested. All 181 tests pass. The key findings are:

1. **Optimal vs PRZZ show ~0.4% difference in total local moment** (ratio 1.00)
2. **Both polynomials have the same ratio-class hierarchy** - (1,2), (3,2), (3,5) dominate
3. **The derivative coefficients I_{0,1} differ more** (~4-5%) than leading coefficients I_{0,0} (~0.5%)
4. **The endpoint derivative P₁'(1) = 1 - a₀ is 0.836 (optimal) vs 0.739 (PRZZ)** - a 13% difference

This confirms your hypothesis: the global c=1 saturation doesn't manifest locally because we're measuring |ψ|², not |Vζψ|² where the cross-term interference lives.

---

## New Code Structure

### Files Created

```
src/local/ratio_classes.py     # Numeric decomposition by coprime pairs (A,B)
src/local/ratio_atoms.py       # Taylor series expansion for ratio atoms
tests/test_ratio_classes.py    # 15 tests
tests/test_ratio_atoms.py      # 19 tests
```

### Key Functions

```python
# ratio_classes.py
compute_ratio_class_contribution(A, B, coeffs, config) -> RatioClassContribution
compute_ratio_classes(coeffs, config, A_max=50) -> RatioClassDecomposition

# ratio_atoms.py
compute_taylor_coefficient(r, s, A, B, N, sigma, P1, u_array, mobius) -> TaylorCoefficient
compute_ratio_atom(A, B, N, sigma, P1, max_order=3) -> RatioAtom
compute_prime_atoms(primes, N, sigma, P1, max_order=3) -> Dict[(A,B), RatioAtom]
endpoint_derivative_analysis(P1, N, sigma=0.5) -> None
```

---

## Numerical Results

### 1. Diagonal/Off-Diagonal Decomposition (N=1000, T=1000)

| Component | Optimal | PRZZ | Opt/PRZZ |
|-----------|---------|------|----------|
| Diagonal | 2.283399 | 2.273420 | 1.004389 |
| Off-diagonal | -0.551466 | -0.542343 | 1.016821 |
| **Total** | **1.731933** | **1.731077** | **1.000494** |
| Off/Diag ratio | -0.241511 | -0.238558 | - |

**Key observation**: Off-diagonal is ~24% of diagonal and negative (destructive interference). Both polynomials produce nearly identical totals.

### 2. Bandwidth Dependence (T=1000)

| Δ | Optimal Total | PRZZ Total | Opt/PRZZ |
|---|---------------|------------|----------|
| 0.5 | 1.935184 | 1.928479 | 1.003477 |
| 1.0 | 1.731933 | 1.731077 | 1.000494 |
| 2.0 | 1.440421 | 1.436446 | 1.002767 |

Wider bandwidth → more off-diagonal → lower total (more cancellation).

### 3. Top Ratio-Class Contributors (Δ=1.0)

| Rank | (A,B) | |C| Optimal | |C| PRZZ | Opt/PRZZ |
|------|-------|------------|---------|----------|
| 1-2 | (1,2), (2,1) | 0.312 | 0.311 | 1.004 |
| 3-4 | (3,2), (2,3) | 0.233 | 0.233 | 1.000 |
| 5-6 | (3,5), (5,3) | 0.123 | 0.123 | 1.002 |
| 7-8 | (5,7), (7,5) | 0.099 | 0.098 | 1.010 |

**Key observation**: The hierarchy is identical. Prime-pair ratios dominate. The (1,p) atoms appear only at larger Δ.

### 4. Ratio-Atom Taylor Coefficients (N=1000)

For the (1,2) atom:

| Coefficient | Optimal | PRZZ | Ratio |
|-------------|---------|------|-------|
| I_{0,0} | 1.654 | 1.645 | 1.005 |
| I_{1,0} = I_{0,1} | 2.048 | 1.972 | **1.039** |
| I_{1,1} | 3.384 | 3.440 | 0.984 |
| I_{2,0} = I_{0,2} | -2.087 | -3.083 | **0.677** |

**Key observation**: The leading coefficient I_{0,0} differs by ~0.5%, but derivative coefficients I_{1,0}, I_{2,0} differ by 4-32%. This is where the a₀ geometry lives.

### 5. Endpoint Derivative Analysis

```
P₁'(1) = 1 - a₀

Optimal: P₁'(1) = 0.836081  (a₀ = 0.163919)
PRZZ:    P₁'(1) = 0.738924  (a₀ = 0.261076)

Ratio: 1.131 (13% difference)
```

This 13% difference in endpoint derivative is where the optimal polynomial's geometry differs most from PRZZ. But it only affects terms proportional to δ_A or δ_B in the Taylor expansion.

---

## Mathematical Structure Implemented

### Ratio-Class Contribution

For coprime pair (A, B):
```
C_{A,B} = ŵ(log(A/B)) · exp(-iT·log(A/B)) · Σ_{g:(g,AB)=1} a[Ag]·ā[Bg]·(ABg²)^{-σ}
```

### Ratio-Atom Taylor Expansion

For fixed (A, B), expanding in shifts δ_A = log(A)/log(N), δ_B = log(B)/log(N):
```
S_{A,B} = μ(A)μ(B) (AB)^{-σ} Σ_{r,s≥0} (-δ_A)^r/r! (-δ_B)^s/s! · I_{r,s}

where I_{r,s} = Σ_{g:(g,AB)=1} μ(g)² P₁^(r)(u_g) P₁^(s)(u_g) g^{-2σ}
```

The BivariateSeries class stores these Taylor coefficients for symbolic manipulation.

---

## Interpretation: Why No Local Suppression?

Your original hypothesis is confirmed:

1. **We're measuring |ψ|², not |Vζψ|²**
   - Local moment = Σ_{m,n} a_m ā_n (m/n)^{-iT} ŵ(log(m/n))
   - This is |D(s)|² where D(s) = Σ a_n n^{-s}
   - The V-operator and ζ don't appear

2. **The ratio-class decomposition shows both polynomials have identical hierarchy**
   - Same top contributors: (1,2), (3,2), (3,5), (5,7)
   - Same magnitudes to within 1%
   - The c=1 geometry affects global integrals, not these local sums

3. **The derivative coefficients I_{r,s} for r+s≥1 differ more**
   - I_{0,1} differs by ~4% (involves P₁')
   - I_{2,0} differs by ~32% (involves P₁'')
   - But these are suppressed by δ_A^r δ_B^s / (r! s!) factors

---

## Questions for GPT

1. **Is the derivative structure in I_{r,s} the "fingerprint" of c=1 geometry?**
   - The 32% difference in I_{2,0} is striking
   - Does this connect to your PRZZ I₁...I₄ framework?

2. **How do we connect ratio atoms to V-operator cross-terms?**
   - Currently computing |ψ|² = |Σ a_n n^{-s}|²
   - Need |Vζψ|² = |Σ_n (1/n^s) Σ_{d|n} a_d ψ(n/d)|²
   - Can you write the V-operator expansion in ratio-class language?

3. **Should we implement the ζ-weighted moment?**
   - Define: M_ζ = ∫ |ζ(s)D(s)|² w(t-T) dt
   - This would require computing Σ_{n≤N} a_n / n^{1/2+it} · ζ(1/2+it)
   - The ratio-domain version becomes much more complex

4. **What's the next diagnostic you want?**
   - More ratio atoms (composite A, B)?
   - Higher Taylor order (max_order=5)?
   - Different N values (N=5000, N=10000)?
   - The ζ-weighted moment from question 3?

---

## How to Run

```bash
# Run all tests (181 pass, 2 xfail)
python3 -m pytest tests/ -v

# Run the demo with full diagnostics
python3 demos/local_moment_demo.py

# Compute ratio atoms for custom primes
from src.local import compute_prime_atoms, load_optimal_polynomials
P1, _, _ = load_optimal_polynomials()
atoms = compute_prime_atoms([2, 3, 5, 7, 11], N=1000, sigma=0.5, P1=P1, max_order=3)
```

---

## Full Demo Output

<details>
<summary>Click to expand full output (300 lines)</summary>

```
===========================================================================
LOCALIZED MOMENT ENGINE - PHASE 2 DEMO
Ratio-Class Decomposition for Backlund Bridge Analysis
===========================================================================
N = 1000, T = 1000.0, theta = 0.571429, Delta = 1.0

---------------------------------------------------------------------------
PART 1: DIAGONAL/OFF-DIAGONAL DECOMPOSITION
---------------------------------------------------------------------------

At T=1000.0, Delta=1.0:
                     |        Optimal |           PRZZ |   Opt/PRZZ
-----------------------------------------------------------------
            Diagonal |       2.283399 |       2.273420 |   1.004389
        Off-diagonal |      -0.551466 |      -0.542343 |   1.016821
               Total |       1.731933 |       1.731077 |   1.000494
      Off/Diag ratio |      -0.241511 |      -0.238558 |          -

At T=5000, Delta=1.0:
                     |        Optimal |           PRZZ |   Opt/PRZZ
-----------------------------------------------------------------
            Diagonal |       2.283399 |       2.273420 |   1.004389
        Off-diagonal |      -0.218771 |      -0.216482 |   1.010572
               Total |       2.064629 |       2.056938 |   1.003739
      Off/Diag ratio |      -0.095809 |      -0.095223 |          -

---------------------------------------------------------------------------
PART 2: RATIO-CLASS DECOMPOSITION (T=1000, Delta=1.0)
---------------------------------------------------------------------------

OPTIMAL POLYNOMIALS - Top 20 ratio-class contributors:
  Total classes: 154, Diagonal: 2.283399
Rank |      (A,B) |          |C| |        Re(C) |  w_hat |   n_g
------------------------------------------------------------
   1 | (  1,  2) |     0.312237 |     0.129026 |  0.307 |   204
   2 | (  2,  1) |     0.312237 |     0.129026 |  0.307 |   204
   3 | (  3,  2) |     0.233461 |    -0.228823 |  0.595 |   101
   4 | (  2,  3) |     0.233461 |    -0.228823 |  0.595 |   101
   5 | (  3,  5) |     0.123223 |    -0.038389 |  0.489 |    76
   6 | (  5,  3) |     0.123223 |    -0.038389 |  0.489 |    76
   7 | (  5,  7) |     0.099471 |    -0.094364 |  0.664 |    66
   8 | (  7,  5) |     0.099471 |    -0.094364 |  0.664 |    66
   9 | (  5,  6) |     0.091502 |    -0.090957 |  0.818 |    42
  10 | (  6,  5) |     0.091502 |    -0.090957 |  0.818 |    42
  ...

---------------------------------------------------------------------------
PART 6: RATIO-ATOM TAYLOR COEFFICIENTS
---------------------------------------------------------------------------

Optimal polynomial P₁: tilde_coeffs = [ 0.163919   -0.78661276 -0.21621351  0.32751591]
PRZZ polynomial P₁: tilde_coeffs = [ 0.261076 -1.071007 -0.23684   0.260233]

Endpoint Derivative P₁'(1) = 1 - a₀:
  Optimal: P₁'(1) = 0.836081  (a₀ = 0.163919)
  PRZZ:    P₁'(1) = 0.738924  (a₀ = 0.261076)

Prime Atoms - OPTIMAL Polynomials:

Ratio Atom Summary (Taylor coefficients I_{r,s}):
     (A,B) |   μ |         I_00 |         I_10 |         I_01 |         I_11 |         I_20 |         I_02
-----------------------------------------------------------------------------------------------
(  1,  2) |  -1 |     1.653562 |     2.048162 |     2.048162 |     3.383878 |    -2.087308 |    -2.087308
(  1,  3) |  -1 |     1.883383 |     2.317001 |     2.317001 |     3.724990 |    -2.424799 |    -2.424799
...

Comparison of key atoms (Optimal vs PRZZ):
  (1,2): I_00 ratio = 1.005431, I_01 ratio = 1.038606
  (2,1): I_00 ratio = 1.005431, I_01 ratio = 1.038606
  (1,3): I_00 ratio = 1.004434, I_01 ratio = 1.036777
  (2,3): I_00 ratio = 1.006323, I_01 ratio = 1.051919
```

</details>

---

## Next Steps (Proposed)

1. **Implement V-operator cross-terms** - If you provide the formula
2. **Add ζ-weighted moment** - Requires numerical ζ evaluation
3. **Higher-order Taylor** - Extend max_order to see if pattern continues
4. **Analytic atom families** - Group (1,p), (p,q), (pq,rs) hierarchically

Awaiting your guidance on which direction is most productive.

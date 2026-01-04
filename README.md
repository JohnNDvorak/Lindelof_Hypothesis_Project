# Lindelöf Hypothesis Project

**Localized Moment Engine for Band-Limited Analysis of Mollified Dirichlet Polynomials**

This project implements a computational framework for analyzing the *local* behavior of mollified zeta function moments, building on the PRZZ (Pratt–Robles–Zaharescu–Zeindler, 2019) mollifier optimization framework.

---

## Central Goal: The Backlund Bridge

We have proven that at the optimal shift parameter R* = 1.14976..., the global main-term constant satisfies:

```
c(R*) = 1  ⟹  κ = 1 - log(c)/R = 1
```

This represents **exact saturation** of the Levinson-Conrey method with K=3 mollifier pieces.

**The Backlund Bridge Question:** Does the same destructive interference geometry that kills the global main term *also* suppress local energy in band-limited windows?

This codebase provides the tools to answer that question numerically.

---

## Mathematical Framework

### Mollified Dirichlet Polynomials

For mollifier length N = T^θ (typically θ = 4/7), we define:

```
D(s) = Σ_{n≤N} aₙ n^{-s}
```

where the coefficients come from three pieces:

- **ψ₁:** `aₙ^(1) = μ(n) · P₁(uₙ)` — Möbius piece
- **ψ₂:** `aₙ^(2) = (μ*Λ)(n)/log(N) · P₂(uₙ)` — First von Mangoldt convolution
- **ψ₃:** `aₙ^(3) = (μ*Λ*Λ)(n)/(log N)² · P₃(uₙ)` — Second convolution

with `uₙ = log(N/n)/log(N) ∈ [0,1]`.

**Key identity:** `(μ*Λ)(n) = -μ(n)·log(n)` (exact, not asymptotic!)

### Fejér Band-Limited Window

The localized moment uses the Fejér kernel:

```
Frequency: ŵ_Δ(ξ) = max(0, 1 - |ξ|/Δ)     [triangle]
Time:      w_Δ(t) = (Δ/2π) · sinc²(Δt/2π)  [sinc-squared]
```

**Key properties:**
- Compact frequency support: |ξ| ≤ Δ
- Non-negative: w_Δ(t) ≥ 0 for all t
- First zero at t = 2π/Δ
- Decay: w_Δ(t) ~ t⁻² for large |t|

### Localized Moment

```
M(T, Δ, σ) = ∫_{-∞}^{∞} w_Δ(t - T) |D(σ + it)|² dt
```

This can be computed equivalently in the **ratio domain**:

```
M(T, Δ, σ) = Σ_{n,m≤N} aₙ ā_m (nm)^{-σ} ŵ_Δ(log(n/m)) e^{-iT log(n/m)}
```

The package implements both and validates their consistency.

---

## Installation

```bash
git clone <repo-url>
cd Lindelof_Hypothesis_Project
pip install numpy pytest
```

### Validate Installation

```bash
python -m pytest tests/test_local_*.py -v
# Expected: 74 tests passing
```

---

## Quick Start

### Basic Usage

```python
from src.local import LocalEngine

# Create engine with optimal polynomials (c=1 saturation coefficients)
engine = LocalEngine.from_config(
    N=1000,           # Mollifier length
    theta=4/7,        # PRZZ exponent
    which_psi=(True, False, False),  # ψ₁ only for MVP
    use_optimal=True, # Use c=1 polynomials (vs PRZZ baseline)
)

# Compute localized moment at T=1000 with bandwidth Δ=1.0
result = engine.compute_moment(T=1000, Delta=1.0)
print(f"Localized moment: {result.moment:.6f}")
print(f"Grid points: {len(result.t_grid)}")
```

### Compare Optimal vs PRZZ Baseline

```python
# Optimal polynomials (c=1 at R*=1.14976)
engine_opt = LocalEngine.from_config(N=1000, use_optimal=True)
moment_opt = engine_opt.compute_moment(T=1000, Delta=1.0).moment

# PRZZ baseline polynomials
engine_przz = LocalEngine.from_config(N=1000, use_optimal=False)
moment_przz = engine_przz.compute_moment(T=1000, Delta=1.0).moment

print(f"Optimal / PRZZ ratio: {moment_opt / moment_przz:.6f}")
```

### Consistency Check (Time vs Ratio Domain)

```python
# Use small N for O(N²) ratio-domain computation
engine = LocalEngine.from_config(N=100)
time_mom, ratio_mom, passed = engine.verify_consistency(T=100, Delta=0.5, rtol=0.01)
print(f"Time-domain:  {time_mom:.6f}")
print(f"Ratio-domain: {ratio_mom:.6f}")
print(f"Status: {'PASS' if passed else 'FAIL'}")
```

### Run the Demo

```bash
python demos/local_moment_demo.py
```

**Sample output:**
```
LOCALIZED MOMENT ENGINE - MVP DEMO
N = 1000, T = 1000.0, theta = 0.571429, Delta = 1.0

Localized moment (optimal): 1.6750412116
Localized moment (PRZZ):    1.6744908852
Optimal / PRZZ ratio: 1.000329 (+0.03%)

SWEEP OVER BANDWIDTHS (fixed T=1000)
   Delta |        Optimal |           PRZZ |    Ratio
     0.5 |       1.871490 |       1.865348 |   1.0033
     1.0 |       1.675041 |       1.674491 |   1.0003
     2.0 |       1.381418 |       1.377621 |   1.0028
```

---

## Architecture

### Directory Structure

```
Lindelof_Hypothesis_Project/
├── src/
│   ├── local/                    # NEW: Localized moment engine
│   │   ├── __init__.py           # LocalEngine class + exports
│   │   ├── fejer.py              # Fejér band-limited window kernel
│   │   ├── sieve.py              # Prime sieve, μ(n), Λ(n), convolutions
│   │   ├── mollifier_coeffs.py   # ψ₁, ψ₂, ψ₃ coefficient generators
│   │   ├── dirichlet_poly.py     # D(s) evaluator (brute + incremental)
│   │   └── local_moment.py       # Localized moment computation
│   │
│   ├── polynomials.py            # P₁, P₂, P₃, Q polynomial classes
│   ├── kappa_engine.py           # Global κ computation engine
│   ├── series.py                 # Truncated formal series
│   ├── series_bivariate.py       # Bivariate series for ratio-atoms
│   ├── quadrature.py             # Gauss-Legendre integration
│   └── ...                       # Other PRZZ infrastructure
│
├── data/
│   ├── optimal_polynomials.json  # c=1 saturation coefficients (PRIMARY)
│   ├── przz_parameters.json      # PRZZ baseline coefficients
│   └── ...
│
├── tests/
│   ├── test_local_fejer.py       # 13 tests
│   ├── test_local_sieve.py       # 15 tests
│   ├── test_local_mollifier_coeffs.py  # 14 tests
│   ├── test_local_dirichlet_poly.py    # 15 tests
│   ├── test_local_moment.py      # 17 tests
│   └── ...                       # Total: 74 new tests
│
├── demos/
│   └── local_moment_demo.py      # MVP demonstration
│
├── proofs/                       # Algebraic verification of c(R*)=1
└── paper/                        # LaTeX source
```

### Module Descriptions

| Module | Purpose |
|--------|---------|
| `fejer.py` | `FejerKernel` class with `w_time()`, `w_freq()`, `from_first_zero()` |
| `sieve.py` | `SieveArrays` dataclass with primes, μ, Λ, (μ*Λ), (μ*Λ*Λ) |
| `mollifier_coeffs.py` | `MollifierCoeffs` dataclass, loads polynomials from JSON |
| `dirichlet_poly.py` | `evaluate_brute()` and `evaluate_incremental()` for D(s) |
| `local_moment.py` | `compute_local_moment()`, `compute_ratio_domain_moment()`, `verify_moment_consistency()` |
| `__init__.py` | `LocalEngine` class unifying all components |

---

## Polynomial Coefficients

### Optimal Polynomials (c=1 saturation) — PRIMARY

From `data/optimal_polynomials.json`:

```python
P₁_tilde: [0.164, -0.787, -0.216, 0.328]   # (1-x) basis
P₂_tilde: [1.006, -0.229, -0.194]          # monomial basis
P₃_tilde: [-1.333, -2.409, -0.151]         # monomial basis
```

These achieve **c = 1** at R* = 1.14976, giving **κ = 1** (100% density on critical line).

### PRZZ Baseline Polynomials

From `data/przz_parameters.json`:

```python
P₁_tilde: [0.261, -1.071, -0.237, 0.260]
P₂_tilde: [1.048, 1.320, -0.940]
P₃_tilde: [0.523, -0.687, -0.050]
```

These give κ ≈ 0.4173 at R = 1.3036 (the published PRZZ result).

---

## Key Formulas

### Global κ Computation (existing infrastructure)

```
κ = 1 - log(c) / R

c(R) = S₁₂(+R) + M · S₁₂(-R) + S₃₄(+R)

M = G · M₀,  where M₀ = exp(R) + 5  (exact for K=3)
```

### Arithmetic Functions

```python
# Exact identity (not asymptotic!)
(μ * Λ)(n) = -μ(n) · log(n)

# Computed via sparse prime-power convolution
(μ * Λ * Λ)(n) = Σ_{pk·qj | n} μ(n/(pk·qj)) · log(p) · log(q)
```

### Dirichlet Polynomial Evaluation

**Brute force (O(N × |grid|)):**
```python
D(σ + it) = Σₙ aₙ · n^{-σ} · exp(-it·log(n))
```

**Incremental phase update (faster for uniform grids):**
```python
zₙ(t_{k+1}) = zₙ(t_k) · rₙ   where rₙ = exp(-i·dt·log(n))
```

### Localized Moment

**Time domain (numerical quadrature):**
```python
M ≈ Σ_k w_Δ(t_k - T) · |D(σ + it_k)|² · Δt
```

**Ratio domain (exact, O(N²)):**
```python
M = Σ_{n,m} aₙ·ā_m · (nm)^{-σ} · ŵ_Δ(log(n/m)) · exp(-iT·log(n/m))
```

---

## API Reference

### FejerKernel

```python
from src.local import FejerKernel

kernel = FejerKernel(Delta=1.0)
kernel.w_time(t)           # Evaluate at time t
kernel.w_freq(xi)          # Evaluate at frequency ξ
kernel.first_zero          # Location of first zero = 2π/Δ
kernel.effective_support() # Approximate truncation point

FejerKernel.from_first_zero(L)  # Create with first zero at t=L
```

### LocalEngine

```python
from src.local import LocalEngine

engine = LocalEngine.from_config(
    N=1000,                           # Mollifier length
    theta=4/7,                        # Exponent
    sigma=0.5,                        # Real part (critical line)
    which_psi=(True, True, False),    # (ψ₁, ψ₂, ψ₃)
    use_optimal=True,                 # Use c=1 polynomials
)

result = engine.compute_moment(T, Delta, n_halfwidth=4.0, n_points_per_zero=20)
# Returns LocalMomentResult with: moment, T, Delta, t_grid, weights, D_squared

time_mom, ratio_mom, passed = engine.verify_consistency(T, Delta, rtol=0.01)
```

### Low-Level Functions

```python
from src.local import (
    compute_sieve_arrays,     # SieveArrays with all arithmetic functions
    compute_mollifier_coeffs, # MollifierCoeffs with a₁, a₂, a₃
    evaluate_dirichlet_poly,  # DirichletPolyResult with values, |D|²
    compute_local_moment,     # LocalMomentResult
    verify_moment_consistency,
)
```

---

## Test Suite

```bash
# Run all local module tests (74 tests)
python -m pytest tests/test_local_*.py -v

# Run specific test file
python -m pytest tests/test_local_fejer.py -v
```

### Test Coverage

| Module | Tests | Key Validations |
|--------|-------|-----------------|
| fejer | 13 | First zero, normalization, positivity, decay |
| sieve | 15 | Prime count π(100)=25, μ values, Λ values, identities |
| mollifier_coeffs | 14 | Boundary values u[1]=1, polynomial loading, sparsity |
| dirichlet_poly | 15 | Analytical verification, brute vs incremental equivalence |
| local_moment | 17 | Constant poly normalization, time/ratio consistency |

---

## Repurposed PRZZ Infrastructure

From the parent PRZZ optimization project:

| File | Used For |
|------|----------|
| `polynomials.py` | `P1Polynomial`, `PellPolynomial` with `eval()`, `eval_deriv()` |
| `series_bivariate.py` | Future ratio-atom expansions (Phase 2) |
| `quadrature.py` | Gauss-Legendre utilities |
| `kappa_engine.py` | Design pattern template for `LocalEngine` |

---

## Next Steps (Backlund Bridge Analysis)

1. **Sweep (T, Δ)** to find regions of maximum local energy suppression
2. **Add ψ₂, ψ₃** contributions for full mollifier analysis
3. **Implement `ratio_atoms.py`** using `BivariateSeries` for per-prime local factors:
   - Atoms (A,B) with squarefree integers
   - Per-prime local factor series F_p(x,y) truncated to (2,2)
   - Weight by ŵ_Δ(log(A/B))
4. **Compare** local suppression patterns with global c=1 saturation geometry
5. **Validate** whether optimal polynomials suppress local energy uniformly or selectively

---

## PRZZ Benchmark Reproduction (Legacy)

The existing `kappa_engine.py` reproduces PRZZ benchmarks:

| Benchmark | R | Target | Computed | Error |
|-----------|------|--------|----------|-------|
| κ | 1.3036 | 0.417293962 | 0.417295933 | 0.0005% |
| κ* | 1.1167 | 0.407511457 | 0.407509790 | 0.0004% |

```python
from src.kappa_engine import KappaEngine

engine = KappaEngine.from_przz_kappa()
result = engine.compute_kappa()
print(f"κ = {result.kappa:.10f}")  # 0.4172959328
```

---

## References

- Pratt, Robles, Zaharescu, Zeindler (2019): "More Than Five-Twelfths of the Zeros of ζ Are on the Critical Line" — [arXiv:1802.10521](https://arxiv.org/abs/1802.10521)
- Dvorak (2025): "Exact Saturation of the Levinson-Conrey Method: c = 1 Achieved"
- See `proofs/coefficients_final.json` for the exact z-basis coefficients
- See `proofs/README.md` for the algebraic proof structure

---

## License

Research code for academic reproducibility.

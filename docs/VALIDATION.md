# Validation Guide

## How to Verify Reproducibility

### Step 1: Run Tests

```bash
cd przz-reproducibility
python -m pytest tests/ -v
```

Expected: All tests pass

### Step 2: Check PRZZ Benchmark Reproduction

```python
from src.kappa_engine import validate_przz_benchmarks

validation = validate_przz_benchmarks(tolerance_pct=0.01)

# kappa benchmark (R=1.3036)
print(f"kappa computed: {validation['kappa']['computed']:.10f}")
print(f"kappa target:   {validation['kappa']['target']:.10f}")
print(f"kappa gap:      {validation['kappa']['gap_pct']:+.6f}%")
# Expected gap: ~0.0005%

# kappa* benchmark (R=1.1167)
print(f"kappa* computed: {validation['kappa_star']['computed']:.10f}")
print(f"kappa* target:   {validation['kappa_star']['target']:.10f}")
print(f"kappa* gap:      {validation['kappa_star']['gap_pct']:+.6f}%")
# Expected gap: ~0.0004%
```

### Step 3: Verify c(R*) = 1

```python
from src.kappa_engine import KappaEngine

# At saturation point
R_star = 1.14976023153715
engine = KappaEngine.from_przz_kappa()
engine.R = R_star
result = engine.compute_kappa()

print(f"c(R*) = {result.c:.15f}")
# Expected: 1.000000000000000 (to machine precision)

print(f"|c - 1| = {abs(result.c - 1):.2e}")
# Expected: < 5e-14
```

### Step 4: Quadrature Convergence

Verify results are stable under quadrature refinement:

```python
from src.kappa_engine import KappaEngine

for n_quad in [60, 80, 100]:
    engine = KappaEngine.from_przz_kappa(n_quad=n_quad)
    result = engine.compute_kappa()
    print(f"n_quad={n_quad}: kappa={result.kappa:.10f}, c={result.c:.10f}")
```

Expected: kappa and c stable to ~1e-8 across refinements

## Key Values to Check

| Quantity | Expected Value | Tolerance |
|----------|----------------|-----------|
| kappa (R=1.3036) | 0.417293962 | 0.01% |
| kappa* (R=1.1167) | 0.407511457 | 0.01% |
| c(R*=1.14976) | 1.0 | 1e-12 |
| M_0 at R=1.3036 | exp(1.3036) + 5 = 8.682 | exact |
| g_I1 | ~1.00095 | 0.1% |
| g_I2 | ~1.01944 | 0.01% |

## Regression Lock

The file `data/golden_kappa_production.json` contains locked reference values. Any code changes that alter these values indicate a regression.

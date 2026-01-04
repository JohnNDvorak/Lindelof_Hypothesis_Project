# Methodology

## The kappa Bound

The Levinson-type bound for the proportion of Riemann zeta zeros on the critical line:

```
kappa >= 1 - log(c) / R
```

where:
- R is the shift parameter (sigma_0 = 1/2 - R/log(T))
- c is the main-term constant in the asymptotic for the mollified mean square

## Computing c

The main-term constant c is computed via the **mirror assembly formula**:

```
c(R) = S_12(+R) + M * S_12(-R) + S_34(+R)
```

where:
- S_12 = I_1 + I_2 (integrals summed over all (ell_1, ell_2) pairs)
- S_34 = I_3 + I_4 (integrals summed over all pairs)
- M = G * M_0 is the full mirror multiplier

### Mirror Multiplier Components

**Structural base (EXACT):**
```
M_0 = exp(R) + (2K - 1)
```
For K=3: M_0 = exp(R) + 5

**Correction factor (DERIVED):**
```
G = f_I1 * g_I1 + (1 - f_I1) * g_I2
```

where:
- g_I1 ~ 1.00095 (log factor self-correction)
- g_I2 = 1 + theta*(2-theta) / (2*K*(2*K+1)) (EXACT variance structure)
- f_I1 = I_1(-R) / (I_1(-R) + I_2(-R)) (I_1 fraction at -R)

For K=3, theta=4/7: G ~ 1.014

## The Four Integrals

For each pair (ell_1, ell_2) with 1 <= ell_1 <= ell_2 <= K:

| Integral | Derivative | Description |
|----------|------------|-------------|
| I_1 | d^2/dxdy | Mixed derivative kernel with log factor |
| I_2 | none | Main term (no derivatives) |
| I_3 | d/dx | Single x-derivative |
| I_4 | d/dy | Single y-derivative |

All four integrals involve the polynomial products P_{ell_1}(u) * P_{ell_2}(u) integrated with kernels that depend on the shift parameter R.

## PRZZ Polynomials

The PRZZ framework uses constrained polynomials:

**P_1:** Constrained form x + x(1-x)*P_tilde (enforces P_1(0)=0, P_1(1)=1)

**P_2, P_3:** Form x*P_tilde (enforces P_ell(0)=0)

**Q:** In (1-2x)^k basis with Q(0)=1 enforced

## Pair Weights

Symmetry factors:
- Diagonal (1,1), (2,2), (3,3): weight 1
- Off-diagonal: weight 2 (counted twice)

Factorial normalization: 1/(ell_1! * ell_2!)

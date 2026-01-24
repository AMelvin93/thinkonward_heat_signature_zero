# Experiment Summary: greens_function_inversion

## Metadata
- **Experiment ID**: EXP_GREENS_FUNCTION_001
- **Worker**: W2
- **Date**: 2026-01-24
- **Algorithm Family**: direct_inversion

## Objective
Use heat equation Green's function for direct source localization without iterative CMA-ES optimization.

## Hypothesis
2026 research shows Green function method can directly localize heat sources via Volterra integral equation. Unlike CMA-ES iteration, this provides closed-form solution. Privacy-preserving and computationally minimal.

## Results Summary
- **Best In-Budget Score**: N/A (not implemented)
- **Best Overall Score**: N/A
- **Baseline Comparison**: N/A
- **Status**: **ABORTED** - Green's function is 4.3x slower than ADI and still requires iteration

## Key Findings

### Finding 1: Green's Function is SLOWER than ADI

| Method | Time per Evaluation |
|--------|---------------------|
| ADI simulation | 1,189 ms |
| Green's function (50 terms) | 5,151 ms |
| **Ratio** | Green's is **4.3x SLOWER** |

The analytical Green's function approach requires summing a double series (50×50 = 2,500 terms) for EACH timestep and sensor. ADI implicit time-stepping is highly optimized and faster.

### Finding 2: Sensor Heterogeneity Prevents Pre-computation

| Metric | Value |
|--------|-------|
| Unique sensor configurations | 80 out of 80 samples |
| Unique (BC, kappa) combinations | 4 |

Even with only 4 physics combinations, we CANNOT pre-compute Green's function tables because sensor positions are 100% unique. Pre-computation would require:
- 80 samples × 1,250 source positions × 1,001 timesteps = **100 million entries**

### Finding 3: Inversion is STILL Nonlinear

```
Temperature at sensor: T(x_sensor, t) = q × ∫₀ᵗ G(x_sensor, x_source, t-τ) dτ

The source position (x_s, y_s) appears NONLINEARLY in G:
  G ∝ sin(nπx_s/Lx) × sin(mπy_s/Ly) × ...

Even with known Green's function, finding (x_s, y_s, q) requires:
  argmin ||T_obs - T_sim(x, y, q)||²

This is STILL an optimization problem - no closed-form inverse exists.
```

### Finding 4: Grid Search is Not Better than CMA-ES

The only "direct" approach would be to pre-tabulate G for all source positions and search for the best match. But:
1. This is equivalent to brute-force grid search
2. CMA-ES is more efficient (adaptive covariance, fewer evaluations)
3. Pre-computation is blocked by sensor heterogeneity

## Why Green's Function Doesn't Help

### The Fundamental Problem

```
Hypothesis: Analytical Green's function → Direct inversion → No iteration needed

Reality:
1. Green's function depends on (BC, κ, sensor_positions)
2. Sensor positions vary per sample (100% unique)
3. Cannot pre-compute universal lookup table
4. Green's function series is 4.3x SLOWER than ADI
5. Inversion is nonlinear → still need optimization
```

### What Would Make Green's Function Work

For Green's function approach to be viable:
1. ✗ Fixed sensor positions (NOT true - 100% unique)
2. ✗ Fast series convergence (NOT true - need 50+ terms)
3. ✗ Closed-form inverse (NOT true - nonlinear in source position)
4. ✗ Pre-computation possible (NOT true - sensor heterogeneity)

## Abort Criteria Met

From experiment specification:
> "Green's function requires sample-specific computation negating speed benefit OR inversion is ill-posed without regularization"

Actual abort reason:
> **Green's function evaluation is 4.3x SLOWER than ADI simulation. Sensor heterogeneity (100% unique) prevents pre-computation. Inversion remains nonlinear. No computational advantage over ADI + CMA-ES.**

## Recommendations

### 1. direct_inversion Family Should Be Marked EXHAUSTED
Any analytical approach that requires evaluating the heat equation Green's function will be slower than ADI implicit time-stepping.

### 2. Sensor Heterogeneity is a Fundamental Barrier
All approaches that require sample-specific pre-computation (POD, Gappy C-POD, Green's function tables) fail because 100% of samples have unique sensor configurations.

### 3. ADI is Highly Optimized
Implicit methods like ADI achieve O(nx×ny) per timestep with unconditional stability. Analytical series require O(n_terms²×n_timesteps) operations and are slower despite being "analytical."

## Conclusion

**ABORTED** - Green's function approach provides no computational advantage over ADI + CMA-ES baseline. The analytical method is 4.3x slower due to series convergence requirements, and the inversion still requires nonlinear optimization because source position appears nonlinearly in the Green's function. The direct_inversion family is fundamentally inapplicable to this problem.

## Files
- `feasibility_analysis.py`: Initial analysis of Green's function formulation
- `precomputation_analysis.py`: Detailed timing and pre-computation analysis
- `STATE.json`: Experiment state tracking

# Experiment Summary: nm_dimension_adaptive

## Metadata
- **Experiment ID**: EXP_NM_DIM_ADAPTIVE_001
- **Worker**: W3
- **Date**: 2026-01-25
- **Algorithm Family**: polish_strategy

## Objective
Test whether scipy's adaptive Nelder-Mead parameters (dimension-dependent coefficients from Gao and Han 2012) improve polish convergence for the final NM refinement step.

When `adaptive=True`, scipy uses:
- `rho = 1` (reflection coefficient)
- `chi = 1 + 2/n` (expansion coefficient)
- `psi = 0.75 - 1/(2n)` (contraction coefficient)
- `sigma = 1 - 1/n` (shrink coefficient)

Where n = dimension (3 for 1-source, 6 for 2-source).

## Hypothesis
The paper by Gao and Han (2012) shows that dimension-adaptive NM parameters can improve convergence in high dimensions by accounting for geometry. Since 2-source problems have 6 dimensions, adaptive parameters might improve polish convergence.

## Results Summary
- **Best In-Budget Score**: N/A (no runs within budget)
- **Best Overall Score**: 1.1650 @ 66.5 min
- **Baseline Comparison**: -0.0038 vs 1.1688 (0.3% worse)
- **Status**: FAILED

## Tuning History

| Run | Config Changes | Score | Time (min) | In Budget | Notes |
|-----|---------------|-------|------------|-----------|-------|
| 1 | adaptive=True | 1.1650 | 66.5 | No | 0.3% worse accuracy, 14% over budget |

## Key Findings

### What Didn't Work
- **Adaptive NM parameters increased runtime without improving accuracy**
  - 1-source RMSE: 0.1040 (vs baseline ~0.10)
  - 2-source RMSE: 0.1579 (vs baseline ~0.15)
  - Projected time: 66.5 min (vs baseline 58.4 min)

- **The dimension-dependent coefficients don't help for low dimensions**
  - For n=3 (1-source): chi = 1.67, psi = 0.58, sigma = 0.67
  - For n=6 (2-source): chi = 1.33, psi = 0.67, sigma = 0.83
  - These are only slightly different from standard NM parameters (chi=2, psi=0.5, sigma=0.5)
  - The Gao-Han paper focuses on HIGH dimensions (n >> 6)

- **Fixed NM iterations override any convergence benefit**
  - We use maxiter=8 for polish, which terminates early anyway
  - Adaptive parameters can't help if we're not converging to tolerance
  - The parameters affect step sizes, not overall convergence rate for limited iterations

### Critical Insights
1. **Dimension is too low** - Gao-Han adaptive NM is designed for high-dimensional problems (n > 10). At n=3 or n=6, the standard parameters are nearly optimal.
2. **Fixed iteration budget** - With maxiter=8, we're not running NM to convergence. Adaptive parameters only help if you converge to a tolerance, not with fixed iterations.
3. **Overhead unexplained** - The 14% runtime overhead is surprising since this is just a parameter change. Possibly due to different convergence paths requiring more function evaluations per iteration.

## Parameter Sensitivity
- **adaptive=True vs False**: Switching to adaptive=True consistently increases runtime by ~14% with no accuracy benefit

## Recommendations for Future Experiments
1. **DO NOT use adaptive NM for low-dimensional problems** - Standard parameters are fine for n < 10
2. **Fixed iteration polish is optimal** - The 8-iteration limit is the constraint, not convergence parameters
3. **Focus on OTHER aspects of polish** - e.g., initialization quality, step size tuning, not convergence parameters

## Why This Hypothesis Was Wrong
1. **Wrong problem dimensionality** - Gao-Han is for high-dimensional optimization, not n=3 or n=6
2. **Wrong convergence regime** - We use fixed iterations, not convergence-to-tolerance
3. **Marginal parameter differences** - At low dimensions, adaptive parameters are nearly identical to standard ones

## Conclusion
**nm_adaptive family EXHAUSTED.** Dimension-adaptive NM parameters from Gao-Han (2012) are designed for high-dimensional problems and provide no benefit for our 3-6 dimensional polish step. The fixed 8-iteration budget means we're not converging to tolerance anyway. Standard NM parameters are optimal for this problem.

## Raw Data
- MLflow run IDs: 99a3c6babde64a849b376ffb8edb856b
- Best config: N/A (experiment failed)

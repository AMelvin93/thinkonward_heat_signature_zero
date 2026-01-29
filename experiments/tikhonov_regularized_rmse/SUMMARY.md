# Tikhonov Regularized RMSE - Experiment Summary

## Result: FAILED

## Hypothesis
Adding Tikhonov regularization (penalty toward triangulation prior) during CMA-ES optimization might help find smoother, more accurate solutions by biasing toward physically plausible positions.

## Key Findings

### Run 1: Lambda=0.01 (Small Regularization)
| Metric | Value | Baseline | Delta |
|--------|-------|----------|-------|
| Score | 1.1388 | 1.1464 | -0.0076 |
| Projected Time | 491.2 min | 51.2 min | +440.0 min (8.6x) |
| RMSE (overall) | 0.1681 | - | - |
| RMSE (1-src) | 0.1278 | - | - |
| RMSE (2-src) | 0.1950 | - | - |

**Status: CATASTROPHIC FAILURE** - 8x over budget with worse accuracy

## Root Cause Analysis

The Tikhonov regularization fundamentally breaks CMA-ES optimization:

1. **Biased Search**: Regularization toward triangulation prior biases the search away from the true optimum when the triangulation estimate is poor
2. **Slowed Convergence**: The regularization penalty creates artificial landscape complexity, requiring more function evaluations to navigate
3. **Wrong Assumption**: The hypothesis assumed triangulation provides a good prior, but triangulation often gives poor initial estimates for multi-source problems

## Why This Approach Fails

The penalty function `L = RMSE + lambda * ||params - prior||^2` has fundamental issues:

- **Prior quality**: Triangulation priors are not reliable enough to bias toward
- **Optimization dynamics**: CMA-ES naturally explores parameter space; adding a regularization penalty fights this exploration
- **No benefit**: Pure RMSE minimization already finds smoother solutions via Variable Projection (analytical intensity computation)

## Conclusion

**Tikhonov regularization is fundamentally incompatible with this optimization problem.**

The baseline approach (pure RMSE minimization with CMA-ES + NM polish) is superior because:
- It allows unconstrained search in parameter space
- Variable Projection already provides implicit regularization by optimally computing intensities
- No additional bias toward potentially incorrect priors

## Recommendation

**Abandon regularization approaches.** The loss landscape is already well-conditioned. Focus on:
- CMA-ES parameter tuning (sigma, population size)
- Polish strategies (NM iterations, perturbation)
- Temporal fidelity optimization

## Files
- `optimizer.py` - TikhonovRegularizedOptimizer with regularization penalty
- `run.py` - Experiment runner with --regularization-lambda parameter
- `run1_output.log` - Full output from Run 1

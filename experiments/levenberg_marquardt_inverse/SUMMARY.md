# Experiment Summary: levenberg_marquardt_inverse

## Metadata
- **Experiment ID**: EXP_LEVENBERG_MARQUARDT_001
- **Worker**: W1
- **Date**: 2026-01-22
- **Algorithm Family**: nonlinear_least_squares

## Objective
Test Levenberg-Marquardt (LM) algorithm via scipy.optimize.least_squares as an alternative to CMA-ES for inverse heat source identification. LM is the classic method for inverse heat problems with proven fast quadratic convergence near the optimum.

## Hypothesis
LM is specifically designed for nonlinear least squares (NLS), which is exactly what our RMSE minimization is. It should converge faster than CMA-ES since it uses Jacobian-based gradient descent with adaptive damping.

## Results Summary
- **Best In-Budget Score**: 1.0350 @ 56 min (with heavily reduced parameters)
- **Best Overall Score**: 1.1210 @ 109.4 min (still not as good as baseline)
- **Baseline Comparison**: -0.1012 vs 1.1362 @ 39 min (9% worse score, 44% more time)
- **Status**: FAILED - Fundamentally unsuitable for this problem

## Tuning History

| Run | Config | Score | Time (min) | In Budget | Notes |
|-----|--------|-------|------------|-----------|-------|
| 1 | 5 starts, 50 nfev, 8 polish | 1.1172 | 197.3 | No | Works but 5x slower |
| 2 | 2 starts, 20 nfev, 3 polish | 1.0350 | 56.0 | Yes | Time OK but -0.10 score |
| 3 | 4 starts, 15 nfev, 5 polish | 1.1210 | 109.4 | No | Better score, 2.8x over budget |
| 4 | 2 starts, 10 nfev, 8 polish | 0.9897 | 71.5 | No | Local minima traps, catastrophic failures |

## Key Findings

### Why LM Failed

1. **Local optimizer limitation**: LM is a local optimizer that converges to the nearest local minimum. The RMSE landscape has multiple local minima, and LM gets stuck.

2. **Expensive Jacobian computation**: LM uses finite differences for the Jacobian, requiring:
   - 3 simulations per iteration for 1-source (1 residual + 2 gradient components)
   - 5 simulations per iteration for 2-source (1 residual + 4 gradient components)
   - With 10-30 iterations needed, this becomes 30-150 simulations per sample

3. **Multi-start doesn't help enough**: Even with multiple starting points, the computational overhead is too high. CMA-ES explores more efficiently with its population-based approach.

4. **Trade-off impossible**:
   - More iterations = better accuracy but exceeds time budget
   - Fewer iterations = faster but gets stuck in local minima

### Comparison with CMA-ES

| Aspect | CMA-ES (baseline) | Levenberg-Marquardt |
|--------|------------------|---------------------|
| Sample efficiency | 20-36 evals | 100-200 evals |
| Global vs local | Global (population-based) | Local (gradient-based) |
| Robustness | High (handles multi-modal) | Low (gets stuck) |
| Convergence speed | O(1) per eval | O(n) per iteration (Jacobian) |

### Critical Insight

**The fundamental mismatch**: LM is designed for problems where function evaluation is cheap and you want fast convergence. Our problem has expensive simulations (~0.5-1s each) and requires global search due to multi-modal landscape. CMA-ES is a better fit because:

1. It uses population-based search (implicit parallelism within sample)
2. No Jacobian computation needed
3. Naturally handles multi-modal landscapes
4. More sample-efficient for expensive objectives

## Parameter Sensitivity

- **n_multi_starts**: More starts help accuracy but add linear time overhead
- **max_nfev**: Critical parameter - too low causes convergence failure, too high exceeds budget
- **nm_polish_iters**: NM polish helps but can't fix fundamentally bad LM convergence

## Recommendations for Future Experiments

1. **Do NOT pursue Levenberg-Marquardt further** - the local optimizer nature is fundamentally incompatible with this problem's multi-modal landscape.

2. **Trust-Region Reflective (TRF) would likely have the same issue** - it's also a local optimizer, just with better bound handling.

3. **The nonlinear_least_squares family should be marked FAILED** - local gradient-based methods don't work for this problem.

4. **Focus on global optimization methods** - CMA-ES, evolutionary strategies, or hybrid approaches that maintain global search.

5. **If gradient methods are desired, need adjoint gradients** - The only way gradient-based methods could work is with O(1) gradient computation via adjoint method (EXP_CONJUGATE_GRADIENT_001), not finite differences.

## Technical Details

### Files Created
- `optimizer.py`: LevenbergMarquardtOptimizer class with multi-start, bounds projection
- `run.py`: Run script with MLflow logging

### Implementation Notes
- Used scipy.optimize.least_squares with method='lm'
- LM doesn't support bounds - used projection to feasible region
- Residual function returns (Y_predicted - Y_observed) vector
- Intensity (q) computed analytically via least squares (same as baseline)

## Raw Data
- No MLflow runs logged (quick iterations without --no-mlflow flag during tuning)
- Best in-budget: Run 2 with config {n_multi_starts: 2, max_nfev: 20, nm_polish_iters: 3}

## Conclusion

**Levenberg-Marquardt is fundamentally unsuitable for the inverse heat source problem** due to:
1. Local optimizer nature vs multi-modal RMSE landscape
2. Expensive finite difference Jacobian vs limited simulation budget
3. Poor sample efficiency compared to evolutionary methods

CMA-ES remains the superior choice for this expensive, multi-modal optimization problem. The nonlinear_least_squares family should be marked as FAILED.

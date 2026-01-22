# Differential Evolution Experiment Summary

## Experiment ID: EXP_DIFFERENTIAL_EVOLUTION_001
**Status:** FAILED
**Worker:** W1
**Family:** differential_evolution

## Hypothesis
Differential Evolution uses mutation and crossover instead of CMA-ES's covariance adaptation. This may find different optima or converge faster for the inverse heat source problem.

## Results

### Tuning Runs

| Config | Samples | Score | Time (min) | Notes |
|--------|---------|-------|------------|-------|
| No temporal fidelity: iter 8/12, pop 3 | 20 | 1.0057 | 109.6 | Way too slow |
| With temporal: iter 10/15, pop 5, refine 8 | 20 | 1.0229 | 63.7 | Still slow, poor accuracy |
| Reduced: iter 5/8, pop 3, refine 4 | 20 | 1.1177 | 31.4 | Faster but worse accuracy |
| Same on 40 samples | 40 | 1.0989 | 44.6 | 2-src uses 330+ sims |
| Minimal: iter 4/5, pop 2, refine 3 | 40 | 1.1250 | 40.8 | Outlier sample |
| **BEST: iter 5/6, pop 2, refine 4** | 40 | **1.1325** | **35.2** | -0.0037 score, -3.8 min |
| More iters: iter 5/7, pop 2, refine 5 | 40 | 1.1241 | 38.5 | More iters didn't help |

### Best In-Budget Configuration
- **Config:** maxiter_1src=5, maxiter_2src=6, popsize=2, refine_maxiter=4
- **Score:** 1.1325
- **Projected Time:** 35.2 min
- **vs Baseline:** -0.0037 score, -3.8 min (1.1362 @ 39 min)

### Per-Source-Count Analysis (40-sample best run)
- **1-source:** RMSE 0.1539 (n=32) - slightly worse than baseline
- **2-source:** RMSE 0.1727 (n=8) - comparable to baseline

## Why DE Failed

### 1. CMA-ES Covariance Adaptation is Superior
CMA-ES explicitly learns the local curvature of the fitness landscape through covariance matrix adaptation. For the inverse heat problem with correlated parameters (x, y positions), this is critical.

DE's mutation and crossover operators are simpler:
- Mutation adds scaled differences between population members
- Crossover exchanges components between solutions
- No explicit learning of parameter correlations

### 2. DE Population Size Overhead
DE's effective population size is `popsize * (ndim + 1)`:
- 1-source (2D): popsize=2 means 6 individuals
- 2-source (4D): popsize=2 means 10 individuals

This causes 2-source to use 3x more function evaluations than 1-source, making it harder to balance the time budget.

### 3. Sample Efficiency
Despite similar-looking population-based search:
- CMA-ES uses ~60-90 simulations per sample optimally
- DE needs ~90-100 simulations to approach similar accuracy
- DE's convergence is less directed without covariance learning

## Implementation Notes
- Added temporal fidelity (40% timesteps) to match baseline
- Used coarse grid (50x25) for optimization, fine grid for final evaluation
- Collected solutions during DE run for candidate diversity
- Applied NM polish to top solutions

## Conclusion
DE cannot match CMA-ES for this inverse heat problem. The key advantage of CMA-ES - learning parameter correlations through covariance adaptation - is essential for accurate convergence in the 2-4D search space. DE is 10% faster but sacrifices accuracy.

## Recommendation
- Mark `differential_evolution` family as **FAILED**
- CMA-ES remains the best evolutionary optimization approach for this problem
- Focus on improving CMA-ES efficiency rather than alternative algorithms

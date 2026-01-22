# Variable Projection Experiment Summary

## Experiment ID: EXP_SEPARABLE_VP_001
**Status:** FAILED
**Worker:** W1
**Family:** variable_projection

## Hypothesis
Variable Projection (VP) exploits the separable structure of our problem:
- Nonlinear parameters: positions (x, y)
- Linear parameters: intensities (q)

VP with Gauss-Newton (via scipy.least_squares) may converge faster than CMA-ES by analytically eliminating the linear parameters and using gradient information for the nonlinear ones.

## Results

### Tuning Runs

| Config | Samples | Score | Time (min) | Notes |
|--------|---------|-------|------------|-------|
| 5 starts, 30 nfev, 8 polish | 20 | 1.0025 | 54.3 | Stuck in local minima |
| 3 starts, 15 nfev, 5 polish | 20 | 0.9888 | 32.9 | Faster but worse |

### Best In-Budget Configuration
- **Config:** n_multi_starts=5, max_nfev=30, refine_maxiter=8
- **Score:** 1.0025
- **Projected Time:** 54.3 min
- **vs Baseline:** -0.1337 score, +15.3 min (1.1362 @ 39 min)

## Why VP Failed

### 1. Local Optimizer in Multi-Modal Landscape
Trust-Region Reflective (used by scipy.least_squares) is a local optimizer. The RMSE landscape for inverse heat problems has multiple local minima:
- Samples 10 and 13 consistently get stuck with RMSE > 0.4
- CMA-ES escapes these by maintaining population diversity

### 2. Baseline Already Uses VP
The key insight: **the baseline already uses Variable Projection implicitly**:
```python
# Baseline approach (already VP)
for each CMA-ES candidate (x, y):
    q_optimal = solve_least_squares(A(x,y) * q = b)  # VP step
    rmse = ||A(x,y) * q_optimal - b||
```

The difference is:
- **Baseline:** CMA-ES explores globally, VP provides optimal q for each position
- **This experiment:** VP + Gauss-Newton tries to do everything locally

### 3. No Improvement Possible
Since the baseline already uses VP for q-optimization, the only potential improvement would be if Gauss-Newton on positions was better than CMA-ES. It's not, because:
- CMA-ES: Global optimizer, escapes local minima, maintains population diversity
- Gauss-Newton: Local optimizer, fast convergence but stuck in local minima

## Conclusion
VP with Gauss-Newton cannot improve on CMA-ES for this problem. The baseline already exploits the separable structure optimally - CMA-ES handles the multi-modal position optimization while least squares handles the linear intensity optimization.

## Recommendation
- Mark `variable_projection` family as **FAILED**
- The baseline's CMA-ES + analytical q combination is already optimal for exploiting separability
- Do not pursue other VP variants (VarPro, etc.) - same local minima issue

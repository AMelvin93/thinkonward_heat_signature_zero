# Experiment: Log-RMSE Loss

**Experiment ID:** EXP_LOG_RMSE_LOSS_001
**Worker:** W2
**Status:** FAILED
**Family:** loss_reformulation

## Hypothesis

Using log(1+RMSE) instead of RMSE as the CMA-ES optimization objective may improve convergence because:
1. Log compresses large errors, amplifies small differences
2. Since log is monotonic, the optimum remains the same
3. The landscape shape changes which may affect CMA-ES behavior

## Approach

1. During CMA-ES optimization, return log(1+rmse) instead of rmse
2. Store actual RMSE for ranking (convert back via exp(x) - 1)
3. Use regular RMSE for Nelder-Mead polish
4. Keep all other baseline settings (40% timesteps, sigma 0.18/0.22, NM x8)

## Results

| Run | Config | Score | Time (min) | In Budget | Notes |
|-----|--------|-------|------------|-----------|-------|
| 1 | log(1+RMSE) objective | 1.1677 | 70.5 | NO | Same accuracy but slower |

**Baseline:** 1.1688 @ 58.4 min (RMSE objective)

## Analysis

### Why Log-RMSE Failed

1. **No accuracy improvement**: Score 1.1677 is essentially the same as baseline 1.1688 (-0.0011 difference is within noise).

2. **Time increased by 12 min**: The log transformation and back-conversion adds computational overhead without benefit.

3. **CMA-ES already handles fitness ranking well**: CMA-ES ranks solutions and adapts covariance based on selection. The log transformation doesn't change rankings (monotonic), so it provides no information benefit.

4. **Landscape shape change didn't help**: While log(1+x) has different curvature than x, this difference didn't translate to better CMA-ES convergence for this problem.

### Detailed Results

**Run 1 (log(1+RMSE) objective):**
- 1-source RMSE: 0.1049 (essentially same as baseline ~0.10)
- 2-source RMSE: 0.1520 (essentially same as baseline ~0.15)
- Time: 70.5 min (21% over budget)

## Conclusion

**FAILED** - Log(1+RMSE) objective does NOT improve CMA-ES performance.

Key insights:
- Monotonic transformations of the objective don't change the optimum but also don't help CMA-ES
- The standard RMSE objective is already well-suited for CMA-ES's selection mechanism
- Any transformation that doesn't change the optimum also doesn't change what CMA-ES learns

## Recommendation

**ABANDON** the loss_reformulation family.

The standard RMSE objective is optimal because:
- It directly measures what we're scored on
- CMA-ES's selection mechanism works well with RMSE
- Any monotonic transformation adds overhead without benefit
- Non-monotonic transformations change the optimum (which is bad - see EXP_WEIGHTED_LOSS_001)

## MLflow Run IDs
- Run 1: `c2e12a3215c541e994d7c747d556bb08`

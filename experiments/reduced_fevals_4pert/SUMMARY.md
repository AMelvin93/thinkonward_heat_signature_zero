# Experiment: reduced_fevals_4pert

## Status: FAILED - Still over budget despite reduced fevals

## Hypothesis
Reduce fevals from 20/44 to 18/40 to save time and fit 4 perturbations + tabu_distance=0.04 within 60-minute budget.

## Results (3 Runs)

| Run | Score | Time (min) | RMSE 1src | RMSE 2src | Budget Status |
|-----|-------|------------|-----------|-----------|---------------|
| 1   | 1.1480 | 65.4 | 0.1146 | 0.1912 | OVER BUDGET |
| 2   | 1.1541 | 62.6 | 0.1212 | 0.1790 | OVER BUDGET |
| 3   | 1.1470 | 67.2 | 0.1180 | 0.1901 | OVER BUDGET |

## Statistics

| Metric | Value |
|--------|-------|
| Mean Score | 1.1497 +/- 0.0031 |
| Mean Time | **65.0 min** |
| vs Baseline (2 pert) | +0.0001 |
| vs 4 pert (20/44 fevals) | -0.0038 |
| Runs in budget | **0/3** |

## Analysis

### Time Breakdown
- Original 4 pert + 20/44 fevals: 61.3 min
- Reduced 4 pert + 18/40 fevals: 65.0 min (WORSE!)

The reduced fevals actually increased runtime. This is counterintuitive but may be explained by:
1. Run-to-run variance in timing
2. CMA-ES convergence behavior (fewer fevals may require more restarts)

### Score Comparison
The reduced fevals config has similar score (1.1497) to baseline (1.1496), meaning:
- The score improvement from 4 perturbations is offset by the accuracy loss from fewer fevals
- Net result is no improvement

## Conclusion

**4 perturbations CANNOT fit within budget**, even with reduced fevals.

The production configuration should remain:
- n_perturbations: 2
- max_fevals_1src: 20
- max_fevals_2src: 44
- tabu_distance: 0.04

**Expected Score**: 1.1496 @ 55.4 min

## Final Recommendation

Do not pursue 4-perturbation configurations. The 60-minute budget constraint is fundamental and cannot be circumvented by reducing other parameters without losing the benefit.

---
**Worker**: W1
**Completed**: 2026-02-02
**Runs**: 3
**Result**: FAILED - over budget, no improvement

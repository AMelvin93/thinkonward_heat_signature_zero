# Best of 2 Seeds Experiment

## Status: FAILED

## Hypothesis
CMA-ES can get stuck in suboptimal basins. Running 2 independent instances with different seeds may find better global optimum.

## Configuration
- Run 2 CMA-ES instances with different seeds
- Each instance gets half the budget (10/18 fevals for 1-src/2-src)
- Select best candidate before NM polish

## Results

| Run | Config | Score | RMSE | Projected Time | Status |
|-----|--------|-------|------|----------------|--------|
| 1 | 2 seeds, 20/36 total | 1.1162 | 0.1543 | 133.9 min | FAILED |

## Analysis

### Why It Failed
1. **Budget splitting hurts convergence**: Each CMA-ES instance only gets 10 fevals (1-src) or 18 fevals (2-src). CMA-ES needs generations to learn the covariance matrix. With half the budget, it can't adapt properly.

2. **Catastrophic failures**: Sample 7 had RMSE=0.5374 (baseline gets ~0.15-0.28). This shows the dual-seed approach completely fails on some samples.

3. **Overhead without benefit**: The approach is 2.2x over budget (133.9 min vs 60 min) while also being -0.0306 worse in score.

### Comparison to Baseline
- Baseline (single seed, full budget): 1.1468 @ 54.2 min
- This experiment (2 seeds, split budget): 1.1162 @ 133.9 min
- Delta: -0.0306 score, +79.7 min time

## Key Finding
**CMA-ES covariance learning is critical.** The algorithm needs sufficient evaluations per instance to adapt its covariance matrix. Splitting budget between seeds degrades performance rather than improving it.

This confirms prior evidence from:
- more_inits_select_best: FAILED
- multistart_elite_selection: FAILED

## Conclusion
Multi-start CMA-ES does NOT help. The single-seed, full-budget approach is optimal. `multi_start_v2` family EXHAUSTED.

## Recommendation
Do not pursue multi-seed or multi-start CMA-ES strategies. Focus on improving the single-run optimization quality.

# Higher CMA-ES Fevals Experiment

## Status: FAILED - Higher fevals performs WORSE than baseline

## Hypothesis
Higher CMA-ES fevals (25/45 vs 20/36) could improve accuracy using the ~2.4 min time buffer.

## Results Summary

| Run | Config | Score | Time (min) | Budget Status |
|-----|--------|-------|------------|---------------|
| 1 | fevals_25_45 | 1.1545 | 58.9 | IN budget |
| 2 | fevals_22_40 | 1.1518 | 59.2 | IN budget |
| 3 | **fevals_20_36_baseline** | **1.1621** | **56.2** | **IN budget (BEST)** |

## Key Findings

### 1. Higher Fevals Hurts Performance!
Surprisingly, increasing CMA-ES evaluations DECREASED accuracy:
- 25/45 fevals: 1.1545 (-0.0076 vs baseline)
- 22/40 fevals: 1.1518 (-0.0103 vs baseline)
- 20/36 fevals: 1.1621 (BEST)

### 2. Baseline is Optimal
The original fevals setting (20/36) is already well-tuned:
- More exploration doesn't help
- Extra time is wasted on evaluations that don't improve convergence
- The coordinate refinement step benefits more from remaining time

### 3. Run-to-Run Variance
The baseline achieved 1.1621 @ 56.2 min in this run, vs 1.1602 @ 57.6 min previously.
This suggests ~0.002 variance in score between runs.

## Analysis

### Why Higher Fevals May Hurt
1. **Diminishing returns**: CMA-ES converges quickly; more evaluations add noise
2. **Time trade-off**: More fevals = less time for NM polish and coord refinement
3. **Exploration vs exploitation**: At 20/36 fevals, sufficient exploration; more is wasteful

### RMSE Breakdown
| Config | RMSE 1-src | RMSE 2-src |
|--------|------------|------------|
| fevals_25_45 | 0.132921 | 0.207687 |
| fevals_22_40 | 0.131393 | 0.216505 |
| **fevals_20_36** | **0.122297** | **0.197739** |

The baseline config has better RMSE on both 1-source and 2-source problems.

## Conclusion

**The current fevals setting (20/36) is already optimal.**

Do NOT increase CMA-ES evaluations - it wastes time and hurts accuracy.

## Recommendations

1. **Keep fevals at 20/36** - don't increase
2. **Mark higher_fevals_v2 family as EXHAUSTED**
3. **Focus tuning efforts elsewhere** (sigma, other hyperparameters)
4. **Production config remains**: 6 NM + coord refine + 20/36 fevals

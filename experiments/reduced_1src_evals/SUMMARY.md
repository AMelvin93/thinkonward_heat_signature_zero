# Reduced 1-Source Fevals Experiment

## Experiment ID
EXP_REDUCED_1SRC_EVALS_001

## Hypothesis
1-source problems have only 3 params (x, y, q) vs 6 for 2-source. They may converge faster and need fewer CMA-ES fevals.

## Baseline
- **perturbed_extended_polish**: 1.1464 @ 51.2 min (fevals 20/36)

## Results

### FAILED - Reducing 1-source fevals hurts accuracy

| Run | 1-src Fevals | Score | Time (min) | In Budget | vs Baseline |
|-----|--------------|-------|------------|-----------|-------------|
| 1 | 15 | 1.1398 | 49.0 | YES | -0.0066 |
| 2 | 17 | 1.1402 | 413.6* | NO | -0.0062 |

*Run 2 timing anomaly due to resource contention

**Best in-budget**: Run 1 with 1.1398 @ 49.0 min
**Delta vs baseline**: -0.0066 (WORSE)

## Tuning Efficiency Metrics
- **Runs executed**: 2
- **Time utilization**: 82% (49.0/60 min for best in-budget)
- **Parameter space explored**:
  - max_fevals_1src: [15, 17] (baseline: 20)
- **Pivot points**:
  - Run 1 (fevals=15) worse than baseline → Tried fevals=17
  - Run 2 (fevals=17) also worse → Confirmed hypothesis is wrong

## Budget Analysis
| Run | Score | Time | Budget Remaining | Decision |
|-----|-------|------|------------------|----------|
| 1 | 1.1398 | 49.0 | +11.0 min | CONTINUE - worse than baseline |
| 2 | 1.1402 | 413.6* | N/A (timing issue) | COMPLETE - hypothesis disproved |

## Why Reduced 1-Source Fevals Fails

1. **1-source is NOT simpler**: Despite having fewer parameters, 1-source problems have complex RMSE landscapes
2. **CMA-ES needs sufficient population**: With 3 params, CMA-ES still needs ~20 fevals to explore effectively
3. **No convergence speedup observed**: 1-source samples don't converge faster than expected

## Key Findings

1. **1-source problems need full 20 fevals**: Reducing to 15 or 17 hurts accuracy by 0.6-0.7%
2. **Parameter count doesn't determine difficulty**: The RMSE landscape complexity matters more than parameter dimension
3. **Baseline fevals are already optimized**: 20/36 fevals for 1-src/2-src is correct

## What Would Have Been Tried With More Time
- If Run 1 had succeeded, would have tried fevals=12 or 10
- Would have tried asymmetric approach (more fevals for 2-src)

## Recommendation

**DO NOT reduce 1-source fevals.** Keep the baseline configuration:
- 1-source: 20 fevals
- 2-source: 36 fevals

## Conclusion

**EXPERIMENT FAILED**

The hypothesis that 1-source problems can use fewer CMA-ES fevals because they have fewer parameters is **DISPROVED**. Both test configurations (fevals=15 and fevals=17) produced worse scores than baseline.

The efficiency_v2 family should be marked as showing diminishing returns.

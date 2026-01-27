# Experiment Summary: cmaes_restart_inject_best

## Status: FAILED

## Experiment ID: EXP_CMAES_WITH_RESTART_BEST_001
## Worker: W1
## Date: 2026-01-26

## Hypothesis
When CMA-ES triggers a fallback restart (due to poor primary results), injecting the best solution found so far into the new population may help by:
- Seeding the new search with a known good starting point
- Preventing regression from primary search
- Combining exploration (new inits) with exploitation (best known)

## Approach
Modified baseline optimizer to use `cma.inject()` when starting fallback CMA-ES:
1. Track best position parameters from primary optimization
2. When fallback triggers, inject best solution into new CMA-ES population
3. Continue optimization with this seed

## Results

| Metric | Value | Baseline | Delta |
|--------|-------|----------|-------|
| **Score** | 1.1285 | 1.1688 | **-3.4%** |
| Projected 400 min | 41.7 | 58.4 | -28.6% |
| RMSE 1-source | 0.1366 | - | - |
| RMSE 2-source | 0.2239 | - | - |
| **Injection used** | 0/80 | - | - |

## Key Finding: Injection Never Triggered

The injection feature was **never used** across all 80 samples because:
- Fallback only triggers when `best_rmse > threshold`
- Threshold is 0.4 for 1-source, 0.5 for 2-source
- Primary optimization succeeded (RMSE below threshold) for ALL samples
- Therefore, no fallback restart occurred, and injection was never invoked

## Why Score is Lower Than Baseline

Even though injection was never used, the score (1.1285) is lower than baseline (1.1688). This suggests:
1. Implementation differences in the optimizer copy
2. Potential subtle bugs in the modified optimization flow
3. Random seed effects

The experiment did not test its actual hypothesis because the fallback path was never exercised.

## Conclusion
**The injection approach could not be evaluated** because:
1. The fallback mechanism is rarely triggered with current thresholds
2. Most samples converge well on primary optimization
3. The injection would only help samples that fail primary optimization AND would benefit from warm-starting

## Recommendation
**Do NOT pursue solution injection on fallback** because:
1. Fallback is rarely triggered
2. When it is triggered, the problem is usually harder, and injection may not help
3. The overhead of tracking and injecting solutions is not justified

If fallback happens more often (with tighter thresholds), the injection approach could be re-evaluated. However, tightening thresholds would likely not improve overall score.

## Files
- `optimizer.py`: Implementation with injection on fallback
- `run.py`: Run script
- `STATE.json`: Experiment state

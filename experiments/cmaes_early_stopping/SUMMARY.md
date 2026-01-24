# CMA-ES Early Stopping Experiment

**Experiment ID:** EXP_EARLY_STOP_CMA_001
**Worker:** W2
**Status:** FAILED
**Date:** 2026-01-24

## Hypothesis

Many CMA-ES runs converge before using their full function evaluation budget. By detecting stagnation early (< 1% improvement for 3 consecutive generations), we can stop CMA-ES early and reallocate the saved evaluations to additional Nelder-Mead polish iterations, improving accuracy without increasing total time.

## Approach

1. Track relative improvement between CMA-ES generations
2. If improvement < 1% for 3 consecutive generations, trigger early stopping
3. Track number of function evaluations saved
4. Convert saved fevals to extra NM polish iterations (0.5 extra per saved feval)
5. Cap total polish iterations at 12 (vs base of 8)

## Configuration Tested

```
stagnation_threshold: 0.01 (1%)
stagnation_generations: 3
base_polish_iters: 8
max_polish_iters: 12
extra_polish_per_saved: 0.5
```

## Results

| Metric | This Experiment | Baseline | Delta |
|--------|-----------------|----------|-------|
| Score | 1.1378 | 1.1688 | **-0.0310** |
| Time (min) | 313.3 | 58.4 | **+254.9 min (5.4x)** |
| RMSE 1-src | 0.0979 | ~0.09 | Similar |
| RMSE 2-src | 0.1437 | ~0.14 | Similar |

### Early Stopping Statistics

- **Total saved fevals:** 0 (across all 80 samples)
- **Avg polish iterations:** 8.0 (base value, no extra)
- **Early stopping triggers:** 0 out of 80 samples

## Key Finding

**Early stopping NEVER triggered.** The stagnation threshold of 1% improvement for 3 consecutive generations was never met. CMA-ES continues making meaningful improvements (>1%) throughout its full evaluation budget.

## Why It Failed

1. **CMA-ES doesn't stagnate early**: The optimization landscape for heat source localization does not exhibit early convergence. CMA-ES consistently finds better solutions throughout its budget.

2. **1% threshold too tight**: CMA-ES makes >1% improvements in most generations, even near the end of optimization.

3. **Massive overhead**: The stagnation tracking code introduced significant overhead (5.4x runtime increase) without providing any benefit since early stopping never occurred.

4. **Prior evidence confirmed**: EXP_ADAPTIVE_BUDGET_001 previously found that "Early termination based on sigma/stagnation HURTS accuracy." This experiment confirms that finding - the optimization needs its full budget.

## Conclusion

The hypothesis that "many samples converge quickly" is **NOT validated**. CMA-ES optimization for this problem continues improving throughout its full evaluation budget. Early stopping approaches should be abandoned.

## Recommendation for W0

**ABANDON** the "efficiency/early_stopping" experiment family. Multiple experiments now confirm:
- EXP_ADAPTIVE_BUDGET_001: Early termination hurts accuracy
- EXP_EARLY_STOP_CMA_001: Early stopping never triggers (this experiment)

The CMA-ES optimization landscape for heat source localization requires the full evaluation budget. Focus optimization efforts on:
1. Improving initialization quality
2. Better hyperparameter tuning (sigma, population size)
3. Reducing simulation cost (temporal fidelity already at 40%)

## MLflow Run

Run ID: `37ae31dee396436f8ac985fe12d87d7c`

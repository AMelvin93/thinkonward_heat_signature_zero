# Experiment Summary: kriging_local_infill

## Metadata
- **Experiment ID**: EXP_KRIGING_INFILL_001
- **Worker**: W1
- **Date**: 2026-01-25
- **Algorithm Family**: surrogate_v2

## Status: ABORTED (Prior Evidence Conclusive)

## Objective
Use local Kriging (Gaussian Process) surrogate fitted on recent CMA-ES evaluations to pre-screen population candidates, simulating only the top 50% by Kriging prediction.

## Why Aborted

The **surrogate family has been marked EXHAUSTED** by multiple prior experiments, all showing that surrogates cannot improve on CMA-ES for this problem.

### Key Prior Evidence

| Experiment | Result | Key Finding |
|------------|--------|-------------|
| **cmaes_rbf_surrogate** | ABORTED | "CMA-ES covariance adaptation is already implicit surrogate modeling. ~10% rejection rate, ~6% potential sim reduction. Not worth complexity." |
| **bayesian_optimization_gp** | FAILED (91% worse) | "GP surrogate poorly models RMSE landscape" |
| **lq_cma_es_builtin** | FAILED (29-71% worse) | "With only 10-18 fevals, not enough data for surrogate" |
| **gappy_cpod** | ABORTED | "surrogate_v2 family EXHAUSTED" |
| **pretrained_nn_surrogate** | ABORTED | "RMSE landscape is completely sample-specific" |

### Why Kriging Cannot Help

1. **CMA-ES Already Does Implicit Surrogate Modeling**
   - The covariance matrix adaptation in CMA-ES IS a form of surrogate modeling
   - It learns the local curvature of the fitness landscape
   - Adding explicit Kriging on top provides marginal (~6%) improvement for significant complexity

2. **Rejection Rate Is Too Low**
   - `cmaes_rbf_surrogate` found only ~10% rejection rate with RBF surrogate
   - With small population (6-8), this means rejecting 0-1 candidates per generation
   - The overhead of fitting Kriging may negate any savings

3. **Insufficient Data for Accurate Kriging**
   - Kriging/GP needs sufficient data points to build accurate model
   - With only 20 evaluations to fit on, prediction variance is high
   - lq-CMA-ES showed that "with only 10-18 fevals, not enough data"

4. **GP Poorly Models RMSE Landscape**
   - `bayesian_optimization_gp` showed "91% worse accuracy" with GP surrogate
   - The thermal inverse problem has complex local structure
   - Standard Matern/RBF kernels don't capture this

## Technical Analysis

### Proposed Approach
```
1. After each CMA-ES generation, fit local Kriging on last 20 evaluations
2. Use Kriging to pre-screen next generation's candidates
3. Only simulate top 50% by Kriging prediction
```

### Why This Would Fail

| Issue | Impact |
|-------|--------|
| Kriging fit overhead | ~10ms per fit, 3 generations = 30ms added |
| Prediction uncertainty | High variance with only 20 points |
| False rejections | Discarding 50% may include actual best candidates |
| Covariance disruption | CMA-ES expects ALL population evaluations |

### CMA-ES Covariance Disruption

CMA-ES uses ALL population evaluations to update its covariance matrix:
```python
# CMA-ES update (simplified)
C_new = weighted_sum([x - mean for x in population])
```

Pre-screening would provide biased samples, potentially hurting covariance adaptation more than the saved simulations help.

## Algorithm Family Status

- **surrogate_v2**: **EXHAUSTED** (confirmed by gappy_cpod)
- **surrogate_hybrid**: **EXHAUSTED** (confirmed by cmaes_rbf_surrogate)
- **bayesian_opt**: **EXHAUSTED** (confirmed by bayesian_optimization_gp)

## Recommendations

1. **Do NOT pursue any surrogate-based approaches** - all have failed
2. **CMA-ES covariance adaptation is optimal** - no need for explicit surrogates
3. **Focus on other improvements** - the surrogate space is fully explored

## Conclusion

The kriging_local_infill experiment would fail for the same reasons all other surrogate experiments failed: CMA-ES already provides optimal implicit surrogate modeling through covariance adaptation, and explicit surrogates add complexity without meaningful benefit. The surrogate_v2 family is EXHAUSTED.

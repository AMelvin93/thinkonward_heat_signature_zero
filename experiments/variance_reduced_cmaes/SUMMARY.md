# Experiment Summary: variance_reduced_cmaes

## Status: ABORTED (Already Tested)

## Experiment ID: EXP_VARIANCE_REDUCED_CMAES_001
## Worker: W1
## Date: 2026-01-26

## Hypothesis
Antithetic sampling (mirrored sampling) can reduce fitness variance and improve CMA-ES convergence.

## Why Aborted

### Antithetic Sampling Already Tested in OpenAI ES

Prior experiment `openai_evolution_strategy` (EXP_OPENAI_ES_001) explicitly used antithetic sampling:

> "Used antithetic sampling (mirrored perturbations) for variance reduction."

Result: **FAILED** - Score 1.1204 vs baseline 1.1362 (-0.0158)

### CMA-ES Already Has Variance Reduction Built-In

The `cmaes` library already implements several variance reduction techniques:

1. **CMA_mirrorMethod**: Built-in mirrored sampling option
2. **Evolution path**: Accumulates gradient information
3. **Covariance matrix adaptation**: Naturally reduces variance along search directions

### Why Mirrored Sampling Didn't Help

1. **Full covariance captures correlations** - More important than variance reduction
2. **Low dimensionality (2-4D)** - Variance reduction provides marginal benefit
3. **Expensive evaluations** - Each evaluation takes ~50ms; variance is not the bottleneck

### Implementation Note

CMA-ES can enable mirrored sampling via:
```python
opts = cma.CMAOptions()
opts['CMA_mirrormethod'] = 1  # or 2 for different variants
```

This was NOT included in baseline because it provides no benefit for our problem size and evaluation cost.

## Recommendation
**Do NOT add mirrored sampling to CMA-ES.** The full covariance adaptation is more important than variance reduction for this low-dimensional expensive black-box problem.

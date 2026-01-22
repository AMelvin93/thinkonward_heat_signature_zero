# Experiment Summary: active_cmaes_covariance

## Metadata
- **Experiment ID**: EXP_ACTIVE_CMAES_001
- **Worker**: W2
- **Date**: 2026-01-19
- **Algorithm Family**: cmaes_variants

## Objective
Test if enabling Active CMA-ES variant improves convergence.

## Result
**ABORTED - Wrong premise.**

## Key Finding

The `CMA_active` option in pycma **defaults to True**:

```python
import cma
print(cma.CMAOptions()['CMA_active'])
# Output: 'True  # negative update, conducted after the original update'
```

**The baseline already uses Active CMA-ES.** There's nothing to enable.

## Verification

Active CMA-ES performs a "negative update" using information from unsuccessful candidates to improve covariance learning. This is already included in the default pycma configuration.

Since the baseline optimizer doesn't explicitly set `CMA_active=False`, it inherits the default True value.

## Recommendation

Remove this experiment from the queue - it tests something already in place.

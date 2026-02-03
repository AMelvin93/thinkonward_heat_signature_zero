# Experiment: Sequential 2-Source Estimation

## Status: FAILED - Fundamentally Flawed Approach

## Hypothesis
Using heat equation linearity, decompose 4D 2-source optimization into 2D + 2D:
1. Find dominant source first (2D)
2. Subtract its contribution
3. Find second source in residual (2D)

Expected benefit: Better convergence for 2-source problems.

## Baseline
- **4pert_nm2**: 1.1482 @ 51.7 min (2-source RMSE ~0.19)

## Results (3 Tuning Runs)

| Run | Fevals | Score | Time (min) | RMSE 2src | Budget |
|-----|--------|-------|------------|-----------|--------|
| 1   | 15/22  | 1.0214 | 74.2 | 0.3071 | OVER |
| 2   | 20/44  | 1.0401 | 109.0 | 0.2296 | MASSIVELY OVER |
| 3   | 10/15  | **1.0157** | **42.5** | 0.3078 | **IN** |

## Key Finding: APPROACH IS FUNDAMENTALLY FLAWED

The sequential 2-source approach **cannot achieve competitive accuracy** at any feval level:

### Why It Fails

1. **Compounding Errors**: Subtracting the first source's contribution introduces errors that propagate to the second source estimation.

2. **Inherently Inefficient**: The sequential approach requires ~102 simulations per 2-source sample (2 optimization phases + joint diversity) vs ~44 for joint CMA-ES.

3. **No Accuracy Benefit**: Even with 2x the simulations, 2-source RMSE is significantly worse:
   - Sequential: 0.23-0.31
   - Baseline: ~0.19

4. **Cannot Fit Budget with Good Accuracy**:
   - 42.5 min config (in budget) → score 1.0157 (-0.13 from baseline)
   - 74.2 min config (over budget) → score 1.0214 (-0.12 from baseline)

### Comparison

| Metric | Sequential Best | Baseline |
|--------|-----------------|----------|
| Score | 1.0401 | **1.1482** |
| Time | 109 min | **51.7 min** |
| 2-src RMSE | 0.2296 | **~0.19** |

The baseline is **0.11 points better** while being **57 min faster**.

## Why Joint 4D Optimization is Better

1. **CMA-ES covariance** captures correlations between source positions
2. **Global view**: Both sources are optimized together, avoiding local minima from sequential decomposition
3. **Efficient**: Single CMA-ES run vs multiple phases

## Tuning Efficiency

- **Runs executed**: 3
- **Time utilization**: N/A (all configs either over budget or terrible accuracy)
- **Parameter space explored**: fevals sweep (10/15, 15/22, 20/44)
- **Conclusion after 3 runs**: Approach cannot work

## Decision Tree Analysis

Per decision tree:
- Run 1: Over budget → PIVOT to reduce params
- Run 2: Massively over budget → Confirmed scaling issue
- Run 3: In budget but score 0.13 worse → FUNDAMENTAL FLAW

## Recommendation

**DO NOT PURSUE sequential 2-source optimization.**

The physics insight (heat equation linearity) is correct, but the optimization approach is flawed:
- Sequential optimization introduces errors that compound
- Joint 4D CMA-ES with covariance adaptation is superior

### Algorithm Family Update

Add to exhausted families:
```
"sequential_decomposition": "FAILED - Sequential 2D+2D optimization introduces compounding errors. Joint 4D CMA-ES is optimal."
```

---
**Worker**: W2
**Completed**: 2026-02-02
**Runs**: 3
**Result**: FAILED - Approach fundamentally cannot achieve competitive accuracy

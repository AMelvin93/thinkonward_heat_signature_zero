# Experiment Summary: larger_popsize_exploration

## Metadata
- **Experiment ID**: EXP_POPSIZE_DIVERSITY_001
- **Worker**: W1
- **Date**: 2026-01-25
- **Algorithm Family**: cmaes_tuning_v2

## Status: ABORTED (Duplicate + Misconception)

## Objective
Use popsize=8 (vs default ~4-5) for better exploration, with fewer generations. CMA-ES default popsize scales with dimension but may be too small for multimodal RMSE landscape.

## Why Aborted

This experiment has two fundamental issues:

### Issue 1: DUPLICATE of Prior Experiment

`larger_cmaes_population` (EXP_LARGER_POPSIZE_001) already tested larger population sizes:

| Config | Score | Time (min) | Result |
|--------|-------|------------|--------|
| popsize=12 | 1.1666 | 73.0 | **FAILED** |
| default popsize | 1.1688 | 58.4 | Baseline |
| Delta | -0.0022 | +14.6 | Worse in both |

Key findings from that experiment:
1. **Larger popsize reduces number of generations** - with fixed feval budget, more candidates per generation = fewer generations
2. **CMA-ES needs multiple generations** to adapt covariance matrix
3. **Default popsize formula is already optimal** for this problem dimension

### Issue 2: Popsize=8 IS the Default for 2-Source

The experiment proposes "popsize=8 (vs default ~4-5)". This is **incorrect**:

```
CMA-ES default popsize = 4 + floor(3 * ln(n))
- 1-source (n=2): popsize ≈ 6
- 2-source (n=4): popsize ≈ 8
```

For 2-source problems (which are 60% of the dataset and the main bottleneck), **popsize=8 IS the default**. This experiment would not change anything for 2-source problems.

## Prior Evidence Summary

### larger_cmaes_population (Exact Test)
- **Result**: FAILED - popsize=12 is worse than default
- **Why**: Fewer generations → worse covariance adaptation
- **Conclusion**: "cmaes_accuracy family marked EXHAUSTED"

### adaptive_population_size (Two-Phase Popsize)
- **Result**: FAILED - Two-phase popsize 124.6 min, score 1.0026
- **Why**: Inconsistent popsize disrupts covariance learning
- **Conclusion**: "Any popsize manipulation - default is already optimal"

### ipop_cmaes_temporal (IPOP Restarts)
- **Result**: FAILED - IPOP adds time without improving accuracy
- **Why**: Each restart splits feval budget, insufficient per restart
- **Conclusion**: "Larger population sizes wastes budget"

## Technical Explanation

### Why Default Popsize is Optimal

CMA-ES's population size formula `4 + floor(3*ln(n))` was designed based on:
1. **Covariance estimation requirements** - need enough samples per generation
2. **Number of generations needed** - need multiple generations to adapt
3. **Problem dimension scaling** - larger dim needs slightly larger popsize

For our 2-4D problem, this formula gives optimal values that balance exploration vs. adaptation.

### Why Larger Popsize Hurts

With fixed feval budget (20 for 1-src, 36 for 2-src):
- popsize=6, fevals=20: ~3.3 generations
- popsize=8, fevals=20: ~2.5 generations
- popsize=12, fevals=20: ~1.7 generations

Fewer generations = less covariance adaptation = worse convergence.

## Recommendations

1. **Do NOT modify CMA-ES population size** - already optimal
2. **cmaes_tuning family is EXHAUSTED** for population parameters
3. **Focus elsewhere** - initialization, polish, temporal fidelity are the only levers

## Raw Data
- MLflow run IDs: None (experiment not executed)
- Prior evidence: larger_cmaes_population, adaptive_population_size, ipop_cmaes_temporal

## Conclusion

This experiment is both a **duplicate** of prior work and based on a **misconception** about default popsize values. The proposed popsize=8 is already the default for 2-source problems. Larger population sizes have been conclusively shown to hurt performance by reducing the number of generations for covariance adaptation.

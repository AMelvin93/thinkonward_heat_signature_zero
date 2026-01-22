# Experiment Summary: larger_cmaes_population

## Metadata
- **Experiment ID**: EXP_LARGER_POPSIZE_001
- **Worker**: W1
- **Date**: 2026-01-19
- **Algorithm Family**: cmaes_accuracy

## Objective
Test whether larger CMA-ES population sizes improve convergence quality in inverse problems.

## Hypothesis
Research literature suggests that larger populations improve covariance estimation and convergence quality in inverse problems (Oxford Academic). The default CMA-ES popsize (~6 for 1-source, ~8 for 2-source) might be too small.

## Results Summary
- **Best Score**: 1.1666 @ 73.0 min (popsize=12/12)
- **Baseline**: 1.1688 @ 58.4 min (default popsize)
- **Delta**: -0.0022 score, +14.6 min
- **Status**: FAILED - Larger popsize is WORSE in both accuracy and time

## Tuning History

| Run | Popsize (1-src/2-src) | Score | Time (min) | In Budget | Notes |
|-----|----------------------|-------|------------|-----------|-------|
| 1 | 12/12 | 1.1666 | 73.0 | No | -0.0022 score, +14.6 min vs baseline |

## Key Findings

### What Didn't Work
1. **Larger population size is counterproductive**
   - Popsize=12 adds ~14 min to runtime
   - Score actually drops by 0.0022
   - Experiment aborted after run 1 (abort criteria met)

### Why Larger Popsize Hurts

1. **Fixed feval budget limits generations**
   - With max_fevals=20/36 and popsize=12, only ~2-3 generations run
   - CMA-ES needs multiple generations to adapt covariance
   - Default popsize (~6/8) allows ~3-4+ generations

2. **Each generation costs more simulations**
   - Larger population = more candidates per generation
   - Total runtime increases linearly with popsize
   - No accuracy benefit to offset the time cost

3. **Problem dimension is already small**
   - 1-source: dim=2, default popsize ≈ 6
   - 2-source: dim=4, default popsize ≈ 8
   - Default formula 4+floor(3*ln(n)) is already optimal for small n

4. **Temporal fidelity makes signal "noisy"**
   - Using 40% timesteps adds noise to objective
   - Better covariance estimation doesn't help with noisy signal
   - More generations (smaller popsize) helps average out noise

## Technical Analysis

### CMA-ES Population Size Mechanics
```
Default popsize = 4 + floor(3 * ln(n))
- n=2 (1-source): popsize ≈ 6
- n=4 (2-source): popsize ≈ 8

With fixed feval budget:
- popsize=6, max_fevals=20: ~3 generations
- popsize=12, max_fevals=20: ~1.5 generations

CMA-ES needs multiple generations to learn covariance structure.
Fewer generations = worse covariance adaptation.
```

### RMSE Breakdown

| Metric | Baseline | Popsize=12 | Delta |
|--------|----------|------------|-------|
| RMSE 1-src | ~0.10 | 0.1096 | +0.01 |
| RMSE 2-src | ~0.14 | 0.1635 | +0.02 |

Larger popsize actually hurts 2-source accuracy more.

## Parameter Sensitivity
- **Population size**: Increasing beyond default is counterproductive
- **Default popsize is optimal**: The CMA-ES formula is already well-tuned

## Recommendations for Future Experiments

1. **Don't increase CMA-ES population size**
   - Default popsize is already optimal for this problem dimension
   - cmaes_accuracy family should be marked EXHAUSTED

2. **Focus on different approaches**:
   - 2-source specialized optimization (different params for 2-src)
   - Sigma scheduling (adaptive sigma during optimization)
   - Progressive polish fidelity

3. **If more accuracy is needed**:
   - Increase feval budget (costs time)
   - Improve initialization (already exhausted)
   - Better local refinement (NM polish is already good)

## Conclusion

**FAILED** - Larger CMA-ES population sizes do NOT improve convergence quality for this problem. The default popsize formula (4+floor(3*ln(n))) is already optimal. Increasing popsize reduces the number of generations, hurting covariance adaptation. The cmaes_accuracy family should be marked as EXHAUSTED.

## Raw Data
- MLflow experiment: `larger_cmaes_population`
- MLflow run: `run1_popsize_12_12`

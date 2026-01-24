# Experiment Summary: separable_cmaes_diagonal

## Metadata
- **Experiment ID**: EXP_SEP_CMAES_001
- **Worker**: W2
- **Date**: 2026-01-24
- **Algorithm Family**: cmaes_variants

## Objective
Test whether Separable CMA-ES (diagonal-only covariance) can provide faster convergence while maintaining accuracy for this 2-4D heat source localization problem.

## Hypothesis
Diagonal-only covariance may be sufficient for low-dimensional problems (2-4D), enabling higher learning rates and faster adaptation. The loss of correlation information between parameters might be acceptable.

## Results Summary
- **Best In-Budget Score**: N/A (failed to stay within budget)
- **Best Overall Score**: 1.1501 @ 312 min
- **Baseline Comparison**: -0.0187 score, +254 min time
- **Status**: FAILED

## Tuning History

| Run | Config Changes | Score | Time (min) | In Budget | Notes |
|-----|---------------|-------|------------|-----------|-------|
| 1 | CMA_diagonal=True, 40% timesteps, 8 NM polish | 1.1501 | 312.0 | NO | Both accuracy and time WORSE |

## Key Findings

### What Didn't Work
- **Diagonal covariance hurts accuracy**: Score dropped from 1.1688 to 1.1501 (-0.0187)
- **Diagonal covariance hurts speed**: Time increased from 58.4 min to 312 min (5.3x slower)
- **Parameter correlations ARE important**: x,y positions have correlation with heat gradient direction
- **No benefit from higher learning rate**: The theoretical faster adaptation didn't materialize

### Critical Insights

**Why Diagonal Covariance Fails for Thermal Source Localization:**

1. **Position correlations matter**: The optimal source position lies along the heat gradient direction from sensors. CMA-ES full covariance learns this correlation (x increases → y should also change proportionally along gradient). Diagonal covariance treats x,y as independent.

2. **More iterations required**: Without correlation information, CMA-ES needs more iterations to find the optimal direction. Each iteration has similar cost, so total cost increases.

3. **Worse convergence quality**: The diagonal search space is axis-aligned, but the RMSE landscape is oriented along heat gradient directions (typically not axis-aligned). This causes poor convergence.

4. **2-source samples hit hardest**: 2-source RMSE increased from ~0.14 (baseline) to 0.1386 (diagonal), and samples with high RMSE (>0.2) appeared:
   - Sample 42: RMSE=0.3976
   - Sample 69: RMSE=0.3629
   - Sample 57: RMSE=0.2690
   - Sample 65: RMSE=0.2360

### Why This Problem Needs Full Covariance

The heat equation creates RMSE landscapes where:
- Source position errors along the heat gradient cost more than perpendicular errors
- CMA-ES covariance matrix learns this anisotropy and optimizes efficiently
- Removing this learning capability forces random-walk-like exploration

## Parameter Sensitivity
- **CMA_diagonal**: Setting to True is definitively WORSE
- No further tuning needed - the approach is fundamentally flawed for this problem

## Recommendations for Future Experiments

1. **DO NOT pursue sep-CMA-ES or other diagonal variants**
2. **DO NOT pursue dd-CMA-ES** (diagonal decoding) - likely same issue
3. **Full covariance CMA-ES is optimal** for this low-dimensional problem
4. **Mark cmaes_variants family (diagonal approaches) as FAILED**

## Recommendations for W0

The diagonal covariance approaches should be marked as EXHAUSTED:
- sep-CMA-ES: FAILED (this experiment)
- dd-CMA-ES: SKIP (same diagonal limitation)
- Coordinate-wise sigma: SKIP (diagonal doesn't help)

The problem structure requires full covariance adaptation to capture parameter correlations along heat gradients.

## Raw Data
- **MLflow run ID**: c6e4f5d256f94e1caea8fe5680176c63
- **Samples completed**: 80/80
- **Best config**: N/A - all configs failed

## Detailed Statistics

### 1-Source Samples (n=32)
- RMSE mean: 0.1018
- Time range: 140-260s per sample

### 2-Source Samples (n=48)
- RMSE mean: 0.1386
- Time range: 300-530s per sample
- Notable failures: samples 42, 69, 57, 65 with RMSE > 0.2

### Comparison to Baseline
| Metric | Baseline (full cov) | Sep-CMA-ES (diagonal) | Delta |
|--------|---------------------|----------------------|-------|
| Score | 1.1688 | 1.1501 | -0.0187 |
| Proj. Time | 58.4 min | 312.0 min | +253.6 min |
| 1-src RMSE | ~0.09 | 0.1018 | +0.01 |
| 2-src RMSE | ~0.14 | 0.1386 | ~same |
| High RMSE cases | 0 | 4 | +4 failures |

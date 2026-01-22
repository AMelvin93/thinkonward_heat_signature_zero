# Weighted Sensor Loss Experiment - FAILED

## Experiment ID
EXP_WEIGHTED_LOSS_001

## Hypothesis
Weight sensors by their informativeness (temporal variance) to improve CMA-ES convergence. Sensors with higher temperature variance are likely closer to heat sources and provide stronger signal.

## Configuration Tested
```
Weighting strategy: variance (weight by temporal variance of sensor readings)
fevals: 20/36 (1-src/2-src)
sigma: 0.18/0.22
timestep_fraction: 0.40
NM polish: 8 iterations on top-3
```

## Results

| Metric | This Run | Baseline (W2) | Delta |
|--------|----------|---------------|-------|
| Score | **1.0131** | 1.1688 | **-0.1557** |
| Time (400 proj) | **90.7 min** | 58.4 min | **+32.3 min** |
| RMSE Mean | 0.1555 | ~0.12 | +0.04 |
| 1-src RMSE | 0.1239 | - | - |
| 2-src RMSE | 0.1766 | - | - |

**STATUS: FAILED** - Both score and timing are significantly worse than baseline.

## Root Cause Analysis

### Why Weighted Loss Failed

1. **Optimization Target Mismatch**: The scoring formula uses unweighted RMSE, but we optimized for weighted RMSE. The optimum of weighted RMSE is different from unweighted RMSE.
   - CMA-ES converges to minimum of weighted loss
   - That minimum is NOT the minimum of unweighted loss
   - Result: Worse unweighted RMSE (what we're scored on)

2. **Diversity Destruction**: ALL 32 one-source samples got only 1 candidate (vs baseline's 2-3):
   - Weighted loss landscape has different shape
   - Multiple initializations converge to same weighted-optimal solution
   - That solution is NOT necessarily diverse in source space

3. **More Fallbacks = More Time**: 6 fallback samples (7.5% of total) vs baseline's ~2%:
   - Weighted loss converges to poor-quality solutions more often
   - Triggers threshold-based fallback
   - Each fallback adds ~200+ extra simulations

4. **Higher 2-Source RMSE**: 0.1766 vs baseline ~0.14:
   - Weighting distorts 2-source optimization more than 1-source
   - Two sources have more complex sensor interactions
   - Weighted loss amplifies these distortions

### Mathematical Insight

The fundamental problem:
```
We optimize: argmin_theta sum_s w_s * (y_s - y_hat_s)^2  [weighted]
But scored on: argmin_theta sum_s (y_s - y_hat_s)^2      [unweighted]

Unless weights are uniform, these have DIFFERENT optima!
```

Weighting makes some sensors more important during optimization. But the scoring formula treats all sensors equally. The "optimal" solution found by weighted optimization is suboptimal for the actual scoring metric.

## Key Learnings

1. **Don't change the loss function**: If you're scored on metric X, optimize metric X directly. Optimizing a proxy metric Y, even if it seems "better", leads to worse X.

2. **Variance weighting biases toward hot sensors**: Sensors near sources have high variance. Weighting by variance over-emphasizes fitting these sensors at the expense of cooler sensors. This can shift the optimal source position.

3. **Diversity depends on loss landscape**: The dissimilarity filter (TAU=0.2) operates in source parameter space, not loss space. If the weighted loss landscape has a single sharp minimum, all CMA-ES runs converge there, destroying diversity.

## Recommendation

**ABANDON** - The loss_function family should be marked FAILED.

Changing the loss function during optimization is fundamentally incompatible with how the scoring works. The only way weighted loss could help is if:
1. We use it only for initial filtering (but early_rejection already failed)
2. The weights somehow preserve the unweighted optimal (unlikely)

Neither scenario is promising. Future experiments should not try alternative loss functions unless the scoring metric itself changes.

## MLflow Run
- Run ID: `3c65557698a542e6b8eae1a2b3ed57c4`
- Experiment: heat-signature-zero

# Experiment Summary: cluster_transfer (EXP_TRANSFER_LEARN_001)

## Metadata
- **Experiment ID**: EXP_TRANSFER_LEARN_001
- **Worker**: W2
- **Date**: 2026-01-18
- **Algorithm Family**: meta_learning

## Objective
Test whether clustering samples by sensor reading features and transferring solutions from cluster representatives can reduce optimization time while maintaining accuracy.

## Hypothesis
Similar sensor patterns have similar optimal heat source configurations. By clustering samples based on sensor features (peak temperatures, onset times, spatial centroids, etc.), we can solve one "representative" sample per cluster fully and use its solution to warm-start optimization for other samples in the same cluster.

## Results Summary
- **Best In-Budget Score**: 1.0804 @ 59.7 min (Run 2)
- **Best Overall Score**: 1.0804 @ 59.7 min
- **Baseline Comparison**: -0.0443 vs 1.1247 baseline
- **Status**: **FAILED**

## Tuning History

| Run | Config Changes | Score | Time (min) | In Budget | Notes |
|-----|---------------|-------|------------|-----------|-------|
| 1 | Initial: transfer_fevals=10, sigma=0.08 | 1.0334 | 61.2 | No | 5 failed samples, 17/70 used transfer |
| 2 | transfer_fevals=15, sigma=0.12 | 1.0804 | 59.7 | Yes | Best run - 3 failed samples, 21/70 used transfer |
| 3 | transfer_fevals=18, sigma=0.15 | 1.0600 | 61.4 | No | MORE budget = WORSE results (4 failures) |

## Key Findings

### What Didn't Work
1. **Clustering by sensor features doesn't predict solution similarity**
   - Samples with similar temperature patterns can have very different heat source positions
   - The relationship between observed temperatures and source locations is highly non-linear
   - Cluster centroids in feature space don't correspond to solution centroids

2. **Transfer initialization often led to WORSE optimization**
   - Run 3 (more budget for transfer) performed WORSE than Run 2
   - This suggests transfer init was directing search to wrong regions
   - Small sigma with transfer init caused getting stuck in wrong local minima
   - Large sigma with transfer init wasted the potential benefit of warm-starting

3. **Several samples consistently failed with exceptions**
   - Samples 42, 78 failed across all runs
   - The transfer initialization caused some numerical issues in 2-source optimization
   - Zero simulations completed for failed samples (exception before optimization started)

4. **Overhead from clustering/representative phases**
   - Solving 10 representatives took significant time (~10 min)
   - This front-loaded cost meant transfer samples needed to be much faster
   - But transfer savings weren't enough to offset representative phase cost

### Why This Approach Fundamentally Failed
1. **Heat equation non-linearity**: Small changes in source position cause large changes in temperature field
2. **Inverse problem degeneracy**: Multiple source configurations can produce similar temperature patterns
3. **Feature-solution mismatch**: Sensor features (peaks, onset times) don't have a simple mapping to source locations
4. **Transfer hurt exploration**: Starting CMA-ES from transferred position reduced exploration diversity

## Parameter Sensitivity
- **transfer_fevals**: Moderate impact - too few (10) fails, but more (18) also hurt
- **transfer_sigma**: Critical - 0.08 too small (stuck), 0.15 too large (wasted init)
- **n_clusters**: Not explored - 10 clusters may be too few or too many
- **Feature selection**: 14 features used may not capture relevant solution similarity

## Recommendations for Future Experiments

### What to AVOID
1. **Do NOT pursue cluster-based transfer learning** - sensor features don't predict solutions
2. **Do NOT use solution transfer for this problem** - the inverse mapping is too non-linear
3. **Do NOT front-load computation** (representative phase) expecting downstream savings

### What W0 Should Consider Instead
1. **WS-CMA-ES with covariance transfer** (not solution transfer)
   - Transfer the learned covariance matrix, not the solution itself
   - Covariance captures search landscape shape which may be more transferable

2. **Solution-space clustering** (post-hoc)
   - Instead of clustering features, cluster the SOLUTIONS after solving
   - Use this for understanding, not optimization

3. **Focus on single-sample optimization improvement**
   - The baseline CMA-ES is already well-tuned
   - Transfer learning adds overhead without benefit
   - Better to focus on surrogates (lq-CMA-ES) or multi-fidelity

4. **Investigate the failing samples**
   - Samples 42, 78 consistently fail across approaches
   - These may have edge cases (sources near boundaries, very low/high kappa)
   - A robust optimizer should handle these gracefully

## Technical Notes on Failures
The samples that failed with `inf` RMSE and 0 simulations had exceptions during optimization. The transfer initialization (4D array for 2-source: [x1, y1, x2, y2]) may have caused issues when:
- Transferred positions were out of bounds after clipping
- Sources were initialized too close together
- Numerical issues in analytical intensity computation

## Raw Data
- MLflow run IDs: Not logged (manual runs)
- Best config (Run 2):
```json
{
  "max_fevals_transfer": 15,
  "sigma0_transfer": 0.12,
  "n_clusters_1src": 4,
  "n_clusters_2src": 6
}
```

## Conclusion
The cluster transfer approach is **fundamentally flawed** for this heat source identification problem. The relationship between sensor observations and optimal source configurations is too non-linear for feature-based clustering to provide useful initialization. This experiment confirms that **reducing simulations through smart optimization (lq-CMA-ES, multi-fidelity)** is more promising than **trying to transfer knowledge between samples**.

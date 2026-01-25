# MAE Polish Loss Experiment

## Experiment ID: EXP_MAE_POLISH_001
**Status**: ABORTED
**Worker**: W1
**Date**: 2026-01-24

## Hypothesis

RMSE emphasizes large errors while MAE (Mean Absolute Error) is more robust to outliers. Using MAE for the NM polish phase might lead to better overall fit.

## Reason for Abortion

This experiment was ABORTED based on prior evidence from related experiments:

### Prior Evidence

1. **EXP_LOG_RMSE_LOSS_001** (FAILED):
   - Used log(1+RMSE) instead of RMSE for optimization
   - Score: 1.1677 vs baseline 1.1688 (-0.0011)
   - Time: 70.5 min (21% over budget)
   - Monotonic transformations preserve rankings but add overhead

2. **EXP_WEIGHTED_SENSOR_LOSS_001** (FAILED):
   - Used weighted RMSE based on sensor importance
   - Score: 1.0131 vs baseline 1.1688 (-0.1557)
   - Time: 90.7 min (over budget)
   - **Critical finding**: Optimizing a different loss finds a DIFFERENT optimum than what we're scored on

### Why MAE Polish Would Fail

The MAE minimizer is NOT the same as the RMSE minimizer:
- RMSE minimizer = arithmetic mean of residuals
- MAE minimizer = median of residuals (approximately)

When the residuals are asymmetric or have outliers, these give DIFFERENT solutions. Since the final scoring uses RMSE, optimizing MAE during polish would:
1. Find a solution that minimizes MAE, not RMSE
2. Result in suboptimal RMSE score
3. Add no benefit while potentially hurting accuracy

## Key Finding

**Changing the loss function is fundamentally unsound for this problem.**

The scoring formula uses RMSE:
```
score = (1/N) * Σ(1/(1+RMSE_i)) + λ * (N/N_max)
```

Any optimization must directly minimize RMSE to maximize this score. Using a different loss function (MAE, weighted RMSE, log-RMSE) finds a different optimum that is NOT optimal for the true objective.

## Recommendation

The `polish_loss` family should be marked as EXHAUSTED:
- Log-RMSE: FAILED
- Weighted RMSE: FAILED
- MAE: ABORTED (would fail for same reason)
- Any other loss transformation: Would fail for same reason

**DO NOT modify the loss function.** RMSE is the correct objective and must be used throughout.

## Files
- `STATE.json`: Experiment state (aborted)
- `optimizer.py`: Copy of baseline (unused)

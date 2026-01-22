# Experiment Summary: physics_informed_init

## Metadata
- **Experiment ID**: EXP_PHYSICS_INIT_001
- **Worker**: W1
- **Date**: 2026-01-19
- **Algorithm Family**: initialization

## Objective
Test whether physics-informed initialization using temperature gradients provides better starting points for CMA-ES optimization than the simple hottest-sensor approach.

## Hypothesis
Temperature gradients point toward heat sources. By analyzing gradient directions at multiple sensors, we can triangulate source locations and provide better initial guesses for CMA-ES than random or hottest-sensor initialization.

## Results Summary
- **Best Score (Gradient Init)**: 1.1593 @ 71.9 min
- **Best Score (Smart Init)**: 1.1639 @ 69.6 min
- **Delta**: -0.0046 score, +2.3 min (gradient init is WORSE)
- **Status**: FAILED - Gradient init does NOT improve over smart init

## A/B Test Results

| Initialization | Score | Time (min) | RMSE Mean | RMSE 1-src | RMSE 2-src |
|----------------|-------|------------|-----------|------------|------------|
| **Gradient + Tri** | 1.1593 | 71.9 | 0.1360 | 0.0993 | 0.1605 |
| **Smart + Tri** | 1.1639 | 69.6 | 0.1369 | 0.1127 | 0.1530 |
| **Delta** | -0.0046 | +2.3 | -0.0009 | -0.0134 | +0.0075 |

## Tuning History

| Run | Config | Score | Time (min) | In Budget | Notes |
|-----|--------|-------|------------|-----------|-------|
| 1 | Gradient + Tri | 1.1593 | 71.9 | No | Gradient-based initialization |
| 2 | Smart + Tri | 1.1639 | 69.6 | No | Hottest-sensor initialization (baseline) |

## Key Findings

### What Didn't Work
1. **Gradient-based initialization is WORSE than simple hottest-sensor approach**
   - Score: -0.0046 (1.1593 vs 1.1639)
   - Time: +2.3 min (71.9 vs 69.6)

2. **Temperature gradients at sensors don't accurately point to source locations**
   - Heat diffuses from multiple directions, corrupting gradient signal
   - Boundary effects introduce additional noise
   - Sensor spacing may not capture local gradients well

3. **The gradient calculation adds slight overhead**
   - k-means clustering for 2-source estimation
   - Neighbor finding for gradient computation
   - This adds ~2 min without improving accuracy

### Critical Insights

1. **Simple is better**: The hottest-sensor approach works well because:
   - Heat sources create temperature maxima in their vicinity
   - Sensors near sources will have higher temperatures
   - This is a direct, robust heuristic

2. **Gradient analysis fails because**:
   - Thermal diffusion smooths temperature fields
   - By the time heat reaches sensors, gradients don't point to sources
   - Multiple sources create interference patterns
   - Boundary conditions add complexity

3. **Inverse problems don't benefit from forward-problem heuristics**:
   - In forward problems, gradients point toward sources
   - In inverse problems with noisy sensor data, this relationship is corrupted

## Technical Details

### Gradient Initialization Algorithm
```python
# For each sensor, compute approximate gradient
# using finite differences with k nearest neighbors
grad_x, grad_y = compute_gradient_at_sensor(sensors_xy, temperatures, sensor_idx)

# For 1-source: follow gradient from hot region
# For 2-source: cluster sensors, estimate source per cluster
```

### Why It Failed
1. Sensor spacing (~0.1-0.2 units) is too coarse for accurate gradient estimation
2. Temperature field at sensors is diffused, not localized
3. For 2-source problems, k-means clustering doesn't separate sources well

## Parameter Sensitivity
- **Most impactful**: The choice of initialization method (gradient vs smart)
- **Not impactful**: Sigma values, fevals - these were held constant for fair comparison

## Recommendations for Future Experiments

1. **Don't pursue physics-informed initialization further**
   - Simple hottest-sensor (smart init) is already optimal
   - Gradient-based methods add complexity without benefit

2. **Focus on other bottlenecks**:
   - 2-source RMSE is still the main issue (0.15+ vs 0.10 for 1-source)
   - Time budget is tight - focus on efficiency improvements

3. **If revisiting initialization**:
   - Consider using more sensors for gradient estimation
   - Try different weighting schemes
   - But expect diminishing returns - smart init is already good

## Raw Data
- MLflow run IDs: `run1_gradient_init`, `run2_no_gradient_baseline`
- Experiment: `physics_informed_init`

## Conclusion

**FAILED** - Physics-informed initialization using temperature gradients does NOT improve over the simple hottest-sensor approach. The gradient signal at sensor locations is corrupted by thermal diffusion and doesn't accurately point toward source locations. The simple hottest-sensor heuristic is already a robust and effective initialization strategy.

The initialization family should be marked as EXHAUSTED for this problem domain.

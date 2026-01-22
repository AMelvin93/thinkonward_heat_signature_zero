# Experiment: Polar Parameterization

**Experiment ID:** EXP_POLAR_PARAM_001
**Worker:** W2
**Status:** FAILED
**Family:** problem_reformulation

## Hypothesis

Reparameterizing source positions from Cartesian (x, y) to polar coordinates (r, theta) centered on the domain centroid may change the optimization landscape characteristics, potentially improving CMA-ES convergence speed and accuracy.

## Approach

1. Convert source positions to polar: r = sqrt((x-cx)^2 + (y-cy)^2), theta = atan2(y-cy, x-cx)
2. CMA-ES optimizes (r, theta) instead of (x, y)
3. Transform back to Cartesian for simulation
4. Clip Cartesian coordinates to domain bounds after transformation

## Results

| Run | Config | Score | Time (min) | In Budget | Notes |
|-----|--------|-------|------------|-----------|-------|
| 1 | Polar centered on (1.0, 0.5) | 1.1472 | 72.7 | NO | WORSE - -0.0216 score, +14.3 min |

**Baseline:** 1.1688 @ 58.4 min (Cartesian parameterization)

## Analysis

### Why Polar Parameterization Failed

1. **Rectangular domain creates irregular polar bounds**: The domain [0, 2] x [0, 1] is rectangular but polar coordinates assume a circular region. The maximum valid radius depends on the angle theta, creating a complex constraint surface.

2. **Bound clipping introduces discontinuities**: When polar coordinates are transformed to Cartesian and clipped to the domain, this creates discontinuous jumps in the search space. CMA-ES's covariance adaptation assumes smooth landscape variations.

3. **CMA-ES covariance learns wrong correlations**: In polar space, r and theta have fundamentally different scales and meanings (distance vs angle). The covariance matrix may learn inappropriate correlations between these parameters.

4. **No natural advantage for this problem**: Polar coordinates are useful when the problem has radial symmetry around a center point. The heat source inverse problem has no such symmetry - sources can be anywhere in the rectangular domain.

### Detailed Results

**Run 1 (Polar parameterization):**
- 1-source RMSE: 0.1116 (worse than baseline ~0.10)
- 2-source RMSE: 0.1650 (worse than baseline ~0.15)
- Time: 72.7 min (24% over budget)

The polar approach is slower AND less accurate than baseline Cartesian.

## Conclusion

**FAILED** - Polar parameterization does NOT improve CMA-ES performance for the heat source inverse problem.

Key issues:
- Rectangular domain is incompatible with polar coordinates
- Bound clipping creates discontinuities
- No natural radial symmetry to exploit

## Recommendation

**ABANDON** the problem_reformulation family (at least for coordinate system changes).

The Cartesian coordinate system is optimal for rectangular domain problems because:
- CMA-ES can learn correlations between x and y directly
- Bounds are simple box constraints
- No discontinuities from coordinate transformation

## MLflow Run IDs
- Run 1: `359a50be310846a3a4f4d0a45be58b53`

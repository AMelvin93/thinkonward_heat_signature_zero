# Experiment: tighter_sigma_with_coord

## Status: FAILED - Coordinate refinement cannot match perturbation

## Results Summary

| Config | Score | Time (min) | Projected 400 | vs Perturbation |
|--------|-------|------------|---------------|-----------------|
| sigma_015_019_coord_step001 | 1.1556 | 7.6 | 38.1 min | -0.0174 |
| sigma_015_019_no_coord_baseline | 1.1597 | 7.4 | 36.8 min | -0.0133 |
| sigma_015_019_nm6_coord_step001 | 1.1602 | 7.3 | 36.7 min | -0.0128 |
| **tighter_sigma_range (perturbation)** | **1.1730** | 10.1 | 50.4 min | baseline |

## Hypothesis
Coordinate refinement provided +0.0105 on older configs. Combining with optimal sigma 0.15/0.19 could push score even higher than perturbation.

## Key Findings

### 1. Coordinate refinement CANNOT match perturbation
- Best with coord refine: 1.1602
- Perturbation baseline: 1.1730
- Delta: **-0.0128** (significantly worse)

### 2. Coord refine can actually HURT performance
| Config | Score | Observation |
|--------|-------|-------------|
| NM 8 + coord | 1.1556 | WORSE |
| NM 8 no coord | 1.1597 | Better |
| NM 6 + coord | 1.1602 | Best here |

With 8 NM iterations, adding coord refine DECREASES score by -0.0041!

### 3. Perturbation is essential
Without perturbation, optimal sigma alone achieves only 1.1597-1.1602.
With perturbation, optimal sigma achieves 1.1730 (+0.0128 to +0.0174 improvement).

### 4. Time comparison
Coord refine is faster (36-38 min) but less accurate.
Perturbation uses more time (50.4 min) but achieves much better accuracy.

## RMSE Breakdown

| Config | RMSE 1-src | RMSE 2-src |
|--------|------------|------------|
| coord_step001 | 0.1418 | 0.1958 |
| no_coord_baseline | 0.1255 | 0.2011 |
| nm6_coord_step001 | 0.1284 | 0.1967 |
| **perturbation** | **0.1160** | **0.1748** |

Perturbation achieves much better RMSE on both 1-src and 2-src problems.

## Analysis

### Why Perturbation Beats Coord Refine
1. **Global vs Local**: Perturbation explores different basins; coord refine only searches within current basin
2. **Local minima escape**: ~25% of samples benefit from perturbation's basin hopping
3. **Coord refine redundancy**: NM polish already does local refinement; coord adds little value

### Why Coord Refine Can Hurt
1. Coord refine uses a fixed step size (0.01) that may overshoot the optimum
2. The coarse grid objective (40% temporal) introduces noise
3. Axis-aligned search may miss diagonal improvements that NM finds

## Conclusion

**FAILED** - Coordinate refinement cannot compete with perturbation.

- Best coord refine: 1.1602 @ 36.7 min
- Perturbation baseline: 1.1730 @ 50.4 min
- Delta: **-0.0128** (1.1% worse)

## Recommendations

1. **DO NOT use coord refinement** - it provides no benefit over perturbation
2. **Perturbation remains the winning technique** for escaping local minima
3. **Mark coord_refine_v2 family as EXHAUSTED**
4. **Focus future experiments on perturbation improvements**

## Family Status

`coord_refine_v2` - **EXHAUSTED**

Coordinate refinement has been thoroughly tested:
- coord_refine_plus_perturbation: FAILED (don't stack)
- tighter_sigma_with_coord: FAILED (doesn't match perturbation)

The technique is inferior to perturbation-based basin hopping.

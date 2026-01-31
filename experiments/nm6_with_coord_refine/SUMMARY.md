# 6 NM Iterations + Coordinate Refinement Experiment

## Status: SUCCESS - Found best IN-BUDGET configuration!

## Hypothesis
6 NM iterations + coordinate refinement should fit within 60 min budget while maintaining accuracy gains.

## Results Summary

| Run | Config | Score | Time (min) | Budget Status |
|-----|--------|-------|------------|---------------|
| 1 | **nm6_coord001** | **1.1602** | 57.6 | IN BUDGET |
| 2 | nm6_coord0015 | 1.1554 | 59.8 | IN BUDGET |
| 3 | nm5_coord001 | 1.1595 | 57.2 | IN BUDGET |

## Key Findings

### 1. All Configs In Budget!
All three configurations complete within the 60-minute budget:
- nm6_coord001: 57.6 min
- nm6_coord0015: 59.8 min
- nm5_coord001: 57.2 min

### 2. Best Configuration Found
**nm6_coord001 (6 NM iterations + coord step=0.01)** achieves the best score:
- Score: 1.1602
- Time: 57.6 min
- Well within budget with 2.4 min buffer

### 3. Step Size Comparison
| Step Size | Score | Observation |
|-----------|-------|-------------|
| 0.01 | 1.1602 | Best |
| 0.015 | 1.1554 | -0.0048 worse |

Smaller step size (0.01) is optimal for coordinate refinement.

### 4. NM Iterations Comparison
| NM Iters | Score | Time |
|----------|-------|------|
| 6 | 1.1602 | 57.6 min |
| 5 | 1.1595 | 57.2 min |

6 iterations slightly better than 5, with minimal time difference.

## Comparison to Previous Experiments

| Configuration | Score | Time | Budget |
|---------------|-------|------|--------|
| No coord refine, 7 NM | 1.1532 | 61.1 min | OVER |
| Coord refine (0.01), 7 NM | 1.1637 | 63.8 min | OVER |
| **Coord refine (0.01), 6 NM** | **1.1602** | **57.6 min** | **IN** |

The 6 NM + coord config sacrifices 0.0035 score vs 7 NM + coord, but gains budget compliance.

## Conclusion

**nm6_coord001 is the new recommended production configuration:**
- sigma0_1src: 0.18
- sigma0_2src: 0.22
- timestep_fraction: 0.40
- refine_maxiter: 6
- enable_coord_refine: True
- coord_step: 0.01

This achieves:
- Score: 1.1602 (improvement from baseline)
- Time: 57.6 min (well within 60 min budget)

## Recommendations

1. **Adopt nm6_coord001 as production config** - best accuracy within budget
2. **Add coordinate refinement to the CoordRefineOptimizer** for future experiments
3. **Mark this experiment as SUCCESS** - found actionable improvement

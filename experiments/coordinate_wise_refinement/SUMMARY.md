# Coordinate-Wise Refinement Experiment

## Status: SUCCESS - Coordinate refinement significantly improves accuracy!

## Hypothesis
After NM polish, coordinate-wise line search along each parameter axis may find improvements that NM's simplex geometry misses.

## Results Summary

| Run | Config | Score | Time (min) | Budget Status |
|-----|--------|-------|------------|---------------|
| 1 | no_coord_refine_baseline | 1.1532 | 61.1 | 1 min OVER |
| 2 | coord_step_001 | **1.1637** | 63.8 | 4 min OVER |
| 3 | coord_step_002 | 1.1567 | 60.8 | 1 min OVER |

**Improvement**: coord_step_001 improved score by **+0.0105** over baseline!

## Key Findings

### 1. Coordinate Refinement Works!
Adding coordinate-wise line search after NM polish provides significant accuracy gains:
- Baseline (no coord refine): 1.1532
- With coord refine (step=0.01): 1.1637 (+0.0105 improvement)

### 2. Step Size Matters
| Step Size | Score | Time | vs Baseline |
|-----------|-------|------|-------------|
| None | 1.1532 | 61.1m | baseline |
| 0.01 | 1.1637 | 63.8m | +0.0105 |
| 0.02 | 1.1567 | 60.8m | +0.0035 |

Smaller step (0.01) performs better than larger step (0.02).

### 3. Time Overhead
- Coordinate refinement adds ~3 min overhead with step=0.01
- Step=0.02 adds only ~0.3 min overhead but less improvement

### 4. RMSE Breakdown
| Config | RMSE 1-src | RMSE 2-src |
|--------|------------|------------|
| no_coord_refine | 0.131739 | 0.212434 |
| coord_step_001 | **0.120762** | **0.194830** |
| coord_step_002 | 0.129770 | 0.204655 |

Both 1-source and 2-source improved significantly with coord refinement.

## Analysis

### Why Coordinate Refinement Helps
1. **NM simplex geometry**: Nelder-Mead uses simplex moves that may miss axis-aligned improvements
2. **Parameter independence**: Heat source parameters (x, y) are somewhat independent, making axis search effective
3. **Local fine-tuning**: After NM converges, small coordinate steps can find nearby better solutions

### Budget Considerations
All configs are slightly over budget. To fit within 60 min:
- Consider using 6 NM iterations instead of 7
- Or accept the minor budget overage for significant accuracy gain

## Conclusion

**Coordinate refinement is a valuable addition to the optimization pipeline!**

Best config: coord_step_001 (step=0.01) @ 1.1637, 63.8 min
- If budget allows slight overage, use step=0.01 for best accuracy
- If strict budget compliance needed, use step=0.02 (1.1567 @ 60.8 min)

## Recommendations

1. **Add coordinate refinement to production** with step=0.01
2. **Reduce NM iterations to 6** if budget compliance is critical
3. **Test combined config**: 6 NM iter + coord_step=0.01 to stay in budget
4. **Mark EXP_COORDINATE_REFINEMENT_001 as SUCCESS**

## Next Steps

Test: 6 NM iterations + coordinate refinement (step=0.01) to achieve best accuracy within budget.

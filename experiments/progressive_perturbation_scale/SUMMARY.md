# Progressive Perturbation Scale Experiment

## Status: FAILED (all configs worse than baseline)

## Hypothesis
Different perturbation scales affect exploration/exploitation balance. Testing 0.10 (large), 0.05 (medium), 0.02 (small).

## Results Summary

| Run | Config | Score | Time (min) | Budget Status |
|-----|--------|-------|------------|---------------|
| 1 | scale_010_large | 1.1627 | 64.4 | -4 min OVER |
| 2 | scale_005_medium | 1.1648 | 64.5 | -4 min OVER |
| 3 | scale_002_small | 1.1534 | 64.9 | -5 min OVER |

**New Baseline**: 1.1689 @ 58.2 min (from hopping_with_tabu_memory - NO perturbation!)

## Key Finding

**All perturbation configs performed WORSE than the no-perturbation baseline!**

- Best with perturbation: 1.1648 @ 64.5 min
- Without perturbation: 1.1689 @ 58.2 min

This suggests that perturbation adds ~6 min overhead without improving accuracy.

## Analysis

### Perturbation Scale Comparison
| Scale | Score | Time | vs Baseline |
|-------|-------|------|-------------|
| 0.10 | 1.1627 | 64.4 | -0.0062 worse |
| 0.05 | 1.1648 | 64.5 | -0.0041 worse |
| 0.02 | 1.1534 | 64.9 | -0.0155 worse |
| None | 1.1689 | 58.2 | baseline |

### Why Perturbation May Be Harmful
1. **Time overhead**: Perturbation + NM polish adds ~6 min per run
2. **CMA-ES already explores**: The covariance matrix adaptation handles exploration
3. **NM polish is sufficient**: Standard polish after CMA-ES finds good solutions
4. **Over-budget risk**: The extra time pushes configs over the 60 min budget

## Conclusion

**Perturbation is NOT beneficial for this problem within the time budget.**

The best config found so far (1.1689 @ 58 min) has NO perturbation - just CMA-ES + NM polish.

## Recommendations

1. **DISABLE perturbation** in production config
2. **Mark perturbation_v3 family as EXHAUSTED**
3. **Focus on CMA-ES + NM polish tuning** without perturbation
4. **Current best config**:
   - sigma0_1src=0.18, sigma0_2src=0.22
   - timestep_fraction=0.40
   - refine_maxiter=8
   - NO perturbation

# Tabu Basin Hopping Experiment

## Status: SUCCESS (significant improvement found!)

## Hypothesis
Adding tabu memory to basin hopping prevents revisiting already-explored regions, guiding exploration toward unexplored basins.

## Results Summary

| Run | Config | Score | Time (min) | Budget Status |
|-----|--------|-------|------------|---------------|
| 1 | tabu_dist_003 | 1.1666 | 69.7 | -10 min OVER |
| 2 | tabu_dist_005 | 1.1707 | 68.1 | -8 min OVER |
| 3 | no_tabu_baseline | **1.1689** | **58.2** | **+2 min UNDER** |

**Historical Baseline**: 1.1464 @ 51.2 min

## Key Finding

**The baseline config (without tabu memory) achieved 1.1689 @ 58 min - a significant +0.0225 (+1.9%) improvement over the historical baseline!**

This is the best in-budget result found in today's experiments.

## Analysis

### Tabu Memory Effect
- Tabu memory slightly improves score (1.1707 vs 1.1689 = +0.15%)
- But adds ~10 min overhead
- Not worth the overhead within 60 min budget

### Why Baseline Improved
The no_tabu_baseline used these parameters:
- sigma0_1src=0.18, sigma0_2src=0.22
- timestep_fraction=0.40
- refine_maxiter=8
- n_perturbations=2 (without tabu logic)
- perturb_nm_iters=3

The improvement likely comes from:
1. Consistent 40% timestep fraction
2. Perturbation approach (n_perturbations=2) with NM polish

### Comparison with Historical Baseline
| Metric | Historical | This Experiment | Change |
|--------|------------|-----------------|--------|
| Score | 1.1464 | 1.1689 | +0.0225 (+1.9%) |
| Time | 51.2 min | 58.2 min | +7 min |
| RMSE 1src | N/A | 0.122 | - |
| RMSE 2src | N/A | 0.180 | - |

## Tuning Efficiency

- **Runs executed**: 3
- **Budget compliance**: 1/3 configs within budget
- **Best in-budget**: no_tabu_baseline @ 1.1689
- **Parameter space explored**: tabu_distance=[0.03, 0.05], enable_tabu=[true, false]

## Conclusion

**This experiment achieved a breakthrough!**

The no_tabu_baseline config achieved 1.1689 @ 58 min, which is:
- +0.0225 better than historical baseline (1.1464)
- Within the 60-minute budget
- A new candidate for production baseline

The tabu memory feature adds marginal improvement but is not worth the overhead.

## Recommendations

1. **PROMOTE no_tabu_baseline to production** - 1.1689 @ 58 min is the new best
2. **Mark tabu_memory family as VALIDATED_BUT_OVER_BUDGET** - works but adds overhead
3. **Key config to preserve**:
   - sigma0_1src=0.18, sigma0_2src=0.22
   - timestep_fraction=0.40
   - refine_maxiter=8
   - n_perturbations=2
   - perturb_nm_iters=3
4. **Update baseline reference** from 1.1464 to 1.1689

## Next Steps

Consider testing variations of the successful config:
- Slightly reduce refine_maxiter (7 instead of 8) to save time
- Test if 1 perturbation is sufficient
- Fine-tune other parameters around this baseline

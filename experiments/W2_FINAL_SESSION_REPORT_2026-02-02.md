# W2 Final Session Report - 2026-02-02

## Executive Summary

This session focused on finding novel approaches to improve the score beyond the validated 4pert_nm2 config (1.1482 @ 51.7 min). **All novel approaches tested failed** - either couldn't fit budget or didn't improve accuracy.

## Experiments Run

### 1. Sequential 2-Source Estimation (3 runs)
**Result: FAILED - Fundamentally flawed**

| Config | Score | Time | vs Baseline |
|--------|-------|------|-------------|
| fevals 15/22 | 1.0214 | 74.2 min | -0.1268 |
| fevals 20/44 | 1.0401 | 109.0 min | -0.1081 |
| fevals 10/15 | 1.0157 | 42.5 min | -0.1325 |

**Conclusion**: Sequential 2D+2D decomposition introduces compounding errors. Joint 4D CMA-ES is fundamentally superior.

### 2. 5 Perturbations Test (3 runs)
**Result: FAILED - Cannot fit budget**

| Config | Score | Time | vs Baseline |
|--------|-------|------|-------------|
| 5pert_nm2 | 1.1534 | 76.0 min | +0.0052 OVER |
| 5pert_nm1 | 1.1533 | 76.1 min | +0.0051 OVER |
| reduced_fevals | 1.1567 | 80.1 min | +0.0085 OVER |

**Conclusion**: 5 perturbations improves score (+0.005-0.009) but adds ~24 min overhead regardless of other settings. Cannot fit in 60-minute budget.

### 3. tau=0.15 Test (1 run - technical failure)
**Result: Technical failure** - Monkey-patch didn't work with multiprocessing.

## Total Runs: 7

## Key Findings

### 1. 5 Perturbations Would Help But Can't Fit
- Score improvement: +0.005 to +0.009
- Time penalty: +24 min fixed overhead
- Budget gap: 16-20 min over

### 2. Sequential Decomposition is Flawed
- Physics insight (heat equation linearity) is correct
- But optimization decomposition introduces errors
- Joint 4D CMA-ES with covariance adaptation is superior

### 3. Current Config is Near-Optimal
- 4pert_nm2: 1.1482 @ 51.7 min (8.3 min remaining)
- All tuning dimensions exhausted
- Novel approaches tested and failed

## Production Configuration (Unchanged)

```python
PRODUCTION_CONFIG = {
    'sigma0_1src': 0.18,
    'sigma0_2src': 0.22,
    'max_fevals_1src': 20,
    'max_fevals_2src': 44,
    'timestep_fraction': 0.40,
    'refine_maxiter': 8,
    'enable_tabu_hopping': True,
    'n_perturbations': 4,
    'perturb_nm_iters': 2,
    'perturbation_scale': 0.05,
    'tabu_distance': 0.04,
    'max_tabu_attempts': 10,
}
# Validated: 1.1482 ± 0.0030 @ 51.7 min
```

## Leaderboard Position

| Rank | Team | Score | Gap |
|------|------|-------|-----|
| 10 | MGöksu | 1.1585 | +0.0103 |
| **Us** | **4pert_nm2** | **1.1482** | -- |
| 11 | bobatea | 1.1295 | -0.0187 |

## What Would Help (But Can't Achieve)

1. **5 perturbations**: Would add +0.005-0.009 but 24 min over budget
2. **Lower tau threshold**: Would increase diversity but couldn't test properly
3. **Better 2-source RMSE**: Current ~0.19, would need ~0.17 for Top 10

## Recommendations

1. **Submit with 4pert_nm2 config** - It's the best validated configuration
2. **Accept current leaderboard position (~11th)** - Top 10 would require novel algorithmic breakthroughs
3. **Focus on documentation for innovation score** - Document all approaches tried and learnings

## Session Statistics

- **Total experiments**: 3 (sequential_2src, 5_perturbations, tau_015)
- **Total runs**: 7
- **Novel approaches tested**: 3
- **Improvements found**: None within budget
- **Time budget utilized**: 86% (51.7/60 min for production config)

## Conclusion

The 4pert_nm2 configuration at 1.1482 @ 51.7 min represents the practical limit of the current approach within the 60-minute budget. All tested approaches to improve the score either:
1. Cannot fit within the time budget (5 perturbations, higher fevals)
2. Are fundamentally flawed (sequential decomposition)
3. Have already been exhausted (all parameter tuning)

To reach Top 10 (1.1585), a fundamentally new algorithmic approach would be needed that hasn't been discovered in the extensive exploration.

---
**Worker**: W2
**Date**: 2026-02-02
**Session Duration**: ~3 hours
**Status**: No improvement found - all novel approaches failed

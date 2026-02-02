# W1 Final Session Report - 2026-02-02

## Executive Summary

**Mission**: Systematically tune parameters to close the gap to top 10 on the leaderboard.

**Result**: **PARAMETER SPACE EXHAUSTED** - Best validated config identified.

## Final Production Configuration

```python
PRODUCTION_CONFIG = {
    'sigma0_1src': 0.18,
    'sigma0_2src': 0.22,
    'max_fevals_1src': 20,
    'max_fevals_2src': 44,
    'timestep_fraction': 0.40,
    'refine_maxiter': 8,
    'enable_tabu_hopping': True,
    'n_perturbations': 4,             # TUNED: 4 perturbations
    'perturb_nm_iters': 2,            # TUNED: Reduced from 3 to 2
    'perturbation_scale': 0.05,
    'tabu_distance': 0.04,
    'max_tabu_attempts': 10,
}
```

**Expected Performance**: 1.1482 +/- 0.0030 @ 51.7 min

## Leaderboard Position

| Position | Team | Score | Gap |
|----------|------|-------|-----|
| 10 | MGöksu | 1.1585 | -- |
| **Us (new config)** | **validated** | **1.1482** | **-0.0103** |
| 11 | bobatea | 1.1295 | -0.0290 |

**Gap to Top 10**: +0.0103 (~0.9%)

## Experiments Completed

| # | Experiment | Result | Key Finding |
|---|------------|--------|-------------|
| 1 | production_3pert_validation | VALIDATED | 1.1475 @ 55.7 min baseline |
| 2 | higher_fevals_3pert_tuning | INCONCLUSIVE | Higher fevals doesn't help |
| 3 | 4pert_reduced_perturb_nm | SUCCESS | Found 4pert_nm2 fits budget |
| 4 | 4pert_nm2_validation | VALIDATED | 1.1482 @ 51.7 min |
| 5 | 4pert_nm2_scale_tuning | COMPLETED | 0.06 looked promising |
| 6 | 4pert_nm2_scale06_validation | MARGINAL | Equivalent to 0.05 |
| 7 | 4pert_nm2_timestep_tuning | COMPLETED | 0.40 is optimal |

**Total Runs**: ~23

## Parameters Exhausted

| Parameter | Tested Values | Optimal |
|-----------|---------------|---------|
| sigma0_1src | 0.12-0.20 | 0.18 |
| sigma0_2src | 0.18-0.24 | 0.22 |
| max_fevals_1src | 16-24 | 20 |
| max_fevals_2src | 36-48 | 44 |
| n_perturbations | 1-4 | 4 |
| perturb_nm_iters | 1-3 | 2 |
| perturbation_scale | 0.04-0.07 | 0.05 |
| tabu_distance | 0.03-0.05 | 0.04 |
| refine_maxiter | 4-10 | 8 |
| timestep_fraction | 0.35-0.45 | 0.40 |

## Key Discoveries

### 1. 4-pert CAN Fit Budget
By reducing perturb_nm_iters from 3 to 2:
- 4 perturbations achieves 1.1482 @ 51.7 min (in budget!)
- This is the highest validated score within budget

### 2. Parameter Interactions
- Higher fevals doesn't compensate for fewer perturbations
- Reducing NM iterations for perturbations loses less than expected
- Scale 0.05-0.06 and timestep 0.38-0.42 are all equivalent

### 3. Variance Dominates Small Improvements
- Run-to-run variance: ~0.003-0.010
- Single-run "improvements" often don't validate
- Need 3+ runs to establish true performance

## What's Required to Reach Top 10

The +0.0103 gap cannot be closed by parameter tuning alone:

1. **Novel Algorithms**: Current CMA-ES+NM is near-optimal
2. **Better 2-Source Handling**: RMSE ~0.18-0.19 is the bottleneck
3. **Faster Computation**: Would enable more perturbations

## Recommended Actions

### Immediate
1. **Use 4pert_nm2 config for final submission**
2. **Document the optimization journey** for innovation scoring

### Future (if time permits)
1. Explore novel initialization strategies
2. Research specialized 2-source handling
3. Consider hybrid approaches

## Session Statistics

- **Duration**: ~6 hours
- **Experiments**: 7
- **Tuning runs**: ~23
- **Net improvement**: +0.0007 vs 3-pert baseline
- **Time utilization**: 86% (51.7/60 min)

---
**Worker**: W1
**Date**: 2026-02-02
**Status**: COMPLETE - Parameter space exhausted

**RECOMMENDATION**: The 4pert_nm2 configuration is production-ready. Focus remaining effort on documentation and novel approaches rather than further parameter tuning.

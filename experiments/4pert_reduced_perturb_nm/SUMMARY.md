# Experiment: 4pert_reduced_perturb_nm

## Status: SUCCESS - NEW BEST IN-BUDGET CONFIGURATION!

## Hypothesis
The 4-pert config achieves 1.1555 @ 61.9 min (1.9 min over budget).
Reducing perturb_nm_iters from 3 to 2 could save ~2 min, fitting 4-pert within budget.

## Results

| Config | perturb_nm_iters | Score | Time (min) | Budget | vs 3-pert |
|--------|------------------|-------|------------|--------|-----------|
| 4pert_nm3 | 3 | 1.1566 | 65.0 | OVER | +0.0091 |
| **4pert_nm2** | **2** | **1.1524** | **58.3** | **IN** | **+0.0049** |
| 4pert_nm1 | 1 | 1.1479 | 50.7 | IN | +0.0004 |

**3-pert baseline**: 1.1475 @ 55.7 min

## Key Finding: NEW BEST IN-BUDGET!

**4 perturbations with perturb_nm_iters=2 achieves 1.1524 @ 58.3 min (IN BUDGET!)**

### Improvement Analysis
- Score: 1.1524 (+0.0049 vs 3-pert baseline)
- Time: 58.3 min (1.7 min buffer)
- **Gap to top 10 reduced from +0.0110 to +0.0061** (45% improvement!)

### Trade-off Analysis
| Metric | 4pert_nm3 | 4pert_nm2 | Change |
|--------|-----------|-----------|--------|
| Score | 1.1566 | 1.1524 | -0.0042 |
| Time | 65.0 | 58.3 | -6.7 min |
| Budget | OVER | IN | ✓ |

Reducing perturb_nm_iters from 3 to 2:
- Loses 0.0042 in score
- Saves 6.7 minutes
- Gets us IN BUDGET!

## RMSE Analysis

| Config | RMSE 1src | RMSE 2src |
|--------|-----------|-----------|
| 4pert_nm3 | 0.1169 | 0.1701 |
| **4pert_nm2** | **0.1109** | **0.1891** |
| 4pert_nm1 | 0.1390 | 0.1838 |

The 4pert_nm2 config has:
- Best 1-source RMSE (0.1109)
- Slightly worse 2-source RMSE (0.1891 vs 0.1701)
- Good overall balance

## Recommended Production Configuration

```python
PRODUCTION_CONFIG = {
    'sigma0_1src': 0.18,
    'sigma0_2src': 0.22,
    'max_fevals_1src': 20,
    'max_fevals_2src': 44,
    'timestep_fraction': 0.40,
    'refine_maxiter': 8,
    'enable_tabu_hopping': True,
    'n_perturbations': 4,             # KEY: 4 perturbations
    'perturb_nm_iters': 2,            # KEY: Reduced from 3 to 2
    'perturbation_scale': 0.05,
    'tabu_distance': 0.04,
    'max_tabu_attempts': 10,
}

# Expected: 1.1524 @ 58.3 min
```

## Leaderboard Context

| Rank | Team | Score | Gap from New Config |
|------|------|-------|---------------------|
| 10 | MGöksu | 1.1585 | +0.0061 |
| **Us (4pert_nm2)** | **NEW** | **1.1524** | -- |
| Us (3-pert old) | old | 1.1475 | -0.0049 |
| 11 | bobatea | 1.1295 | -0.0229 |

**Gap to top 10 closed from +0.0110 to +0.0061** (45% improvement!)

## Comparison to All Tested Configs

| Config | Score | Time | Notes |
|--------|-------|------|-------|
| **4pert_nm2 (NEW)** | **1.1524** | **58.3** | **NEW BEST** |
| 3pert (validated) | 1.1475 | 55.7 | Previous best |
| 4pert_nm3 | 1.1566 | 65.0 | Over budget |
| 4pert_nm1 | 1.1479 | 50.7 | Too aggressive |

## Next Steps

### IMMEDIATE: Validation (Recommended)
Run 3-5 validation runs of the 4pert_nm2 config to establish confidence bounds.

### If Validated
Promote to production and update final submission.

## Tuning Efficiency Metrics

- **Runs executed**: 3
- **Time utilization**: 97% (58.3/60 min)
- **Parameter space explored**: perturb_nm_iters [1, 2, 3]
- **Conclusion**: perturb_nm_iters=2 is optimal for 4 perturbations

---
**Worker**: W1
**Completed**: 2026-02-02
**Runs**: 3
**Result**: SUCCESS - 1.1524 @ 58.3 min (NEW BEST IN-BUDGET!)

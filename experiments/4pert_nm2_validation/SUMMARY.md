# Experiment: 4pert_nm2_validation

## Status: VALIDATED SUCCESS - Marginal improvement over 3-pert

## Purpose
Validate the new candidate config: 4 perturbations + perturb_nm_iters=2

## Prior Results
- Initial tuning: 1.1524 @ 58.3 min
- 3-pert baseline: 1.1475 @ 55.7 min

## Validation Results (3 Runs)

| Run | Score | Time (min) | RMSE 1src | RMSE 2src | Budget |
|-----|-------|------------|-----------|-----------|--------|
| 1   | 1.1444 | 52.4 | 0.1279 | 0.1855 | **IN** |
| 2   | 1.1486 | 51.6 | 0.1165 | 0.1893 | **IN** |
| 3   | 1.1517 | 51.3 | 0.1193 | 0.1775 | **IN** |

## Statistics

| Metric | Value |
|--------|-------|
| **Mean Score** | **1.1482 +/- 0.0030** |
| Score Range | [1.1444, 1.1517] |
| **Mean Time** | **51.7 min** |
| **vs 3-pert baseline** | **+0.0007** |
| Runs in budget | **3/3 (100%)** |
| **Gap to Top 10** | +0.0103 |

## Key Finding: VALIDATED BUT MARGINAL

The 4-pert_nm2 config shows a small improvement over 3-pert baseline:
- Score: +0.0007 (marginal)
- Time: -4 min faster (more buffer)
- Reliability: 100% in budget

### Initial vs Validated
| Metric | Initial Run | Validated Mean |
|--------|-------------|----------------|
| Score | 1.1524 | 1.1482 |
| Time | 58.3 min | 51.7 min |

The initial run (1.1524) was on the high end of the distribution.

## Comparison to 3-pert

| Config | Mean Score | Mean Time | Improvement |
|--------|------------|-----------|-------------|
| 3-pert (validated) | 1.1475 | 55.7 min | baseline |
| **4pert_nm2 (validated)** | **1.1482** | **51.7 min** | **+0.0007** |

The 4-pert_nm2 config:
- Has slightly higher score (+0.0007)
- Is faster (-4 min)
- Has similar variance

## Configuration

```python
PRODUCTION_CONFIG_4PERT = {
    'sigma0_1src': 0.18,
    'sigma0_2src': 0.22,
    'max_fevals_1src': 20,
    'max_fevals_2src': 44,
    'timestep_fraction': 0.40,
    'refine_maxiter': 8,
    'enable_tabu_hopping': True,
    'n_perturbations': 4,             # 4 perturbations
    'perturb_nm_iters': 2,            # KEY: Reduced from 3 to 2
    'perturbation_scale': 0.05,
    'tabu_distance': 0.04,
    'max_tabu_attempts': 10,
}

# Expected: 1.1482 +/- 0.0030 @ 51.7 min
```

## Recommendation

**ADOPT 4pert_nm2 as production** because:
1. Score is equal or slightly better than 3-pert
2. Timing is better (more safety margin)
3. 100% budget compliance

However, the improvement is marginal. Either config is acceptable:
- 4pert_nm2: 1.1482 @ 51.7 min (slightly better, more margin)
- 3pert: 1.1475 @ 55.7 min (well-validated, reliable)

## Leaderboard Context

| Rank | Team | Score | Gap |
|------|------|-------|-----|
| 10 | MGöksu | 1.1585 | +0.0103 |
| **Us (4pert_nm2)** | validated | **1.1482** | -- |
| Us (3-pert) | validated | 1.1475 | -0.0007 |
| 11 | bobatea | 1.1295 | -0.0187 |

## Tuning Efficiency

- **Runs executed**: 3
- **Time utilization**: 86% (51.7/60 min)
- **Budget margin**: 8.3 min
- **Variance**: ±0.0030

---
**Worker**: W1
**Completed**: 2026-02-02
**Runs**: 3
**Result**: VALIDATED SUCCESS - 1.1482 @ 51.7 min (marginal improvement)

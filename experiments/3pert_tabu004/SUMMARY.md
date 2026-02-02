# Experiment: 3pert_tabu004

## Status: MARGINALLY OVER BUDGET (mean 60.8 min)

## Hypothesis
Test 3 perturbations + tabu_distance=0.04 as a compromise between the 4-pert config (over budget) and 2-pert config (in budget).

## Results (3 Runs)

| Run | Score | Time (min) | RMSE 1src | RMSE 2src | Budget Status |
|-----|-------|------------|-----------|-----------|---------------|
| 1   | 1.1471 | 56.6 | 0.1197 | 0.1828 | **IN BUDGET** |
| 2   | 1.1434 | 61.1 | 0.1149 | 0.1985 | OVER BUDGET |
| 3   | 1.1524 | 64.9 | 0.1133 | 0.1868 | OVER BUDGET |

## Statistics

| Metric | Value |
|--------|-------|
| Mean Score | 1.1476 +/- 0.0037 |
| Mean Time | 60.8 min |
| vs Baseline (1.1337) | +0.0139 |
| vs 4pert+tabu04 (1.1535) | -0.0059 |
| Runs in budget | 1/3 (33%) |

## Analysis

3 perturbations is still marginally over budget on average. The configuration options are:

| Config | Mean Score | Mean Time | Budget Status | Runs In Budget |
|--------|------------|-----------|---------------|----------------|
| 4 pert + tabu 0.04 | 1.1535 | 61.3 min | OVER | 1/3 |
| 3 pert + tabu 0.04 | 1.1476 | 60.8 min | MARGINAL | 1/3 |
| **2 pert + tabu 0.04** | **1.1496** | **55.4 min** | **IN BUDGET** | **3/3** |

## Recommendation

**USE 2 PERTURBATIONS + TABU_DISTANCE=0.04 FOR PRODUCTION**

This config provides:
- Higher score than 3-pert (1.1496 vs 1.1476)
- More reliable timing (55.4 min with 4.6 min buffer)
- 100% of runs in budget

The counter-intuitive result (2 pert > 3 pert in score) is due to variance - but the timing reliability makes 2 pert the better choice.

## Final Production Configuration

```python
config = {
    'sigma0_1src': 0.18,
    'sigma0_2src': 0.22,
    'max_fevals_1src': 20,
    'max_fevals_2src': 44,
    'timestep_fraction': 0.40,
    'refine_maxiter': 8,
    'enable_tabu_hopping': True,
    'n_perturbations': 2,           # KEEP AT 2
    'perturb_nm_iters': 3,
    'perturbation_scale': 0.05,
    'tabu_distance': 0.04,          # IMPROVED
    'max_tabu_attempts': 10,
}
```

**Expected**: 1.1496 +/- 0.0046 @ 55.4 min

---
**Worker**: W1
**Completed**: 2026-02-02
**Runs**: 3
**Result**: MARGINALLY OVER BUDGET - not recommended

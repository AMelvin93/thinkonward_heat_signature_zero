# Experiment: 4pert_nm2_scale06_validation

## Status: MARGINAL - Equivalent to scale=0.05

## Prior Finding
Initial tuning: 1.1563 @ 52.2 min with scale=0.06 (+0.0081 vs scale=0.05)

## Validation Results (3 Runs)

| Run | Score | Time (min) | RMSE 1src | RMSE 2src | Budget |
|-----|-------|------------|-----------|-----------|--------|
| 1   | 1.1549 | 52.5 | 0.1198 | 0.1815 | **IN** |
| 2   | 1.1428 | 54.7 | 0.1207 | 0.2133 | **IN** |
| 3   | 1.1485 | 54.2 | 0.1145 | 0.1917 | **IN** |

## Statistics

| Metric | Value |
|--------|-------|
| **Mean Score** | **1.1487 +/- 0.0049** |
| Score Range | [1.1428, 1.1549] |
| **Mean Time** | **53.8 min** |
| **vs 4pert_nm2 scale=0.05** | **+0.0005** |
| Runs in budget | **3/3 (100%)** |
| **Gap to Top 10** | +0.0098 |

## Key Finding: ESSENTIALLY EQUIVALENT

The scale=0.06 config is essentially equivalent to scale=0.05:

| Config | Mean Score | Std | Mean Time |
|--------|------------|-----|-----------|
| scale=0.05 | 1.1482 | 0.0030 | 51.7 min |
| **scale=0.06** | **1.1487** | **0.0049** | **53.8 min** |

Differences:
- Score: +0.0005 (within noise)
- Variance: Higher (0.0049 vs 0.0030)
- Time: Slower (53.8 vs 51.7 min)

## Initial vs Validated

| Metric | Initial Run | Validated Mean |
|--------|-------------|----------------|
| Score | 1.1563 | 1.1487 |
| Improvement | +0.0081 | +0.0005 |

The initial result was an outlier on the high end.

## Recommendation

**Keep scale=0.05** as production default because:
1. Lower variance (more predictable)
2. Faster timing (more buffer)
3. Score improvement is within noise

Both configs are acceptable. Scale=0.05 is slightly safer.

## Configuration

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
    'perturbation_scale': 0.05,       # Keep at 0.05 for lower variance
    'tabu_distance': 0.04,
    'max_tabu_attempts': 10,
}

# Expected: 1.1482 +/- 0.0030 @ 51.7 min
# Alternative: scale=0.06 gives 1.1487 +/- 0.0049 @ 53.8 min
```

## Tuning Efficiency

- **Runs executed**: 3
- **Conclusion**: scale=0.06 ≈ scale=0.05 (no significant improvement)
- **Scale tuning EXHAUSTED for 4pert_nm2**

---
**Worker**: W1
**Completed**: 2026-02-02
**Runs**: 3
**Result**: MARGINAL - scale=0.06 equivalent to scale=0.05

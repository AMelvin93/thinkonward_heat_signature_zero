# W2 Session Summary (Continued) - 2026-02-02

## Experiments Run This Session

### 1. Sequential 2-Source Estimation
**Status**: FAILED (3 runs)

Hypothesis: Use heat equation linearity to decompose 4D optimization into 2D + 2D.

| Run | Config | Score | Time | vs Baseline |
|-----|--------|-------|------|-------------|
| 1 | fevals 15/22 | 1.0214 | 74.2 min | -0.1268 OVER |
| 2 | fevals 20/44 | 1.0401 | 109.0 min | -0.1081 MASSIVELY OVER |
| 3 | fevals 10/15 | 1.0157 | 42.5 min | -0.1325 IN but terrible |

**Conclusion**: Sequential decomposition is fundamentally flawed. Joint 4D CMA-ES is superior.

### 2. 5 Perturbations Test
**Status**: FAILED - Cannot fit budget (3 runs)

| Run | Config | Score | Time | vs Baseline |
|-----|--------|-------|------|-------------|
| 1 | 5pert_nm2 | 1.1534 | 76.0 min | +0.0052 OVER |
| 2 | 5pert_nm1 | 1.1533 | 76.1 min | +0.0051 OVER |
| 3 | reduced_fevals | 1.1567 | 80.1 min | +0.0085 OVER |

**Key Finding**: 5 perturbations improves score (+0.005-0.008) but adds ~24+ min overhead regardless of other settings. Cannot fit in budget.

## Total Runs: 6 (systematic tuning)

## Current Best Configuration

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
# Expected: 1.1482 ± 0.0030 @ 51.7 min
```

## Leaderboard Status

| Rank | Team | Score | Gap |
|------|------|-------|-----|
| 10 | MGöksu | 1.1585 | +0.0103 |
| **Us** | **4pert_nm2** | **1.1482** | -- |
| 11 | bobatea | 1.1295 | -0.0187 |

## Exhausted Approaches (Summary)

| Approach | Status | Reason |
|----------|--------|--------|
| Sequential 2-source | FAILED | Compounding errors, worse accuracy |
| 5 perturbations | CANNOT FIT | +24 min overhead, over budget |
| Adaptive NM by source | FAILED | Fixed NM=8 is optimal |
| All tuning dimensions | EXHAUSTED | See exhausted families list |

## Remaining Options

1. **Validation runs**: Establish higher confidence in current config (priority 1 in queue)
2. **Accept current config**: 1.1482 may be near the limit of this approach
3. **Look for novel algorithmic changes**: All tested approaches have failed

## Gap Analysis

To reach Top 10 (1.1585):
- Need +0.0103 improvement
- Current 2-source RMSE ~0.19 (bottleneck)
- 5 perturbations would help (+0.005-0.008) but can't fit budget
- No in-budget improvement found

## Recommendations

1. **For submission**: Use 4pert_nm2 config (validated 1.1482 @ 51.7 min)
2. **For further improvement**: Novel algorithmic approaches needed
3. **Time budget**: 8.3 min remaining could be used but no effective use found

---
**Worker**: W2
**Date**: 2026-02-02
**Total Runs**: 6
**Status**: No improvement found this session

# W2 Session Summary - 2026-02-02

## Experiments Completed (7 total)

### 1. 4pert_tabu004_combined (resumed from W1)
**Status**: PARTIAL SUCCESS - Over budget | **3 runs**
- Tested: 4 perturbations + tabu_distance=0.04
- **Result**: Mean 1.1555 @ 61.9 min (OVER BUDGET by 1.9 min)
- **Key Finding**: Additive improvement confirmed but over budget

### 2. 3pert_tabu004_budget
**Status**: SUCCESS | **3 runs**
- Tested: 3 perturbations + tabu_distance=0.04
- **Result**: Mean 1.1478 @ 58.4 min (IN BUDGET)
- **Key Finding**: Optimal in-budget configuration

### 3. 4pert_reduced_fevals
**Status**: FAILED | **3 runs**
- Tested: 4 perturbations with reduced fevals (18/40)
- **Result**: Mean 1.1465 @ 66.7 min (OVER BUDGET + WORSE)
- **Key Finding**: Reducing fevals made timing WORSE (counterintuitive)

### 4. 4pert_reduced_polish
**Status**: FAILED | **3 runs**
- Tested: 4 perturbations with reduced NM polish (6 iterations)
- **Result**: Mean 1.1451 @ 57.9 min (IN BUDGET but WORSE than 3-pert)
- **Key Finding**: 8 NM polish iterations are crucial for accuracy

### 5. production_3pert_validation
**Status**: VALIDATED | **5 runs**
- Tested: Production config (3-pert + tabu 0.04) with 5 runs
- **Result**: Mean 1.1475 +/- 0.0031 @ 55.7 min (100% IN BUDGET)
- **Key Finding**: PRODUCTION READY - high confidence validated score

### 6. higher_fevals_3pert
**Status**: PROMISING BUT OVER BUDGET | **3 runs**
- Tested: 22/48 fevals with 3 perturbations
- **Result**: Mean 1.1568 @ 63.4 min (OVER BUDGET, but run 2 hit 1.1610!)
- **Key Finding**: Higher fevals significantly improves score but exceeds budget

### 7. intermediate_fevals_3pert
**Status**: INCONCLUSIVE | **3 runs**
- Tested: 21/46 fevals with 3 perturbations (between 20/44 and 22/48)
- **Result**: Mean 1.1466 @ 57.4 min (IN BUDGET but high variance ±0.0061)
- **Key Finding**: No reliable improvement, high variance

## Total Runs: 23

## Key Validated Findings

| Approach | Result | Why |
|----------|--------|-----|
| **3-pert + tabu 0.04** | **OPTIMAL** | Best balance of score and time |
| 4-pert + tabu 0.04 | OVER BUDGET | +0.0077 score but +3.5 min |
| Reduce fevals to fit 4-pert | FAILED | Made timing worse, not better |
| Reduce NM polish to fit 4-pert | FAILED | In budget but worse score |

## Optimal In-Budget Configuration (VALIDATED)

```python
config = {
    'sigma0_1src': 0.18,
    'sigma0_2src': 0.22,
    'max_fevals_1src': 20,
    'max_fevals_2src': 44,
    'timestep_fraction': 0.40,
    'refine_maxiter': 8,              # CRITICAL: Don't reduce
    'enable_tabu_hopping': True,
    'n_perturbations': 3,             # OPTIMAL: 3 (not 4)
    'perturb_nm_iters': 3,
    'perturbation_scale': 0.05,
    'tabu_distance': 0.04,            # VALIDATED improvement
    'max_tabu_attempts': 10,
}
```

**VALIDATED Score**: 1.1475 +/- 0.0031 @ 55.7 min (5 runs, 100% in budget)

## Leaderboard Analysis

| Rank | Team | Score | Gap from Us |
|------|------|-------|-------------|
| 10 | MGöksu | 1.1585 | +0.0110 |
| **Us** | **VALIDATED** | **1.1475** | -- |
| 11 | bobatea | 1.1295 | -0.0180 |

**Gap to Top 10**: +0.0110 (approximately 1.0%)

## Conclusions

### 1. 4 Perturbations Cannot Fit Budget
Multiple approaches to fit 4 perturbations within budget all failed:
- Reducing fevals: Made things SLOWER (counterintuitive)
- Reducing NM polish: Lower score than 3-pert

### 2. Current Parameters Are Near-Optimal
Both CMA-ES fevals and NM polish iterations are efficiently spent:
- Reducing either hurts accuracy significantly
- The 8 NM iterations provide meaningful refinement

### 3. Variance Remains Significant
Run-to-run variance of ~0.003-0.007 means:
- Single-run conclusions are unreliable
- 3+ runs needed for validation
- Some "improvements" may be noise

## What Would Bridge the Gap to Top 10

The gap to top 10 is +0.0107. Options to explore:
1. **Novel algorithm approaches**: Current CMA-ES is near-optimal
2. **Better 2-source handling**: 2-source RMSE (~0.19) is the bottleneck
3. **Initialization improvements**: Could reduce exploration budget
4. **Accept risk with 4-pert**: If timing variance is favorable

## Session Statistics

- **Total Experiments**: 7
- **Total Tuning Runs**: 23
- **Session Duration**: ~9 hours
- **Key Finding**: 3-pert + 20/44 fevals + tabu 0.04 is VALIDATED optimal in-budget

## Fevals Analysis

| Config | Score | Variance | Time | Budget |
|--------|-------|----------|------|--------|
| 20/44 | **1.1475** | **±0.0031** | 55.7 | **IN** |
| 21/46 | 1.1466 | ±0.0061 | 57.4 | IN (high variance) |
| 22/48 | 1.1568 | ±0.0041 | 63.4 | OVER |

**Conclusion**: 20/44 fevals is optimal - higher fevals adds variance and timing risk

## Final Validated Configuration (PRODUCTION READY)

```python
# VALIDATED: 1.1475 +/- 0.0031 @ 55.7 min (5 runs, 100% in budget)
config = {
    'sigma0_1src': 0.18,
    'sigma0_2src': 0.22,
    'max_fevals_1src': 20,
    'max_fevals_2src': 44,
    'timestep_fraction': 0.40,
    'refine_maxiter': 8,
    'enable_tabu_hopping': True,
    'n_perturbations': 3,
    'perturb_nm_iters': 3,
    'perturbation_scale': 0.05,
    'tabu_distance': 0.04,
    'max_tabu_attempts': 10,
}
```

---
**Worker**: W2
**Date**: 2026-02-02
**Last Updated**: ~14:00 UTC

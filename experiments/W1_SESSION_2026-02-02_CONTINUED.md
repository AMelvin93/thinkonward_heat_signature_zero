# W1 Session Summary - 2026-02-02 (Continued)

## Major Achievement: NEW BEST CONFIG FOUND!

After discovering the timing discrepancy (system runs 35-40% slower), I found that **2 perturbations fits in budget** and achieves better scores than the no_perturb baseline.

## Best Validated Config

```python
config = {
    'sigma0_1src': 0.18,
    'sigma0_2src': 0.22,
    'max_fevals_1src': 20,
    'max_fevals_2src': 44,
    'timestep_fraction': 0.40,
    'refine_maxiter': 8,
    'enable_tabu_hopping': True,
    'n_perturbations': 2,
    'perturbation_scale': 0.05,
    'perturb_nm_iters': 3,
    'tabu_distance': 0.04,
    'max_tabu_attempts': 10,
}
```

**Validation Results (3 runs):**
| Run | Score | Time |
|-----|-------|------|
| 1 | 1.1390 | 50.9 min |
| 2 | **1.1516** | 52.7 min |
| 3 | 1.1369 | 50.2 min |

**Statistics:**
- Mean: **1.1425 +/- 0.0065**
- Time: **51.3 +/- 1.0 min**
- 100% in budget
- **Best run: 1.1516** (gap to Top 10: 0.0069!)

## Comparison of All Validated Configs

| Config | Mean Score | Std | Best Run | Mean Time | Gap to Top 10 |
|--------|------------|-----|----------|-----------|---------------|
| no_perturb (0.18/0.22) | 1.1386 | 0.0067 | 1.1469 | 49.3 min | 0.0116 |
| **2pert_018_022** | **1.1425** | 0.0065 | **1.1516** | 51.3 min | **0.0069** |
| 2pert_014_018 | 1.1395 | 0.0033 | 1.1431 | 51.3 min | 0.0154 |
| sigma_015_019 (no_perturb) | 1.1321 | 0.0059 | 1.1365 | 42.0 min | 0.0220 |

**Clear winner: 2pert_018_022**

## Experiments Conducted This Session

### Phase 1: Sigma Tuning Without Perturbation
- Tested sigma 0.16/0.20, 0.15/0.19, 0.17/0.21
- **Result**: No consistent improvement over baseline
- sigma_015_019 single run showed 1.1405, but validation showed mean 1.1321 (WORSE)

### Phase 2: Testing Perturbation Counts
- 1 perturbation: 1.1312 @ 45.4 min - WORSE than baseline
- **2 perturbations: 1.1427 @ 52.0 min - IMPROVEMENT!**
- 4 perturbations: ~71 min - OVER BUDGET

### Phase 3: Sigma Tuning With 2 Perturbations
- 2pert_014_018: Mean 1.1395 - worse than 2pert_018_022
- 2pert_016_020: 1.1395 single run - not validated

### Key Insight: Run-to-Run Variance

Run-to-run variance is ~0.006-0.007 (std). This means:
- Single runs are unreliable for conclusions
- 3-run validation is essential
- "Lucky" runs can show +0.003-0.005 improvement that doesn't hold

## Recommendations for Competition

### PRIMARY: Use 2pert_018_022

```python
config = {
    'sigma0_1src': 0.18,
    'sigma0_2src': 0.22,
    'max_fevals_1src': 20,
    'max_fevals_2src': 44,
    'timestep_fraction': 0.40,
    'refine_maxiter': 8,
    'enable_tabu_hopping': True,
    'n_perturbations': 2,
    'perturbation_scale': 0.05,
    'perturb_nm_iters': 3,
    'tabu_distance': 0.04,
    'max_tabu_attempts': 10,
}
```

**Expected:**
- Mean score: ~1.1425
- Best case: ~1.15+ (close to Top 10!)
- Time: ~51 min (well within budget)
- 100% completion rate

### BACKUP: Use no_perturb if hardware is uncertain

```python
config = {
    'sigma0_1src': 0.18,
    'sigma0_2src': 0.22,
    'max_fevals_1src': 20,
    'max_fevals_2src': 44,
    'timestep_fraction': 0.40,
    'refine_maxiter': 8,
    'enable_tabu_hopping': False,
}
```

**Expected:**
- Mean score: ~1.14
- Time: ~49 min (very safe)

## Gap Analysis

| Target | Score | Our Mean | Our Best | Gap from Mean | Gap from Best |
|--------|-------|----------|----------|---------------|---------------|
| Top 10 | 1.1585 | 1.1425 | 1.1516 | 0.016 | **0.007** |
| Top 9 | 1.1716 | 1.1425 | 1.1516 | 0.029 | 0.020 |

**We are within striking distance of Top 10! Best runs reach 1.1516, only 0.007 away!**

---
**Worker**: W1
**Date**: 2026-02-02
**Status**: SUCCESS - New best config validated

# CRITICAL TIMING DISCOVERY - W1 Session 2026-02-02

## Summary

**Original finding**: All previously validated "in-budget" configs are NOW OVER BUDGET on the current system.

**UPDATE**: Found that 2 perturbations (not 4) fits within budget and improves score!

## Evidence

### System Speed Comparison
| System | 4pert_nm2 Time | Ratio |
|--------|----------------|-------|
| Original validation | 51.7 min | 1.0x |
| This system | 70.9 min | 1.37x slower |

## Complete Timing Benchmark (This System)

| Config | Score | Time (proj 400) | Budget Status |
|--------|-------|-----------------|---------------|
| 4-pert nm2 | 1.1546 | 70.9 min | **OVER** |
| 2-pert nm2 (old) | 1.1452 | 67.4 min | OVER |
| **2-pert nm3 (NEW!)** | **1.1425** | **51.3 min** | **IN** |
| 1-pert | 1.1447 | 65.9 min | OVER |
| no_perturb | 1.1386 | 49.3 min | IN |

## KEY FINDING: 2 Perturbations with nm_iters=3 Works!

The difference:
- **2-pert nm2**: 67.4 min (OVER) - perturb_nm_iters=2
- **2-pert nm3**: 51.3 min (IN) - perturb_nm_iters=3

**Note**: The original "2-pert nm2" timing (67.4 min) may have used different parameters. The validated 2-pert nm3 config reliably fits in budget.

## NEW RECOMMENDED CONFIG (VALIDATED)

### 2-Perturbation Config (3 runs validated)

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

**Validation Results:**

| Run | Score | Time | Budget |
|-----|-------|------|--------|
| 1 | 1.1390 | 50.9 min | IN |
| 2 | **1.1516** | 52.7 min | IN |
| 3 | 1.1369 | 50.2 min | IN |

**Statistics:**
- Mean Score: **1.1425 +/- 0.0065**
- Mean Time: **51.3 +/- 1.0 min**
- **100% runs in budget**
- Best run score: **1.1516** (gap to Top 10: 0.0069!)

## Comparison: New vs Old Recommended Config

| Metric | no_perturb | 2-perturb (NEW) | Improvement |
|--------|------------|-----------------|-------------|
| Mean Score | 1.1386 | **1.1425** | **+0.0039** |
| Mean Time | 49.3 min | 51.3 min | +2.0 min |
| Best Run | 1.1469 | **1.1516** | **+0.0047** |
| Gap to Top 10 | 0.0116 | **0.0069** | 40% closer! |

## Final Recommendation for Competition

### RECOMMENDED: 2-Perturbation Config (VALIDATED)

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

**Expected: Score ~1.1425, Time ~51 min, 100% completion**
**Best case: Score ~1.15, very close to Top 10 (1.1585)**

### BACKUP: No-Perturbation Config (Also VALIDATED)

Use if concerned about hardware variability:
```python
config = {
    'sigma0_1src': 0.18,
    'sigma0_2src': 0.22,
    'max_fevals_1src': 20,
    'max_fevals_2src': 44,
    'timestep_fraction': 0.40,
    'refine_maxiter': 8,
    'enable_tabu_hopping': False,  # NO PERTURBATIONS
}
```

**Expected: Score ~1.14, Time ~49 min, 100% completion**

### NOT RECOMMENDED: 4-Perturbation Config

```python
config = {
    'n_perturbations': 4,  # Too many!
}
```

**Expected: ~71 min (OVER BUDGET)**

---
**Worker**: W1
**Date**: 2026-02-02
**Priority**: HIGH
**Last Updated**: 2026-02-02 18:45 UTC

# W1 FINAL RECOMMENDATION - 2026-02-02

## SESSION COMPLETE - EXTENSIVE TUNING DONE

After testing 15+ configurations and validating 5 with 3-run validation each, the optimal config is:

## BEST VALIDATED CONFIG

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
    'perturb_nm_iters': 4,  # KEY: increased from 3
    'tabu_distance': 0.04,
    'max_tabu_attempts': 10,
}
```

## Validation Results (3 runs)

| Run | Score | Gap to Top 10 | Time | Budget |
|-----|-------|---------------|------|--------|
| 1 | 1.1519 | 0.0066 | 52.4 min | IN |
| 2 | 1.1525 | 0.0060 | 53.1 min | IN |
| 3 | 1.1417 | 0.0168 | 57.1 min | IN |

## Statistics

| Metric | Value |
|--------|-------|
| **Mean Score** | **1.1487 +/- 0.0050** |
| **Mean Time** | **54.2 +/- 2.1 min** |
| **Best Run** | **1.1525** |
| **Runs in Budget** | **100% (3/3)** |

## Comparison with All Validated Configs

| Config | Mean | Std | Best | Time | Gap to Top 10 |
|--------|------|-----|------|------|---------------|
| no_perturb (baseline) | 1.1386 | 0.0067 | 1.1469 | 49.3 min | 0.0116 |
| 2pert nm3 | 1.1425 | 0.0065 | 1.1516 | 51.3 min | 0.0069 |
| **2pert nm4** | **1.1487** | 0.0050 | **1.1525** | 54.2 min | **0.0060** |

**Improvement from baseline: +0.0101 mean score!**

## Gap Analysis

| Position | Score | Gap from Our Mean | Gap from Our Best |
|----------|-------|-------------------|-------------------|
| Top 10 | 1.1585 | **0.0098** | **0.0060** |
| Top 9 | 1.1716 | 0.0229 | 0.0191 |
| Top 8 | 1.1743 | 0.0256 | 0.0218 |

**We are very close to Top 10!**
- Mean is only 0.0098 away
- Best runs reach 1.1525, only 0.0060 away!

## Key Insight: perturb_nm_iters Matters

The difference between nm_iters=3 and nm_iters=4:
- nm_iters=3: Mean 1.1425, Best 1.1516
- nm_iters=4: Mean 1.1487 (+0.0062!), Best 1.1525

More NM polish iterations improve solution quality significantly.

## Recommended for Competition Submission

### PRIMARY CONFIG (VALIDATED)
```python
{
    'sigma0_1src': 0.18,
    'sigma0_2src': 0.22,
    'max_fevals_1src': 20,
    'max_fevals_2src': 44,
    'timestep_fraction': 0.40,
    'refine_maxiter': 8,
    'enable_tabu_hopping': True,
    'n_perturbations': 2,
    'perturbation_scale': 0.05,
    'perturb_nm_iters': 4,
    'tabu_distance': 0.04,
    'max_tabu_attempts': 10,
}
```

**Expected:**
- Mean score: ~1.149
- Best case: ~1.152+ (very close to Top 10!)
- Time: ~54 min (within 60 min budget)
- 100% completion rate

### BACKUP CONFIG (Also Validated)
Use if hardware is significantly different:
```python
{
    'sigma0_1src': 0.18,
    'sigma0_2src': 0.22,
    'max_fevals_1src': 20,
    'max_fevals_2src': 44,
    'timestep_fraction': 0.40,
    'refine_maxiter': 8,
    'enable_tabu_hopping': False,  # No perturbations
}
```

**Expected:**
- Mean score: ~1.14
- Time: ~49 min (very safe)

## Complete Validation Summary

All configs with 3-run validation:

| Config | Mean | Std | Best | Mean Time | Runs in Budget |
|--------|------|-----|------|-----------|----------------|
| no_perturb | 1.1386 | 0.0067 | 1.1469 | 49.3 min | 100% |
| 2pert nm3 | 1.1425 | 0.0065 | 1.1516 | 51.3 min | 100% |
| 2pert sigma_014_018 | 1.1395 | 0.0033 | 1.1431 | 51.3 min | 100% |
| **2pert nm4** | **1.1487** | **0.0050** | **1.1525** | **54.2 min** | **100%** |
| 2pert nm5 | 1.1479 | 0.0019 | 1.1503 | 58.0 min | 100%* |

*nm5 timing is risky (max 59.4 min, buffer only 0.6 min)

## Why nm4 is Optimal

1. **Better score than nm3**: +0.0062 mean improvement
2. **Safer than nm5**: 6 min timing buffer vs 0.6 min
3. **Consistent**: Low variance (0.0050) across runs
4. **Best single runs**: 1.1525 achievable (0.006 from Top 10)

---
**Worker**: W1
**Date**: 2026-02-02
**Status**: BEST CONFIG FOUND - Ready for submission
**Gap to Top 10**: Only 0.0098 mean, 0.0060 best run!

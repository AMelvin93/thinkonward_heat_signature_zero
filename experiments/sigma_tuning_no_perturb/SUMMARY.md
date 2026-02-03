# Experiment: sigma_tuning_no_perturb

## Status: SUCCESS - BEST CONFIG FOUND!

## Executive Summary

Through systematic experimentation, found that **2 perturbations with perturb_nm_iters=4** achieves the best validated score while fitting within budget.

## FINAL BEST CONFIG

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
    'perturb_nm_iters': 4,  # KEY PARAMETER
    'tabu_distance': 0.04,
    'max_tabu_attempts': 10,
}
```

## Best Validation Results (nm4_scale05, 3 runs)

| Run | Score | Gap to Top 10 | Time | Budget |
|-----|-------|---------------|------|--------|
| 1 | 1.1519 | 0.0066 | 52.4 min | IN |
| 2 | **1.1525** | **0.0060** | 53.1 min | IN |
| 3 | 1.1417 | 0.0168 | 57.1 min | IN |

### Statistics
- **Mean Score: 1.1487 +/- 0.0050**
- **Mean Time: 54.2 +/- 2.1 min**
- **100% runs in budget**
- **Best run: 1.1525** (only 0.0060 from Top 10!)

## All Validated Configs Comparison

| Config | Mean | Std | Best | Time | Gap to Top 10 |
|--------|------|-----|------|------|---------------|
| no_perturb (baseline) | 1.1386 | 0.0067 | 1.1469 | 49.3 min | 0.0116 |
| 2pert nm3 | 1.1425 | 0.0065 | 1.1516 | 51.3 min | 0.0069 |
| 2pert sigma_014_018 | 1.1395 | 0.0033 | 1.1431 | 51.3 min | 0.0154 |
| **2pert nm4** | **1.1487** | 0.0050 | **1.1525** | 54.2 min | **0.0060** |

**Total improvement from baseline: +0.0101 mean score!**

## Experiments Conducted

### Phase 1: Sigma Tuning (no perturbation)
| Config | Score | Time | Delta vs baseline |
|--------|-------|------|-------------------|
| sigma_016_020 | 1.1379 | 42.7 min | -0.0007 |
| sigma_015_019 | 1.1405 | 43.3 min | +0.0019 |
| sigma_017_021 | 1.1315 | 43.8 min | -0.0071 |

**Finding**: Sigma tuning without perturbation shows no consistent improvement.

### Phase 2: Sigma 0.15/0.19 Validation (3 runs)
| Run | Score | Time |
|-----|-------|------|
| 1 | 1.1365 | 42.9 min |
| 2 | 1.1360 | 40.5 min |
| 3 | 1.1238 | 42.6 min |

**Mean: 1.1321 +/- 0.0059 @ 42.0 min** - WORSE than baseline

### Phase 3: Testing 1 Perturbation
| Config | Score | Time |
|--------|-------|------|
| sigma_015_019_1perturb | 1.1312 | 45.4 min |

**Finding**: 1 perturbation not sufficient, worse than no_perturb baseline.

### Phase 4: Testing 2 Perturbations
| Config | Score | Time | Status |
|--------|-------|------|--------|
| **2perturb_018_022** | **1.1427** | 52.0 min | **IN** |
| 2perturb_015_019 | 1.1394 | 52.0 min | IN |

### Phase 5: Validation of 2perturb_018_022 (3 runs)
| Run | Score | Time |
|-----|-------|------|
| 1 | 1.1390 | 50.9 min |
| 2 | 1.1516 | 52.7 min |
| 3 | 1.1369 | 50.2 min |

**Mean: 1.1425 +/- 0.0065 @ 51.3 min**

### Phase 6: Sigma Tuning with 2 Perturbations
| Config | Score | Time | Status |
|--------|-------|------|--------|
| 2pert_sigma_014_018 | 1.1454 | 50.1 min | IN |
| 2pert_sigma_016_020 | 1.1395 | 50.2 min | IN |

**Validation of 2pert_sigma_014_018 showed mean 1.1395** - not better.

### Phase 7: Perturbation Parameters Tuning
| Config | Score | Time | Status |
|--------|-------|------|--------|
| **nm4_scale05** | **1.1568** | 52.1 min | IN |
| nm3_scale06 | 1.1494 | 51.5 min | IN |

### Phase 8: Validation of nm4_scale05 (3 runs) - FINAL BEST
| Run | Score | Gap to Top 10 | Time |
|-----|-------|---------------|------|
| 1 | 1.1519 | 0.0066 | 52.4 min |
| 2 | 1.1525 | 0.0060 | 53.1 min |
| 3 | 1.1417 | 0.0168 | 57.1 min |

**Mean: 1.1487 +/- 0.0050 @ 54.2 min** - NEW BEST!

## Key Insights

1. **Perturbations are essential** - Without them, sigma tuning cannot improve the baseline.

2. **2 perturbations is the sweet spot** for this system timing.

3. **perturb_nm_iters=4 is better than 3** - More polish iterations improve quality (+0.0062 mean).

4. **Timing on this system**: ~54 min with nm4, well within 60 min budget.

## Gap Analysis

| Target | Score | Gap from Our Mean | Gap from Our Best |
|--------|-------|-------------------|-------------------|
| Top 10 | 1.1585 | **0.0098** | **0.0060** |
| Top 9 | 1.1716 | 0.0229 | 0.0191 |
| Top 8 | 1.1743 | 0.0256 | 0.0218 |

**We are very close to Top 10!**

## Recommendation for Competition

### PRIMARY: Use nm4_scale05 config

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
- Best case: ~1.152+ (only 0.006 from Top 10!)
- Time: ~54 min (within budget)
- 100% completion rate

---
**Worker**: W1
**Date**: 2026-02-02
**Status**: SUCCESS - BEST CONFIG FOUND

# Larger Perturbation Pool Experiment

## Hypothesis
Test more perturbations (3-4) on the new best config. More exploration without diversity loss.

## Result: NEW BEST FOUND!

**New best score**: 1.1437 ± 0.0010 @ 42.2 min
**Improvement vs 2-perturb baseline**: +0.28%

## Tuning Results

| Config | Score | Avg Cands | Time | vs Baseline |
|--------|-------|-----------|------|-------------|
| 3 perturbations | 1.1378 | 2.80 | 38.7 min | -0.0027 |
| **4 perturbations** | **1.1436** | **2.86** | **43.3 min** | **+0.0031** |
| 2 perturbations | 1.1398 | 2.83 | 43.9 min | baseline |

## 4-Perturbation Validation (3 runs)

| Run | Score | Time | vs Baseline |
|-----|-------|------|-------------|
| 1 | 1.1436 | 43.3 min | +0.0031 |
| 2 | 1.1449 | 42.9 min | +0.0044 |
| 3 | 1.1426 | 40.5 min | +0.0021 |
| **Mean** | **1.1437** | **42.2 min** | **+0.0032** |

**Std Dev**: 0.0010 (very low - highly reproducible!)

## Leaderboard Update

| Rank | Config | Mean Score | Std | Time |
|------|--------|------------|-----|------|
| 1 | **0.18/0.22 + 20/44 + 4perturb** | **1.1437** | 0.0010 | 42.2 min |
| 2 | 0.18/0.22 + 20/44 + 2perturb | 1.1405 | 0.0058 | 42.3 min |
| 3 | 0.15/0.19 + 20/36 + 2perturb | 1.1337 | 0.0027 | 44.0 min |

## Why 4 Perturbations Works

1. **More exploration**: 4 perturbations explore more local optima around CMA-ES solution
2. **No diversity loss**: Avg candidates remained high (2.84-2.86)
3. **Similar runtime**: Only 0.1 min slower than baseline

## Recommendation

**PROMOTE TO PRODUCTION**

New best configuration:
```python
config = {
    'sigma0_1src': 0.18,
    'sigma0_2src': 0.22,
    'max_fevals_1src': 20,
    'max_fevals_2src': 44,
    'timestep_fraction': 0.40,
    'refine_maxiter': 8,
    'n_perturbations': 4,
    'perturbation_scale': 0.05,
    'perturb_nm_iters': 3,
}
```

---
**Worker**: W2
**Completed**: 2026-02-01
**Runs**: 6 (3 tuning + 3 validation)

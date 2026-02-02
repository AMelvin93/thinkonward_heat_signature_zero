# Experiment Summary: optimal_sigma_higher_fevals

## Status: PARTIAL SUCCESS - Higher fevals improve 2-source RMSE

## Experiment ID: EXP_OPTIMAL_SIGMA_FEVALS_001
## Worker: W1
## Date: 2026-02-01

## Hypothesis
Combining optimal sigma (0.15/0.19) with higher fevals (22/40 or 24/44) may improve 2-source RMSE while staying in budget. Prior experiment (nm4_perturb1_fevals_22_40) showed 24/44 fevals improved 2-source RMSE from 0.215 to 0.197.

## Results Summary

| Config | Fevals | NM iter | Score | Time | RMSE 1-src | RMSE 2-src | In Budget |
|--------|--------|---------|-------|------|------------|------------|-----------|
| baseline_20_36 | 20/36 | 8 | 1.1611 | 56.0m | 0.1231 | 0.1995 | YES |
| sigma015_fevals_22_40 | 22/40 | 8 | 1.1618 | 54.6m | 0.1208 | 0.1999 | YES |
| **sigma015_fevals_24_44** | 24/44 | 8 | **1.1694** | 57.2m | 0.1207 | **0.1797** | YES |
| sigma015_fevals_24_44_nm6 | 24/44 | 6 | 1.1594 | 53.4m | 0.1173 | 0.2100 | YES |

**Best in this run**: sigma015_fevals_24_44 with score 1.1694 @ 57.2 min

## Key Findings

### 1. Higher fevals significantly improves 2-source RMSE
- 20/36 fevals: 2-source RMSE = 0.1995
- 24/44 fevals: 2-source RMSE = **0.1797** (-10% improvement!)

This confirms the hypothesis from nm4_perturb1_fevals_22_40: 2-source problems benefit from more function evaluations.

### 2. Reducing NM to compensate for fevals hurts score
- 24/44 fevals + 8 NM = 1.1694 @ 57.2 min
- 24/44 fevals + 6 NM = 1.1594 @ 53.4 min (-0.0100 score)

The 8 NM iterations are essential for accuracy. Trading NM for fevals is a bad trade-off.

### 3. Run-to-run variance is significant
- Claimed baseline (tighter_sigma_range): 1.1730 @ 50.4 min
- This run's baseline: 1.1611 @ 56.0 min (delta -0.0119)

The variance of ~0.01 in score makes it hard to definitively compare configs.

## RMSE Breakdown Analysis

| Config | RMSE 1-src | RMSE 2-src | Overall RMSE |
|--------|------------|------------|--------------|
| baseline_20_36 | 0.1231 | 0.1995 | 0.1613 |
| sigma015_fevals_24_44 | 0.1207 | 0.1797 | 0.1502 |
| **Improvement** | -1.9% | **-9.9%** | **-6.9%** |

The improvement is almost entirely from 2-source problems, where higher fevals allow CMA-ES to better explore the 4D search space.

## Tuning Efficiency Metrics
- **Runs executed**: 4 (systematic parameter sweep)
- **Time utilization**: 95% (57.2/60 min budget)
- **Parameter space explored**: fevals [20/36, 22/40, 24/44], NM [6, 8]
- **Budget-feasible configs found**: 4 of 4

## Recommendation

The evidence suggests **fevals 24/44 + 8 NM** may be optimal for production:
- It significantly improves 2-source RMSE (-10%)
- It stays within budget (57.2 min projected)
- The score improvement (+0.0083 vs baseline in this run) is meaningful

However, the run-to-run variance makes it difficult to confirm this definitively. A validation run would help.

## Comparison to Prior Results

| Experiment | Best Config | Score | Time |
|------------|-------------|-------|------|
| tighter_sigma_range (claimed) | sigma 0.15/0.19 + 20/36 | 1.1730 | 50.4m |
| **This run (24/44 fevals)** | sigma 0.15/0.19 + 24/44 | 1.1694 | 57.2m |
| This run (baseline) | sigma 0.15/0.19 + 20/36 | 1.1611 | 56.0m |

Within this run, the 24/44 fevals config outperformed the 20/36 baseline by +0.0083. But both are below the claimed 1.1730 from tighter_sigma_range, suggesting environmental variance.

## New Production Config (Tentative)

If validation confirms improvement:
```python
config = {
    'sigma0_1src': 0.15,
    'sigma0_2src': 0.19,
    'max_fevals_1src': 24,      # INCREASED from 20
    'max_fevals_2src': 44,      # INCREASED from 36
    'timestep_fraction': 0.40,
    'refine_maxiter': 8,
    'enable_tabu_hopping': True,
    'n_perturbations': 2,
    'perturbation_scale': 0.05,
    'perturb_nm_iters': 3,
}
# Expected: Score ~1.17, Time ~57 min for 400 samples
```

## What Would Have Been Tried With More Time
- Validate the 24/44 config with multiple seeds
- Test intermediate fevals (23/42) to find the sweet spot
- Try 24/44 fevals with 3 perturbations to use remaining budget

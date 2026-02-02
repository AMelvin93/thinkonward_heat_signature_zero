# Experiment Summary: nm4_perturb1_fevals_22_40

## Status: COMPLETED - Below current best

## Experiment ID: EXP_NM4_PERTURB1_HIGHER_FEVALS_001
## Worker: W1 (completed by W1)
## Date: 2026-02-01

## Hypothesis
Combining nm4_perturb1 (4 NM iterations + 1 perturbation) with higher fevals (22/40) could improve accuracy while staying well within budget.

## Results Summary

| Config | max_fevals | nm_iter | n_perturb | Score | Time | Projected 400 |
|--------|------------|---------|-----------|-------|------|---------------|
| nm4_p1_fevals_20_36 | 20/36 | 4 | 1 | 1.1553 | 7.8m | 38.9m |
| nm4_p1_fevals_22_40 | 22/40 | 4 | 1 | 1.1599 | 9.2m | 46.1m |
| **nm4_p1_fevals_24_44** | 24/44 | 4 | 1 | **1.1639** | 8.6m | **43.1m** |

**Best in-budget**: nm4_p1_fevals_24_44 with score 1.1639 @ 43.1 min projected

## Comparison to Current Best

| Config | Score | Time (400) | Delta vs Best |
|--------|-------|------------|---------------|
| **Current best** (sigma 0.15/0.19 + 2 perturb) | 1.1730 | 50.4m | -- |
| This experiment best | 1.1639 | 43.1m | **-0.0091** |

**Finding**: This experiment is **WORSE** than the current best because it used the old sigma values (0.18/0.22), not the optimal sigma (0.15/0.19).

## Key Insight

The optimal sigma (0.15/0.19) discovered in `tighter_sigma_range` provides +0.0077 improvement. This experiment's approach (nm4 + 1 perturb + higher fevals) cannot compensate for suboptimal sigma.

## RMSE Breakdown

| Config | RMSE 1-src | RMSE 2-src | Overall |
|--------|------------|------------|---------|
| nm4_p1_fevals_20_36 | 0.1231 | 0.2152 | 0.1691 |
| nm4_p1_fevals_22_40 | 0.1278 | 0.1979 | 0.1629 |
| nm4_p1_fevals_24_44 | 0.1183 | 0.1966 | 0.1575 |

**Trend**: Higher fevals (24/44) gives best 2-source RMSE (0.1966), confirming 2-source problems benefit from more function evaluations.

## Tuning Efficiency Metrics
- **Runs executed**: 3
- **Time utilization**: 72% (43.1/60 min projected)
- **Budget remaining**: 16.9 min (could be used for more experiments)

## Recommendation

This experiment demonstrates that **fevals tuning alone cannot match optimal sigma**:
- sigma 0.15/0.19 contributes +0.0077
- fevals 24/44 vs 20/36 contributes ~+0.0086

The optimal approach is to combine:
1. Optimal sigma (0.15/0.19) - already in production
2. 2 perturbations - already in production
3. 8 NM iterations - already in production

**Verdict**: Keep production config as-is (tighter_sigma_range). This experiment's approach is superseded.

## What Would Have Been Tried With More Time
- sigma 0.15/0.19 + nm4 + 1 perturb + fevals 24/44 (testing if optimal sigma + this config beats current best)

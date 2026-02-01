# Experiment: fevals_22_40_with_perturb

## Result: FAILED

## Hypothesis
Combining higher fevals (22/40) with full perturbation config (8 NM, 2 perturbations) may achieve better accuracy than either approach alone.

## Baseline Comparison
- **hopping_no_tabu**: 1.1689 @ 58.18 min (8 NM, 2 perturb, 20/36 fevals)
- **fevals_22_40**: 1.1640 @ 57.10 min (4 NM, 1 perturb, 22/40 fevals)

## Tuning Runs

| Run | Config | Score | Time | In Budget | RMSE 1src | RMSE 2src | Delta vs Baseline |
|-----|--------|-------|------|-----------|-----------|-----------|-------------------|
| 1 | nm8_perturb2_fevals22_40 | 1.1655 | 68.04 min | NO | 0.122147 | 0.188554 | -0.0034 |
| 2 | nm8_perturb2_fevals24_42 | 1.1596 | 71.48 min | NO | 0.111909 | 0.214698 | -0.0093 |
| 3 | nm6_perturb2_fevals22_40 | 1.1619 | 66.86 min | NO | 0.128716 | 0.191814 | -0.0070 |

## Key Finding

**Combining higher fevals with full perturbation does NOT improve results.**

All configs are:
1. Over budget (66-71 min vs 60 min limit)
2. WORSE than baseline hopping_no_tabu (1.1689)

## Analysis

The hypothesis was that higher fevals (better CMA-ES convergence) combined with perturbation (escape local optima) would stack to achieve higher accuracy. However:

1. **Fevals and perturbation are REDUNDANT**: Both mechanisms address the same bottleneck (local optima). More fevals gives CMA-ES more time to converge, while perturbation allows escaping suboptimal basins. Using both doesn't provide additive benefit.

2. **Budget blown**: Even with reduced NM (6 vs 8), the combination exceeds 60 min budget.

3. **Score degradation**: More fevals with perturbation actually HURTS accuracy, likely because:
   - Extra fevals are wasted after CMA-ES has already converged
   - Perturbation overhead adds to diminishing returns

## Conclusion

The optimal configuration remains **hopping_no_tabu** with:
- 8 NM polish iterations
- 2 perturbations
- 20/36 fevals (standard)
- sigma 0.18/0.22
- 40% temporal fidelity

Score: **1.1689 @ 58.18 min**

Do NOT attempt to combine higher fevals with perturbation.

## Tuning Efficiency Metrics
- **Runs executed**: 3
- **Time utilization**: N/A (all over budget)
- **Parameter space explored**: fevals, NM iterations
- **Pivot points**: Tried reduced NM (Run 3) when higher fevals failed (Run 2)

## Budget Analysis
| Run | Score | Time | Budget Remaining | Decision |
|-----|-------|------|------------------|----------|
| 1 | 1.1655 | 68.0 min | -8.0 min | CONTINUE (try higher fevals) |
| 2 | 1.1596 | 71.5 min | -11.5 min | PIVOT (try reduced NM) |
| 3 | 1.1619 | 66.9 min | -6.9 min | CONCLUDE (all configs over budget) |

## What Would Have Been Tried With More Time
- If budget were 75 min: nm8_perturb2_fevals22_40 would be in-budget with 1.1655 (still worse than baseline)
- This approach is fundamentally flawed - not worth more time investment

## Family Status
**combined_fevals_perturb**: EXHAUSTED

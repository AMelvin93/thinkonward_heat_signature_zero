# Final Gradient Verification Experiment

## Status: FAILED

## Hypothesis
Quick L-BFGS-B verification on final best candidate could catch edge cases without full optimization overhead. Running gradient-based refinement at the end, only on the best candidate, might improve accuracy with minimal time cost.

## Results Summary

| Run | Config | Score | Time (min) | Budget Status |
|-----|--------|-------|------------|---------------|
| 1 | gradient_verify_3iter_fine | 1.1906 | 146.2 | -86 min OVER |
| 2 | gradient_verify_2iter_fine | 1.1859 | 138.1 | -78 min OVER |
| 3 | no_gradient_verify_baseline | 1.1556 | 68.8 | -9 min OVER |

**Baseline**: 1.1464 @ 51.2 min

## Key Finding

**Gradient verification significantly improves accuracy but adds prohibitive overhead.**

- Best score with gradient verify: **1.1906** (but takes 146 min - 2.4x over budget)
- Without gradient verify: **1.1556** (still slightly over budget at 69 min)
- Score improvement from gradient verify: +0.035 (+3.0%)
- Time overhead from gradient verify: +70-78 min per run

## Root Cause Analysis

The L-BFGS-B verification on the fine grid is extremely expensive:

1. **Fine grid evaluation is slow**: Each L-BFGS-B iteration requires gradient computation via finite differences, which requires multiple PDE solves on the 100x50 fine grid
2. **Gradient computation overhead**: Each gradient evaluation for a 2-source problem needs 8-10 PDE solves (2 params x 2 sources x 2 for central difference)
3. **L-BFGS-B needs multiple iterations**: Even with just 2-3 iterations, the gradient evaluations dominate runtime

## Analysis of Improvement Source

The gradient verification improved RMSE significantly:
- 1-source RMSE: 0.128 → 0.100 (22% improvement)
- 2-source RMSE: 0.210 → 0.145 (31% improvement)

This suggests that NM polish leaves significant room for improvement near the final solution. However, capturing this improvement within budget requires a fundamentally different approach.

## Tuning Efficiency Metrics

- **Runs executed**: 3
- **Total experiment time**: ~5.5 hours
- **Parameter space explored**: gradient_verify_maxiter=[2,3], gradient_verify_use_fine_grid=[true]
- **Budget utilization**: All configs over budget

## Budget Analysis

| Run | Score | Time | Budget Remaining | Decision |
|-----|-------|------|------------------|----------|
| 1 | 1.1906 | 146 min | -86 min | Document over-budget |
| 2 | 1.1859 | 138 min | -78 min | Slightly faster but still over |
| 3 | 1.1556 | 69 min | -9 min | Baseline also over budget |

## Conclusion

**Final gradient verification FAILS** due to excessive runtime overhead.

The approach shows that gradient-based refinement significantly improves accuracy, but the cost is prohibitive within the 60-minute budget. The fine grid L-BFGS-B verification adds 70-78 minutes per run.

Key insights:
1. L-BFGS-B can improve final solution by 22-31% RMSE reduction
2. But gradient computation on fine grid is extremely expensive
3. Even 2 L-BFGS-B iterations add ~70 min overhead
4. The baseline without gradient verify also runs slightly over budget (69 min)

## What Would Have Been Tried With More Time

If budget were ~200 min:
- Try gradient verify on coarse grid (faster gradient evaluation)
- Try reduced timesteps for gradient computation
- Try single L-BFGS-B iteration

## Recommendations

1. **Mark gradient_verification family as EXHAUSTED** - cannot fit within budget
2. **Do not pursue fine-grid gradient methods** - too expensive
3. **Potential future direction**: Coarse grid gradient verification (but likely limited benefit)
4. **Key insight for future work**: The ~30% RMSE improvement potential suggests the NM polish is leaving accuracy on the table, but we need a cheaper way to capture it

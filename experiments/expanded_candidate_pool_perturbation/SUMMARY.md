# Expanded Candidate Pool Perturbation Experiment

## Status: FAILED

## Hypothesis
CMA-ES produces limited candidates. Perturbing top-K candidates can generate a larger pool of quality candidates, then selecting the best subset could improve accuracy.

## Results Summary

| Run | Config | Score | Time (min) | Budget | Status |
|-----|--------|-------|------------|--------|--------|
| 1 | expand_3x1_nm4 | 1.1414 | 407.2 | -347.2 min | OVER BUDGET |
| 2 | expand_2x1_nm3_reduced | 1.1422 | 374.7 | -314.7 min | OVER BUDGET |
| 3 | minimal_expand | 1.1415 | 338.9 | -278.9 min | OVER BUDGET |

**Baseline**: 1.1464 @ 51.2 min

## Key Finding

**The expanded candidate pool approach is fundamentally incompatible with the time budget.**

Even the most minimal configuration (expand_top_n=2, perturb_nm_iters=2, refine_maxiter=6) still results in 338.9 min projected time - nearly **6x over budget**.

## Root Cause Analysis

The approach adds excessive overhead through multiple mechanisms:

1. **Perturbation overhead**: For each of the top-K candidates, we generate perturbations
2. **NM polish overhead**: Each perturbed position requires NM optimization
3. **Fine-grid evaluation**: All candidates (original + perturbed) must be evaluated on the fine grid

The baseline optimizer already perturbs just 1 candidate with 2 perturbations. Expanding this to perturb multiple candidates creates multiplicative overhead.

## Tuning Efficiency Metrics

- **Runs executed**: 3
- **Time utilization**: N/A (all over budget)
- **Parameter space explored**: expand_top_n=[2,3], n_perturbations=[1,2], perturb_nm_iters=[2,3,4]
- **Pivot points**: Run 1 (407 min) → reduced to expand_2x1 → still over → minimal_expand → still 6x over budget

## Budget Analysis

| Run | Score | Time | Budget Remaining | Decision |
|-----|-------|------|------------------|----------|
| 1 | 1.1414 | 407 min | -347 min | PIVOT - reduce params |
| 2 | 1.1422 | 375 min | -315 min | PIVOT - more reduction |
| 3 | 1.1415 | 339 min | -279 min | ABORT - fundamentally over budget |

## Conclusion

**Expanded candidate pool perturbation FAILS** due to excessive runtime overhead.

The approach cannot fit within the 60-minute budget even with minimal parameters. This confirms that the baseline's single-candidate perturbation approach is near-optimal for the time budget.

Key insights:
1. Perturbing multiple top candidates adds multiplicative overhead
2. Each NM polish on perturbed positions is expensive (~50-100 evals)
3. Fine-grid verification of all candidates scales poorly
4. The baseline's perturb_top_n=1 is optimal for the budget

## What Would Have Been Tried With More Time
- If budget were 200 min: Try expand_top_n=2 with n_perturbations=1
- If budget were 400 min: Try expand_top_n=3 with n_perturbations=2

## Recommendations

1. **Do not pursue expanded candidate pool approaches** - they cannot fit the budget
2. **Baseline perturbation (perturb_top_n=1) is optimal** for this time constraint
3. **Focus on improving perturbation quality** rather than quantity
4. **Mark candidate_expansion family as EXHAUSTED**

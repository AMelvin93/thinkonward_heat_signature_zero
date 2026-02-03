# Experiment: reduced_fevals_4pert_v2

## Status: FAILED - Reduced fevals still over budget

## Hypothesis
Reducing CMA-ES function evaluations might allow 4 perturbations to fit within budget.

## Results

| Config | fevals | Score | Time (proj 400) | Budget |
|--------|--------|-------|-----------------|--------|
| 4pert_nm2 (baseline) | 20/44 | 1.1546 | 70.9 min | OVER |
| fevals_16_32_4pert | 16/32 | **1.1574** | **78.0 min** | OVER |
| fevals_14_28_4pert | 14/28 | 1.1467 | 74.0 min | OVER |

## Key Finding: Reduced fevals makes it SLOWER

This is counterintuitive! Reducing fevals from 20/44 to 16/32:
- Score slightly improved: 1.1546 → 1.1574
- But time INCREASED: 70.9 → 78.0 min

Possible explanations:
1. Fewer CMA-ES iterations lead to worse initial solutions
2. More NM polish iterations are needed to compensate
3. Significant run-to-run variance in these measurements

## Conclusion

**RESULT: FAILED - Cannot fit 4 perturbations in budget via fevals reduction**

All tested configs exceed the 60 min budget significantly.

## Final System Benchmark Summary

On THIS system, the only in-budget option is:
- **no_perturb @ 20/44**: 56.3 min, score 1.1367

All perturbation configs tested are over budget:
- 4pert_nm2 @ 20/44: 70.9 min
- All NM variations: 69-74 min
- All fevals variations: 74-78 min

---
**Worker**: W1
**Date**: 2026-02-02
**Status**: FAILED

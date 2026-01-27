# Experiment Summary: cmaes_quasi_newton_polish

## Status: ABORTED (Family Exhausted)

## Experiment ID: EXP_CMAES_QN_HYBRID_001
## Worker: W1
## Date: 2026-01-26

## Hypothesis
Literature shows CMA-ES + quasi-Newton hybrid can outperform pure CMA-ES. QN provides faster local convergence after CMA-ES finds the basin.

## Why Aborted

### L-BFGS-B Polish Already Tested and Failed

Prior experiment `cmaes_then_gradient_refinement` (EXP_HYBRID_CMAES_LBFGSB_001):

> "L-BFGS-B polish NOT better than NM polish. Finite diff overhead (O(n) extra sims/iter) outweighs gradient advantage. Best in-budget L-BFGS-B: 1.1174 @ 42 min (WORSE than NM x8: 1.1415 @ 38 min)."

### BFGS Polish Also Aborted

Prior experiment `bfgs_polish_after_cmaes` (EXP_BFGS_POLISH_001):

> "BFGS is essentially the same as L-BFGS-B but without limited memory. Both require finite differences for gradients, which is the core issue."

### Why NM Beats Gradient Methods for 2-4D Polish

| Method | Evals per Iteration | Issue |
|--------|---------------------|-------|
| Nelder-Mead | ~5 (simplex) | Optimal for low-D |
| BFGS/L-BFGS-B | 2n+1 (finite diff) | Gradient overhead |
| Powell | O(n) × O(10-30) | Many line searches |

For 4D (2-source) problems:
- NM: ~5 evals/iter
- BFGS: ~9 evals/iter (just for gradient)

### Local Search Family is EXHAUSTED

All tested alternatives (L-BFGS-B, Powell, BFGS, COBYLA, trust-constr) either:
1. Use more function evaluations than NM
2. Achieve worse accuracy
3. Both

Nelder-Mead is optimal for 2-4D polish with expensive function evaluations.

## Recommendation
**STAY with Nelder-Mead for polish.** Do NOT pursue any quasi-Newton or gradient-based polish methods.

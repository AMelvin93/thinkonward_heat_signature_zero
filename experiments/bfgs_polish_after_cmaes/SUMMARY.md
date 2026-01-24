# BFGS Polish After CMA-ES

## Experiment ID: EXP_BFGS_POLISH_001

## Status: ABORTED (Prior Evidence)

## Original Hypothesis
BFGS builds approximate Hessian from gradient evaluations. With finite differences, it may converge faster than NM for final polishing from a good CMA-ES starting point.

## Why Aborted

### Prior Evidence: All Alternative Polish Methods Have Failed

#### 1. L-BFGS-B Polish (EXP_HYBRID_CMAES_LBFGSB_001)
Already tested gradient-based polish with L-BFGS-B. Result: **FAILED**

> "L-BFGS-B polish NOT better than NM polish. Finite diff overhead (O(n) extra sims/iter) outweighs gradient advantage. Best in-budget L-BFGS-B: 1.1174 @ 42 min (WORSE than NM x8: 1.1415 @ 38 min)."

BFGS is essentially the same as L-BFGS-B but without limited memory. Both require finite differences for gradients, which is the core issue.

#### 2. Powell Polish (EXP_POWELL_POLISH_001)
Tested coordinate-wise line search polish. Result: **FAILED MASSIVELY**

- Score: 1.1413 @ 244.9 min (4.2x over budget)
- Powell uses 1000-3500 sims for 2-source samples (vs ~200 for NM)
- Coordinate-wise line searches require O(n) 1D optimizations per iteration

### Why NM Wins for 2-4D Polish

| Method | Evaluations per Iteration | Why It Fails |
|--------|--------------------------|--------------|
| NM | ~5 (simplex operations) | Optimal for low-D |
| BFGS/L-BFGS-B | 2n+1 (finite diff gradient) | Gradient overhead |
| Powell | O(n) × O(10-30) per line search | Many 1D optimizations |

For 4D (2-source) problems:
- NM: ~5 evals/iter
- BFGS: ~9 evals/iter (just for gradient)
- Powell: 40-120 evals/iter

The finite difference overhead makes any gradient-based method slower than NM for this low-dimensional problem.

## Recommendation

**STAY with Nelder-Mead for polish.** The local_search family should be considered EXHAUSTED for alternative polish methods.

All tested alternatives (L-BFGS-B, Powell, BFGS) either:
1. Use more function evaluations than NM
2. Achieve worse accuracy
3. Both

NM's simplex approach is optimal for 2-4D problems where each function evaluation is expensive.

## Prior Experiment References
- `experiments/cmaes_then_gradient_refinement/SUMMARY.md` (L-BFGS-B polish results)
- `experiments/powell_polish_instead_nm/SUMMARY.md` (Powell polish results)

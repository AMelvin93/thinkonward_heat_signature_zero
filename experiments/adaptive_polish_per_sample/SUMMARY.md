# Experiment Summary: adaptive_polish_per_sample

## Status: ABORTED (Prior Evidence)

## Experiment ID: EXP_ADAPTIVE_POLISH_BUDGET_001
## Worker: W1
## Date: 2026-01-26

## Hypothesis
Use more polish iterations for samples where CMA-ES converged poorly (high RMSE), and fewer for samples that converged well (low RMSE).

## Why Aborted

### Already Tested in adaptive_nm_iterations

Prior experiment `adaptive_nm_iterations` (EXP_ADAPTIVE_NM_POLISH_001) tested exactly this approach:

> "Dynamically adjust Nelder-Mead (NM) polish iterations based on convergence rate"

**Result: FAILED**
- Adaptive 4-12 iters: Score 1.1545 @ 78.7 min
- Fixed 8/8: Score 1.1607 @ 78.3 min
- Baseline: Score 1.1688 @ 58.4 min

### Key Findings from Prior Experiment

From `experiments/adaptive_nm_iterations/SUMMARY.md`:

1. **Fixed iterations are already optimal**
   - NM's built-in tolerance (`fatol`, `xatol`) handles early termination
   - No benefit from manual convergence checking

2. **Multiple minimize() calls add overhead**
   - Batched approach (4+2+2+...) is slower than single call (8)
   - Each `minimize()` call has startup overhead

3. **More iterations ≠ better accuracy**
   - Hard samples are structurally harder (4D search space), not under-iterated
   - Giving them more iterations doesn't improve accuracy

### Proposed vs Tested

| adaptive_polish_per_sample | adaptive_nm_iterations (Run 1) |
|---------------------------|--------------------------------|
| RMSE > 0.15: 10 iters | Adaptive 4-12 iters based on convergence |
| RMSE < 0.10: 6 iters | Early stopping with threshold |
| Otherwise: 8 iters | Batch size = 2 |

Both approaches try to allocate polish budget based on difficulty. The prior test showed this doesn't work.

## Recommendation

**Do NOT pursue adaptive polish iterations.** The `refinement` family is EXHAUSTED:
- Fixed 8 NM iterations is optimal
- Adaptive approaches add overhead without accuracy benefit
- NM's built-in tolerance handles early termination

See `experiments/adaptive_nm_iterations/SUMMARY.md` for detailed analysis.

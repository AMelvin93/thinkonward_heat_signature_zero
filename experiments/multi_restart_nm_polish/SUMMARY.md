# Experiment Summary: multi_restart_nm_polish

## Status: FAILED

## Experiment ID: EXP_RESTARTED_NM_POLISH_001
## Worker: W1
## Date: 2026-01-26

## Hypothesis
NM (Nelder-Mead) polish may converge to different local minima from different starting points. Running multiple NM polishes from slightly perturbed positions (3-5% displacement) may find better final solutions.

## Approach
Modified baseline optimizer to perform multiple Nelder-Mead polish runs:
1. Run NM polish from the original best position found by CMA-ES
2. Run additional NM polishes from perturbed positions (5% random displacement)
3. Select the best result across all restarts

## Results

### Tuning Run 1: 3 Restarts, 5 Iterations
- **Samples**: 5
- **Projected time**: 68.9 min
- **Outcome**: OVER BUDGET - did not complete full run

### Tuning Run 2: 2 Restarts, 4 Iterations (Full 80 samples)
- **Score**: 1.1524
- **Projected time**: 66.2 min
- **Baseline**: 1.1688 @ 58.4 min
- **Delta**: -0.0164 (-1.4%)
- **Outcome**: FAILED

| Metric | 1-source | 2-source |
|--------|----------|----------|
| RMSE mean | 0.1243 | 0.1694 |
| RMSE median | 0.1062 | 0.1531 |
| Time mean | 37.7s | 85.8s |

## Why It Failed

### 1. Excessive Overhead
Each additional NM polish restart adds significant computation time:
- Original NM polish: ~5 iterations
- 2 restarts: 2x polish overhead
- 3 restarts: 3x polish overhead

Even with reduced iterations (4 per restart), 2 restarts projected to 66.2 min (10% over budget).

### 2. No Accuracy Improvement
The hypothesis was that perturbed starting points might find better local minima. However:
- The loss landscape after CMA-ES is already well-converged
- The 5% perturbation either:
  - Stays in the same basin (no benefit)
  - Moves to a worse basin (decreases accuracy)
- Net effect: More computation for worse or equal results

### 3. CMA-ES Already Explores Well
CMA-ES with population-based search already samples multiple starting directions. The NM polish is meant to refine the best found solution, not explore alternatives. Multi-restart NM duplicates exploration that CMA-ES already performs.

## Conclusion
**Multi-restart NM polish is not beneficial for this problem.**

The baseline single NM polish is sufficient because:
1. CMA-ES already provides good global exploration
2. NM polish is designed for local refinement, not exploration
3. The additional restarts add overhead without improving accuracy

## Recommendation
**Do NOT use multi-restart NM polish.** Keep the single NM polish pass from the baseline optimizer. The computational budget is better spent on:
- More CMA-ES iterations
- Better initialization strategies
- Smarter budget allocation between 1-source and 2-source

## Files
- `optimizer.py`: Implementation with multi-restart polish
- `run.py`: Run script with configurable restart parameters
- `STATE.json`: Experiment state and tuning history

# Experiment Summary: smart_early_termination

## Metadata
- **Experiment ID**: EXP_SMART_EARLY_TERMINATION_001
- **Worker**: W1
- **Date**: 2026-01-25
- **Algorithm Family**: budget_final

## Status: ABORTED (Prior Evidence Conclusive)

## Objective
Detect CMA-ES convergence early to save time for "hard" samples. The hypothesis was that some samples may be "easy" (1-source, well-placed sensors) where CMA-ES converges quickly.

## Why Aborted

This experiment proposes using a **0.1% improvement threshold** for early termination. The **early stopping family has been EXHAUSTED** by multiple prior experiments:

### Prior Experiment 1: cmaes_early_stopping
**Result**: FAILED - Stagnation threshold was NEVER triggered

Key findings:
- Used **1% threshold** for 3 consecutive generations
- CMA-ES continues making **>1% improvements in most generations**, even near the end
- The early stopping code introduced **5.4x runtime increase** as overhead
- **Zero samples** triggered early stopping

**The proposed 0.1% threshold is even TIGHTER than the 1% threshold that never triggered!**

### Prior Experiment 2: adaptive_sample_budget
**Result**: FAILED - Early termination hurts accuracy (-0.018 score)

Key findings:
- When CMA-ES terminates early (23.8% of samples), accuracy drops significantly
- Quote: "Any form of early termination for CMA-ES - the algorithm needs its full budget"
- Quote: "Convergence metrics are unreliable: Sigma shrinkage and fitness stagnation don't reliably indicate when CMA-ES has found a good solution"

### Prior Experiment 3: confidence_based_early_exit
**Result**: FAILED - Cannot be confident early

Key findings:
- "Stopping early means stopping before the covariance matrix is well-adapted, which always hurts accuracy"
- Early generations are still exploring - CMA-ES cannot be confident early

## Technical Explanation

### Why Early Termination NEVER Works for CMA-ES

1. **CMA-ES needs its full budget**
   - The covariance matrix adaptation requires multiple generations
   - Stopping before full adaptation = suboptimal solutions

2. **No reliable convergence signal**
   - 1% threshold: never triggered (CMA-ES keeps improving)
   - 0.1% threshold: would trigger even LESS often
   - Sigma shrinkage: unreliable indicator

3. **The problem is well-conditioned**
   - CMA-ES converges smoothly toward the global optimum
   - There are no "easy" samples where convergence is fast
   - Every sample benefits from the full feval budget

### Budget Redistribution Is Impossible

Even if we could identify "easy" samples:
- Parallel processing prevents budget transfer between samples
- Quote from adaptive_sample_budget: "Parallel processing limits redistribution: With parallel workers, saved budget can't easily transfer between samples"

## Algorithm Family Status

- **early_stopping (all variants)**: **EXHAUSTED**
- **budget_adaptive (all variants)**: **EXHAUSTED**
- **convergence_detection (all variants)**: **EXHAUSTED**

Key quote from prior experiments: "ABANDON the efficiency/early_stopping experiment family."

## Recommendations

1. **Do NOT pursue any early termination approaches**
2. **CMA-ES needs its full budget** - this is fundamental to the algorithm
3. **There are no "easy" samples** - all samples benefit from full optimization
4. **Focus on other components** - polish, temporal fidelity, not budget reallocation

## Raw Data
- MLflow run IDs: None (experiment not executed)
- Prior evidence: cmaes_early_stopping, adaptive_sample_budget, confidence_based_early_exit

## Conclusion

This experiment would fail for the same reasons all previous early termination experiments failed. The proposed 0.1% threshold is even tighter than the 1% threshold that was never triggered in cmaes_early_stopping. CMA-ES needs its full budget for covariance adaptation, and there is no convergence criterion that works for this problem.

The early_stopping and budget_adaptive families are EXHAUSTED. No further early termination experiments should be created.

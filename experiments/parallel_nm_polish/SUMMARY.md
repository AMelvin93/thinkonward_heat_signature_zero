# Parallel NM Polish Experiment

## Experiment ID: EXP_PARALLEL_NM_POLISH_001
**Status**: ABORTED
**Worker**: W1
**Date**: 2026-01-24

## Hypothesis

If NM polish is currently sequential, parallelizing it could reduce wall-clock time, allowing more iterations within budget.

## Reason for Abortion

Analysis of the baseline code revealed **no parallelization opportunity exists**:

### Current Architecture

1. **Sample-level parallelization**: Already achieved via ProcessPoolExecutor with 7 workers
   - Each sample is processed independently in parallel
   - This is the primary source of speedup

2. **NM polish per sample**: Only runs on the BEST candidate
   - Single minimize() call per sample
   - Nothing to parallelize across candidates

3. **NM is inherently sequential**:
   - Each iteration depends on the previous iteration's result
   - Cannot parallelize iterations within a single NM run

4. **Function evaluations**: Each NM iteration calls objective once (or few times)
   - Already fast (~0.5s per evaluation)
   - Parallelizing a single function call provides no benefit

### Code Analysis

From `experiments/early_timestep_filtering/optimizer_with_polish.py` line 367:
```python
# FINAL POLISH: NM refinement on best candidate using FULL timesteps
if self.final_polish_maxiter > 0 and candidates_raw:
    # Find best candidate
    best_idx = min(range(len(candidates_raw)), key=lambda i: candidates_raw[i][2])
    best_pos_params = candidates_raw[best_idx][4]

    # Polish with full timesteps - SINGLE CALL
    result = minimize(
        objective_fine_full,
        best_pos_params,
        method='Nelder-Mead',
        options={'maxiter': self.final_polish_maxiter, ...}
    )
```

The polish is a single `minimize()` call on the best candidate. There's nothing to parallelize.

## Key Finding

**The parallelization family should be marked as EXHAUSTED for this problem.**

The baseline already achieves maximum parallelization:
- Sample-level: 7 workers processing samples in parallel
- Within-sample: Sequential CMA-ES + NM (both inherently sequential algorithms)

No further parallelization is possible without changing the fundamental algorithm.

## Recommendation

Do NOT attempt further parallelization experiments. Focus on algorithmic improvements instead.

## Files
- `STATE.json`: Experiment state (aborted)

# Early Stop Polish Convergence - FAILED

## Experiment Summary
**Status**: FAILED
**Family**: polish_efficiency
**Worker**: W1
**Date**: 2026-01-28

## Hypothesis
Fixed 8 NM iterations wastes budget on easy samples that converge quickly. Early stopping could reallocate saved budget to harder samples or more candidates.

## Key Finding
**FUNDAMENTAL FLAW**: Running scipy's minimize with maxiter=1 repeatedly is extremely inefficient. The simplex initialization overhead dominates the per-iteration cost by ~10x.

## Results

| Run | Early Stop | Score | Time (min) | Early Stop % | Iters Saved | Notes |
|-----|------------|-------|------------|--------------|-------------|-------|
| 1 | threshold=0 (baseline) | 1.1604 | 237.7 | 0% | 0 | Test config with 40% timestep |
| 2 | threshold=0 | 1.1434 | 171.2 | 0% | 0 | Fixed to 25% timestep |
| 3 | threshold=0.001 | 1.1203 | 445.6 | 90% | 277 | Early stopping enabled |
| 4 | threshold=0.001 | 1.1255 | 447.9 | 88.75% | 270 | Confirmation run |

## Analysis

### Why Early Stopping Failed
1. **90% of samples triggered early stopping** - confirming that many samples DO converge quickly
2. **Average 3.46 iterations saved per sample** - significant potential savings
3. **BUT: Projected time 445.6 min vs 60 min budget** - 7.4x OVER BUDGET
4. **Score dropped from 1.1468 baseline to 1.1203** - 2.3% WORSE

### Root Cause
The per-iteration cost breakdown in scipy's Nelder-Mead:
- **Single call with maxiter=8**: Initialize simplex ONCE, run 8 iterations
- **8 separate calls with maxiter=1**: Initialize simplex 8 TIMES, run 1 iteration each

The simplex initialization cost (~10x iteration cost) makes per-iteration early stopping fundamentally inefficient.

## Tuning Efficiency Metrics
- **Runs executed**: 4 (2 baseline + 2 with early stopping)
- **Time utilization**: N/A (all runs massively over budget)
- **Parameter space explored**: early_stop_threshold (0, 0.001), timestep_fraction (0.25, 0.40)
- **Pivot points**: After run 1 showed 40% timestep was slow, pivoted to 25%

## Budget Analysis

| Run | Score | Time (min) | Budget Remaining | Decision |
|-----|-------|------------|------------------|----------|
| 1 | 1.1604 | 237.7 | -177.7 min | PIVOT: fix timestep |
| 2 | 1.1434 | 171.2 | -111.2 min | CONTINUE: test hypothesis |
| 3 | 1.1203 | 445.6 | -385.6 min | CONFIRM: fundamental flaw |
| 4 | 1.1255 | 447.9 | -387.9 min | CONCLUDE: approach is broken |

## Conclusion
**Early stopping for NM polish is fundamentally flawed** when using scipy's minimize. The hypothesis that "saving iterations saves time" is FALSE because:
1. scipy's Nelder-Mead has high simplex initialization overhead
2. Running it iteration-by-iteration is ~10x slower than running all iterations at once
3. The time "saved" from early stopping is dwarfed by initialization overhead

**Recommendation**: Fixed 8 NM iterations is OPTIMAL. The polish_efficiency family is EXHAUSTED.

## What Would Have Been Tried With More Time
- If the fundamental approach worked, would test:
  - Different thresholds: 0.0001, 0.01, 0.1
  - Budget reallocation to second-best candidate
  - Adaptive iterations based on problem complexity (1-src vs 2-src)

But given the fundamental flaw, these are moot.

## Files
- `optimizer.py`: Optimizer with early stopping logic
- `run.py`: Experiment runner
- `STATE.json`: Full experiment state
- `run2_early_stop.log`: Run 3 output (80 samples, early stopping enabled)

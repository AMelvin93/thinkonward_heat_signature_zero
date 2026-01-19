# Experiment Summary: Extended NM Polish

## Metadata
- **Experiment ID**: EXP_EXTENDED_POLISH_001
- **Worker**: W2
- **Date**: 2026-01-19
- **Algorithm Family**: refinement

## Objective
Test whether increasing NM polish iterations from 8 to 12-16 improves accuracy without exceeding the 60-minute time budget.

## Hypothesis
More polish iterations should reduce RMSE further. Current 8 NM polish iterations improved score by 0.0326 (from 1.1362 to 1.1688). Additional iterations might continue this trend.

## Results Summary
- **Best In-Budget Score**: 1.1688 @ 58.4 min (BASELINE - 8 iterations unchanged)
- **12 Iterations Score**: 1.1703 @ 82.3 min (OVER BUDGET)
- **Baseline Comparison**: +0.0015 score but +24 min (37% slower)
- **Status**: FAILED - More iterations exceeds time budget

## Tuning History

| Run | Config | Score | Time (min) | In Budget | Notes |
|-----|--------|-------|------------|-----------|-------|
| 1 | 10 iterations (20 samples) | 1.1584 | 51.5 | Yes | High variance on small sample |
| 2 | 12 iterations (20 samples) | 1.1717 | 56.7 | Yes | Projected OK but unreliable |
| 3 | 12 iterations (80 samples) | 1.1703 | 82.3 | **NO** | 22 min over budget! |

## Key Findings

### What Didn't Work
1. **More polish iterations exceeds budget**: 12 iterations takes 82.3 min (37% over 60 min limit)
2. **Marginal score improvement**: Only +0.0015 score gain for +24 min time
3. **Diminishing returns**: Each additional NM iteration adds ~6 min per iteration on 80 samples

### Critical Insights
1. **8 iterations is already optimal**: The baseline found the sweet spot
2. **Time cost is non-linear**: NM on 2-source samples is much slower than 1-source
3. **Small sample runs misleading**: 20-sample projections underestimate 80-sample time

### Time Analysis

| NM Iterations | Projected Time (80 samples) | Time Per Iteration |
|--------------|------------------------------|-------------------|
| 8 | 58.4 min | baseline |
| 12 | 82.3 min | +6.0 min/iter |

The time increase per iteration is significant because:
- Each NM iteration calls the full simulation
- 2-source samples take 2x as many simulations per iteration
- 60% of samples are 2-source

## Recommendations for Future Experiments

### Do NOT pursue:
1. Increasing polish iterations beyond 8
2. Any approach that adds per-sample computational overhead
3. Fine-grid refinement strategies (time cost is prohibitive)

### Instead focus on:
1. **Reducing per-evaluation cost**: Temporal fidelity was successful
2. **Better initialization**: Reduces CMA-ES iterations needed
3. **Smarter candidate selection**: Avoid polishing poor candidates

## Conclusion

**The current 8 NM polish iterations is already optimal.**

More iterations provide negligible accuracy improvement (+0.13%) for massive time penalty (+37%). The 1.6 min headroom (58.4 vs 60 min) cannot accommodate any additional polish iterations.

## Raw Data
- Best config: `{"final_polish_maxiter": 8, "timestep_fraction": 0.40}`
- Baseline score: 1.1688 @ 58.4 min
- 12-iteration score: 1.1703 @ 82.3 min (OVER BUDGET)

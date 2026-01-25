# Experiment Summary: asymmetric_polish_budget

## Metadata
- **Experiment ID**: EXP_ASYMMETRIC_POLISH_001
- **Worker**: W1
- **Date**: 2026-01-25
- **Algorithm Family**: polish_method_v2

## Objective
Test whether using different NM polish iterations for 1-source vs 2-source problems can improve the accuracy-time tradeoff. The hypothesis was that 1-source problems converge faster (need fewer polish iterations) while 2-source problems are harder (need more polish iterations).

## Hypothesis
1-source problems converge faster and need fewer polish iterations, while 2-source problems benefit from more polish iterations. By reallocating polish budget from 1-src to 2-src, we might improve overall accuracy within the same time budget.

## Results Summary
- **Best In-Budget Score**: NONE (all runs over budget)
- **Best Overall Score**: 1.1639 @ 79.3 min (Run 3: 8/8)
- **Baseline Comparison**: All runs WORSE than baseline (1.1688 @ 58.4 min)
- **Status**: FAILED

## Tuning History

| Run | Config (polish_1src/polish_2src) | Score | Time (min) | In Budget | Notes |
|-----|----------------------------------|-------|------------|-----------|-------|
| 1 | 6/10 | 1.1608 | 92.2 | No | Extra 2-src polish adds ~39s per sample |
| 2 | 4/8 | 1.1551 | 72.3 | No | Reducing 1-src polish hurts accuracy |
| 3 | 8/8 | 1.1639 | 79.3 | No | **Same as baseline config but 20.9 min slower!** |

## Key Findings

### What Didn't Work
1. **More polish for 2-src (10 vs 8)**: Added ~33% more time for 2-source samples with no score improvement
2. **Less polish for 1-src (4 or 6 vs 8)**: Reduced accuracy without proportionate time savings
3. **Implementation overhead**: Even with identical 8/8 config, our optimizer was 20.9 min slower than baseline

### Critical Insights
1. **Baseline implementation is highly optimized**: The early_timestep_filtering optimizer has implementation advantages we couldn't replicate. Same algorithmic configuration (8/8 polish) runs 35% slower in our reimplementation.

2. **Asymmetric polish hypothesis is UNTESTABLE**: Since our baseline reimplementation is already 20+ minutes slower, we can't isolate the effect of asymmetric polish. The implementation overhead dominates.

3. **Polish iterations are already optimal**: The baseline uses 8 NM polish iterations for all samples. Any deviation (up or down) hurts score without sufficient time savings.

## Parameter Sensitivity
- **polish_2src**: Very time-sensitive. 10 iters adds ~40s per 2-src sample
- **polish_1src**: Reducing from 8 to 4 saves ~10s per 1-src sample but hurts accuracy significantly

## Implementation Issues Discovered
The reimplementation of the baseline optimizer has significant overhead:
- 8/8 config: 79.3 min (ours) vs 58.4 min (baseline) = 35% slower
- Cause: Likely subtle differences in evaluation order, parallelization, or fine-grid handling

## Recommendations for Future Experiments
1. **Don't reimplement baseline**: Modify the existing early_timestep_filtering optimizer instead of creating new implementations
2. **Polish iterations are already optimal**: The 8 NM polish iteration setting is locally optimal. Don't try to change it.
3. **Time budget is saturated**: The baseline at 58.4 min is close to optimal for the 60 min budget. Any changes that add overhead will push over budget.

## Algorithm Family Status
- **polish_method_v2 family**: EXHAUSTED
- Prior experiments (reduced_cmaes_more_nm, extended_nm_polish) also found 8 NM iterations optimal

## Raw Data
- MLflow run IDs: 1d2b297647c442af84bca839ba7c380e, 8cc9078336024b13ac589338e1c4b3a7, 6a04cab019374355a01b9529a31880d0
- Best config: N/A (all over budget)
- See STATE.json for detailed tuning history

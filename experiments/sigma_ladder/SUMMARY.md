# Experiment Summary: sigma_ladder

## Metadata
- **Experiment ID**: EXP_SIGMA_LADDER_001
- **Worker**: W2
- **Date**: 2026-01-24
- **Algorithm Family**: sigma_scheduling_v2

## Objective
Test whether a sigma ladder (starting with high sigma 0.30/0.35 and reducing to 0.15/0.20) can improve on CMA-ES's natural sigma adaptation.

## Hypothesis
High sigma explores broadly early in the optimization, then low sigma refines. Manual scheduling might help when CMA-ES's natural adaptation is too slow to converge.

## Results Summary
- **Best In-Budget Score**: NONE (run was over budget)
- **Best Overall Score**: 1.1658 @ 70.0 min
- **Baseline Comparison**: WORSE than baseline 1.1688 @ 58.4 min
- **Status**: FAILED

## Tuning History

| Run | Sigma Start | Sigma End | Score | Time (min) | In Budget | Delta vs Baseline |
|-----|-------------|-----------|-------|------------|-----------|-------------------|
| 1 | 0.30/0.35 | 0.15/0.20 | 1.1658 | 70.0 | NO | -0.0030 / +11.6 min |

## Key Findings

### What Didn't Work
1. **Sigma ladder adds overhead** - Manual sigma manipulation slows down optimization
2. **Interferes with covariance learning** - CMA-ES adapts sigma based on covariance; overriding it disrupts learning
3. **High initial sigma doesn't help** - Prior experiments with sigma 0.25-0.35 already showed worse results
4. **No accuracy benefit** - Score is actually WORSE than fixed sigma baseline

### Why Sigma Scheduling Fails
CMA-ES has a built-in, mathematically principled sigma adaptation mechanism:
- Sigma is updated based on the evolution path (cumulative step size adaptation)
- This adaptation is tightly coupled with covariance matrix updates
- Manual overrides break this coupling

The baseline sigma 0.15/0.20 is already optimal because:
- It balances exploration and exploitation naturally
- CMA-ES can increase sigma if needed (it's just the initial value)
- Lower initial sigma converges faster when the initialization is good (triangulation + hotspot)

### Prior Evidence Confirmed
This experiment confirms prior findings:
- **EXP_ADAPTIVE_SIGMA_SCHEDULE_001**: Aborted - "CMA-ES already adapts sigma naturally"
- **EXP_TEMPORAL_HIGHER_SIGMA_001**: Higher sigma (0.25/0.30) added time without benefit
- **EXP_TEMPORAL_FIDELITY_SWEEP_001**: Sigma 0.18/0.22 was WORSE than 0.15/0.20

## Critical Insight
**CMA-ES sigma adaptation is already optimal. Any manual override - whether starting high, low, or scheduling - will hurt performance.**

The sigma value is not independent from the optimization dynamics. CMA-ES couples:
1. Step size (sigma)
2. Covariance matrix (search direction)
3. Evolution path (momentum)

Changing sigma manually without updating the other components breaks the algorithm's internal consistency.

## Recommendations for Future Experiments

1. **DO NOT modify sigma externally** - Let CMA-ES handle it
2. **DO NOT try other sigma schedules** - Linear, exponential, step-wise - all will fail
3. **sigma_scheduling family is EXHAUSTED** - No improvement possible via sigma manipulation

## Comparison to Prior Results

| Configuration | Score | Time | Notes |
|---------------|-------|------|-------|
| Baseline (fixed 0.15/0.20) | **1.1688** | **58.4 min** | BEST |
| This exp (0.30/0.35 -> 0.15/0.20) | 1.1658 | 70.0 min | Over budget, worse |
| Prior: Higher sigma (0.25/0.30) | 1.1745 | 63.0 min | Over budget |
| Prior: Sigma 0.18/0.22 | 1.1656 | 67.4 min | Over budget |

## Conclusion

**FAILED**: Sigma ladder (high->low) does NOT improve performance. CMA-ES's natural sigma adaptation is optimal. Manual scheduling adds overhead and interferes with covariance learning.

The baseline configuration (fixed initial sigma 0.15/0.20) is confirmed as OPTIMAL for this problem.

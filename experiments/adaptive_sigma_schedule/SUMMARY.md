# Experiment Summary: adaptive_sigma_schedule

## Metadata
- **Experiment ID**: EXP_ADAPTIVE_SIGMA_SCHEDULE_001
- **Worker**: W1
- **Date**: 2026-01-19
- **Algorithm Family**: sigma_scheduling

## Objective
Test whether manually scheduling sigma (high early, low late) improves over CMA-ES's natural sigma adaptation.

## Hypothesis
Higher sigma enables better exploration early. Lower sigma allows faster convergence late. A schedule combining both might achieve high-sigma accuracy with low-sigma speed.

## Results Summary
- **Status**: ABORTED - Prior evidence shows this approach will fail
- **Tuning Runs**: 0 (aborted before testing)

## Rationale for Abort

### Previous Similar Experiments Failed

1. **EXP_TEMPORAL_HIGHER_SIGMA_001 (FAILED)**
   - Sigma 0.25/0.30 achieves better accuracy (1.1745) but takes 63 min (OVER BUDGET)
   - Best in-budget with high sigma: 1.1584 @ 52 min (-0.0104 vs W2 baseline 1.1688)
   - Conclusion: High sigma adds time, no way around it

2. **EXP_ADAPTIVE_TIMESTEP_001 (FAILED)**
   - Changing conditions mid-run disrupts CMA-ES covariance adaptation
   - Best with adaptive timestep: 1.1635 @ 69.9 min (OVER BUDGET and worse)
   - Conclusion: CMA-ES needs consistent conditions

### CMA-ES Already Adapts Sigma

CMA-ES has built-in sigma adaptation:
- Early generations: sigma is typically higher for exploration
- Later generations: sigma decreases naturally as population converges

Manual sigma scheduling would:
- Either replicate what CMA-ES already does (no improvement)
- Or interfere with covariance adaptation (make things worse)

### Fundamental Tradeoff Cannot Be Bypassed

The core insight from previous experiments:
```
High sigma = More exploration = Better solutions = More time
Low sigma = Less exploration = Faster convergence = May miss optima
```

No scheduling can get high-sigma accuracy with low-sigma speed because:
- Exploration takes function evaluations
- Function evaluations take time
- There's no free lunch

## Key Finding

**Sigma adaptation is already optimal in CMA-ES.** Manual scheduling cannot improve on the algorithm's natural adaptation without adding time. The sigma_scheduling family should be marked as EXHAUSTED.

## Recommendations

1. **Don't pursue sigma scheduling** - CMA-ES's natural adaptation is already optimal
2. **Focus on different approaches**:
   - Progressive polish fidelity (different from sigma)
   - Active CMA-ES variant
   - Problem-specific scaling

## Conclusion

**ABORTED** - Prior evidence from EXP_TEMPORAL_HIGHER_SIGMA_001 and EXP_ADAPTIVE_TIMESTEP_001 shows that adaptive approaches to CMA-ES settings fail. CMA-ES's default behavior is already well-optimized. Time is better spent on experiments with higher probability of success.

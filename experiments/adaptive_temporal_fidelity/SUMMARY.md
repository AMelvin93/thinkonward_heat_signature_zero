# Experiment Summary: adaptive_temporal_fidelity

## Metadata
- **Experiment ID**: EXP_ADAPTIVE_TEMPORAL_001
- **Worker**: W1
- **Date**: 2026-01-25
- **Algorithm Family**: temporal_fidelity_v2

## Status: ABORTED (Duplicate of Previous Failed Experiment)

## Objective
Start CMA-ES with 20% timestep fidelity, increase to 40% when fitness variance drops. The hypothesis was that early exploration can use lower fidelity (faster), with high fidelity only needed for final convergence.

## Why Aborted

This experiment is a **duplicate** of `experiments/adaptive_timestep_fraction` (EXP_ADAPTIVE_TIMESTEP_001) which was already tested by W2 on 2026-01-19 and **FAILED**.

### Previous Results (from adaptive_timestep_fraction)

| Run | Config | Score | Time (min) | Notes |
|-----|--------|-------|------------|-------|
| 1 | low=0.25, high=0.40, switch=0.50 | 1.1473 | 66.0 | 25% early too inaccurate |
| 2 | low=0.35, high=0.40, switch=0.50 | 1.1635 | 69.9 | Better but still worse than baseline |

- **Baseline**: 1.1688 @ 58.4 min (fixed 40% timesteps)
- **Best from previous experiment**: 1.1635 @ 69.9 min (-0.0053 score, +11.5 min time)

## Why It Failed (Key Findings from Previous Experiment)

1. **CMA-ES covariance adaptation requires consistent fitness landscape**
   - Switching fidelity mid-run disrupts the covariance learning
   - The algorithm cannot properly adapt when the objective function changes

2. **Lower early fidelity doesn't save time if it leads to worse solutions**
   - Poor initial solutions require more fallback runs
   - More refinement needed later to compensate
   - Net effect: MORE time, not less

3. **Fixed approaches outperform adaptive ones**:
   - Fixed 25% timesteps: 1.1219 @ 30.3 min (fast but less accurate)
   - Fixed 40% timesteps: 1.1362 @ 39.0 min (good balance)
   - Fixed 40% + 8 NM polish: 1.1688 @ 58.4 min (best overall)
   - Adaptive 25%->40%: 1.1473 @ 66.0 min (WORSE than all fixed alternatives)

## Critical Insight

**Consistent fidelity is essential for CMA-ES** - the algorithm optimizes by adapting covariance based on fitness differences. If the fitness function changes mid-run (due to fidelity switch), the learned covariance becomes invalid. Multi-fidelity optimization with CMA-ES requires specialized approaches, not simple mid-run switching.

## Recommendations

1. **Do NOT pursue adaptive timestep approaches with CMA-ES** - they fundamentally conflict with covariance adaptation
2. **Multi-fidelity requires specialized algorithms** - if wanted, try multi-fidelity Bayesian optimization instead
3. **Focus on other optimizations** - the temporal fidelity gain has been captured with fixed 40%
4. **temporal_fidelity family is EXHAUSTED** - no further experiments in this direction

## References
- See `experiments/adaptive_timestep_fraction/SUMMARY.md` for full details
- MLflow run IDs from previous: 357d73dc2f0f4b2f9eb640b4a8348434, 385d6e0a12874c4ea28a85e9645181d3

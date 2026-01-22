# Experiment Summary: adaptive_timestep_fraction

## Metadata
- **Experiment ID**: EXP_ADAPTIVE_TIMESTEP_001
- **Worker**: W2
- **Date**: 2026-01-19
- **Algorithm Family**: temporal_fidelity_extended

## Objective
Test whether using variable timestep fidelity during CMA-ES optimization (starting low, switching to higher fidelity mid-run) could save time while maintaining accuracy.

## Hypothesis
Early CMA-ES iterations need only rough landscape estimates (25% timesteps fine). Later iterations benefit from more accurate signal (40% timesteps). Variable fidelity may save time while maintaining accuracy.

## Results Summary
- **Best In-Budget Score**: N/A (all runs over budget)
- **Best Overall Score**: 1.1635 @ 69.9 min (Run 2)
- **Baseline Comparison**: -0.0053 to -0.0215 vs 1.1688 @ 58.4 min
- **Status**: FAILED

## Tuning History

| Run | Config Changes | Score | Time (min) | In Budget | Notes |
|-----|---------------|-------|------------|-----------|-------|
| 1 | low=0.25, high=0.40, switch=0.50 | 1.1473 | 66.0 | No | 25% early too inaccurate, misleads CMA-ES |
| 2 | low=0.35, high=0.40, switch=0.50 | 1.1635 | 69.9 | No | Better than 25% but still worse than fixed 40% |

## Key Findings

### What Didn't Work
- **Variable timestep fidelity during CMA-ES is counterproductive**
  - Both adaptive configurations performed WORSE than fixed 40% baseline
  - Time increased by 7.6-11.5 min (not decreased as expected)
  - Score decreased by 0.0053-0.0215

### Why It Failed
1. **CMA-ES covariance adaptation requires consistent fitness landscape**
   - Switching fidelity mid-run disrupts the covariance learning
   - The algorithm cannot properly adapt when the objective function changes

2. **Lower early fidelity doesn't save time if it leads to worse solutions**
   - Poor initial solutions require more fallback runs
   - More refinement needed later to compensate
   - Net effect: MORE time, not less

3. **Comparison to fixed approaches** (from baseline experiment):
   - Fixed 25% timesteps: 1.1219 @ 30.3 min (fast but less accurate)
   - Fixed 40% timesteps: 1.1362 @ 39.0 min (good balance)
   - Fixed 40% + 8 NM polish: 1.1688 @ 58.4 min (best overall)
   - Adaptive 25%->40%: 1.1473 @ 66.0 min (WORSE than fixed alternatives)

### Critical Insights
- **Consistent fidelity is essential for CMA-ES** - the algorithm optimizes by adapting covariance based on fitness differences. If the fitness function changes mid-run (due to fidelity switch), the learned covariance becomes invalid.
- **Multi-fidelity optimization requires specialized approaches** - standard CMA-ES cannot handle changing objective functions. Would need algorithms specifically designed for variable fidelity (e.g., multi-fidelity BO).
- **The baseline approach is already optimal** - fixed 40% timesteps during CMA-ES + full-timestep NM polish is the right combination.

## Parameter Sensitivity
- **Most impactful parameter**: low_fidelity_fraction (25% vs 35% made significant difference)
- **Switching point** had less impact than expected

## Recommendations for Future Experiments
1. **Do NOT pursue adaptive timestep approaches** - they fundamentally conflict with CMA-ES
2. **Multi-fidelity requires specialized algorithms** - if wanted, try multi-fidelity Bayesian optimization instead of CMA-ES
3. **Focus on other optimizations** - the temporal fidelity gain has been captured; look elsewhere for improvements
4. **Possible alternative**: Instead of switching mid-run, use low-fidelity for initial population, then fixed high-fidelity for all remaining iterations

## Raw Data
- MLflow run IDs: 357d73dc2f0f4b2f9eb640b4a8348434, 385d6e0a12874c4ea28a85e9645181d3
- Best config: None (all failed)
- Baseline remains best: 40% timesteps + 8 NM polish = 1.1688 @ 58.4 min

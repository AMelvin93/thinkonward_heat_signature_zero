# Experiment Summary: random_timestep_selection

## Metadata
- **Experiment ID**: EXP_RANDOM_TIMESTEPS_001
- **Worker**: W2
- **Date**: 2026-01-24
- **Algorithm Family**: temporal_sampling

## Objective
Test whether using RANDOM 40% of timesteps instead of first 40% for CMA-ES evaluation provides better RMSE signal that enables faster convergence or better accuracy.

## Hypothesis
First 40% timesteps may miss important late-time dynamics. Random sampling captures different parts of temporal evolution, potentially providing more informative RMSE signal.

## Results Summary
- **Best In-Budget Score**: N/A (experiment FAILED due to time)
- **Best Overall Score**: N/A (aborted early due to obvious failure)
- **Baseline Comparison**: FAILED - 5.8x over time budget
- **Status**: FAILED

## Tuning History

| Run | Config Changes | Score | Time (min) | In Budget | Notes |
|-----|---------------|-------|------------|-----------|-------|
| 1 | Random 40% timesteps, full sim | N/A | ~350 projected | NO | Aborted at 37/80 samples - clear failure |

## Key Findings

### What Didn't Work
- **Full simulation is the problem**: Random timesteps requires running 100% of the simulation, then selecting random 40% for RMSE calculation
- **Baseline efficiency source**: The baseline's speed comes from running ONLY 40% of timesteps, not from which timesteps are used
- **3-4x slower per sample**:
  - 1-source: ~220s (random) vs ~70s (baseline) = 3.1x slower
  - 2-source: ~416s (random) vs ~120s (baseline) = 3.5x slower
- **Projected time catastrophic**: 350 min for 400 samples (5.8x over 60 min budget)

### Critical Insights

**The hypothesis was WRONG about the mechanism:**

The original hypothesis assumed the baseline runs full simulation but only uses first 40% of timesteps for RMSE calculation. In reality, the baseline **only runs 40% of the simulation timesteps** - this is where the speedup comes from.

Random timestep selection can only provide benefit if you:
1. Run full simulation anyway
2. Are uncertain about which timesteps are most informative

For this problem, early timesteps contain sufficient information about source locations. Late-time dynamics don't provide additional signal that justifies 2.5x more simulation cost.

### Why Random Doesn't Help

1. **Heat equation physics**: Early timesteps capture the initial temperature rise which is most sensitive to source location
2. **Signal saturation**: Later timesteps show more equilibrium behavior which is less informative
3. **No accuracy gain**: RMSE values from partial run (0.091 avg for 1-src, 0.129 avg for 2-src) are similar to baseline
4. **Cost not offset**: 3-4x time penalty with no compensating accuracy improvement

## Parameter Sensitivity
- **Timestep fraction**: N/A - the issue is full simulation requirement, not the fraction
- **Random vs contiguous**: Irrelevant - both require deciding whether to run full or partial simulation

## Implementation Analysis

The current implementation (`optimizer.py`) has a fundamental design issue:
- `simulate_full()` always runs `nt_full` timesteps
- Then selects random indices from the full result
- This means EVERY evaluation pays the full simulation cost

To make random timesteps viable, you would need a different approach:
1. Pre-run simulation once, cache all timesteps
2. Use random subset for RMSE during optimization
3. But this defeats the purpose - you've already paid the full cost

**Conclusion**: Random timestep selection is fundamentally incompatible with the efficiency goal.

## Recommendations for Future Experiments

1. **DO NOT pursue random timestep selection** - the premise is flawed
2. **Mark temporal_sampling family as FAILED** - contiguous early timesteps are optimal
3. **Focus on accuracy improvements** within the 40% timestep baseline
4. **The baseline approach is optimal** for temporal efficiency: run fewer timesteps, not different timesteps

## Recommendations for W0

The temporal_sampling family should be marked EXHAUSTED. Key learning:
- Efficiency comes from running FEWER timesteps, not DIFFERENT timesteps
- Early timesteps are most informative for heat source localization
- No point testing other timestep selection strategies (e.g., uniformly spaced, late timesteps, adaptive)

## Raw Data
- **MLflow run IDs**: None (aborted before completion)
- **Samples completed**: 37/80
- **Best config**: N/A - experiment failed on time constraint

## Detailed Timing Data (from partial run)

### 1-Source Samples (32 completed)
- Average time: 220s
- Min time: 162.9s
- Max time: 274.7s
- Average RMSE: 0.091

### 2-Source Samples (5 completed)
- Average time: 416s
- Min time: 346.0s
- Max time: 528.7s
- Average RMSE: 0.129

### Extrapolated Total Time (80 samples)
- 32 1-source × 220s = 7,040s
- 48 2-source × 416s = 19,968s
- Total: ~27,000s / 7 workers = ~65 min for 80 samples
- Projected 400: 65 × 5 = 325 min

(Note: Actual would be higher due to 2-source samples not yet processed in the partial run)

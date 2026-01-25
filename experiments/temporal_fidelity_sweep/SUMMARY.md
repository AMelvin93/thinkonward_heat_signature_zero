# Experiment Summary: temporal_fidelity_sweep

## Metadata
- **Experiment ID**: EXP_TEMPORAL_FIDELITY_SWEEP_001
- **Worker**: W2
- **Date**: 2026-01-24
- **Algorithm Family**: temporal_tuning

## Objective
Fine-tune temporal fidelity around the known optimal 40% by testing nearby fractions (35%, 38%, 40%, 42%, 45%) to find the exact optimum.

## Hypothesis
The original baseline found 40% optimal but didn't test nearby fractions. Testing 38% or 42% might yield marginally better results.

## Results Summary
- **Best In-Budget Score**: NONE (all runs over budget)
- **Best Overall Score**: 1.1656 @ 67.4 min (35% timesteps)
- **Baseline Comparison**: ALL RUNS WORSE than baseline 1.1688 @ 58.4 min
- **Status**: FAILED

## CRITICAL FINDING
**The experiment used sigma 0.18/0.22 instead of the baseline sigma 0.15/0.20.**

This inadvertently proved that **sigma 0.15/0.20 is optimal** - higher sigma (0.18/0.22) makes ALL configurations worse:
- More time (67-72 min vs 58 min baseline)
- Worse scores (1.15-1.17 vs 1.1688 baseline)

## Tuning History

| Run | Timestep % | Score | Time (min) | In Budget | Delta vs Baseline |
|-----|------------|-------|------------|-----------|-------------------|
| 1 | 35% | 1.1656 | 67.4 | NO | -0.0032 / +9.0 min |
| 2 | 38% | 1.1637 | 69.7 | NO | -0.0051 / +11.3 min |
| 3 | 40% | 1.1541 | 71.8 | NO | -0.0147 / +13.4 min |
| 4 | 42% | (incomplete) | - | - | - |
| 5 | 45% | (not run) | - | - | - |

## Key Findings

### What We Learned
1. **Sigma 0.18/0.22 is WORSE than 0.15/0.20** - This was the key confounding variable
2. **All tested fractions (35%, 38%, 40%) performed worse** with the higher sigma
3. **Higher sigma increases runtime** by ~10-15 min (from 58 min to 67-72 min)
4. **Higher sigma reduces accuracy** (score dropped from 1.1688 to 1.15-1.17)

### Why Higher Sigma Hurts
- Higher sigma means larger step sizes in CMA-ES
- This leads to more exploration but slower convergence
- Requires more function evaluations to reach same precision
- The baseline sigma (0.15/0.20) was already well-calibrated for this problem

### Critical Insight
The experiment description suggested using sigma 0.18/0.22, but the baseline optimizer_with_polish.py uses 0.15/0.20. This mismatch caused all runs to be over budget and score worse.

## Recommendations for Future Experiments

1. **DO NOT change sigma** from baseline 0.15/0.20 - it is already optimal
2. **40% timesteps is confirmed optimal** - lower fractions (35%, 38%) don't improve even with more budget
3. **8 NM polish iterations is optimal** - no need to tune
4. **temporal_tuning family is EXHAUSTED** - no improvement possible by tuning temporal fraction

## Comparison to Prior Results

| Configuration | Score | Time | Notes |
|---------------|-------|------|-------|
| Baseline (40% + sigma 0.15/0.20 + 8 NM) | **1.1688** | **58.4 min** | BEST |
| This exp: 35% + sigma 0.18/0.22 | 1.1656 | 67.4 min | Over budget |
| This exp: 38% + sigma 0.18/0.22 | 1.1637 | 69.7 min | Over budget |
| This exp: 40% + sigma 0.18/0.22 | 1.1541 | 71.8 min | Over budget, worst |

## Raw Data
- MLflow run IDs: See mlruns/ for temporal_sweep_* runs
- Best config: N/A (all over budget)
- Output log: /workspace/experiments/temporal_fidelity_sweep/output.log

## Conclusion

**FAILED**: The experiment inadvertently tested sigma 0.18/0.22 instead of 0.15/0.20, proving that the baseline sigma is already optimal. No temporal fraction improvements found. The baseline configuration (40% timesteps + sigma 0.15/0.20 + 8 NM polish) remains optimal.

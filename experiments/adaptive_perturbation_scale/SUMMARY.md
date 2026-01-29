# Adaptive Perturbation Scale Experiment

## Experiment ID
EXP_ADAPTIVE_PERTURB_SCALE_001

## Hypothesis
Worse solutions may be stuck in poor basins and need larger perturbations to escape. Better solutions need smaller perturbations for fine-tuning.

## Baseline
- **perturbed_extended_polish**: 1.1464 @ 51.2 min (fixed scale 0.05)

## Results

### FAILED - Adaptive perturbation scale does NOT improve over fixed scale

| Run | High Scale | Mid Scale | Low Scale | Score | Time (min) | In Budget | vs Baseline |
|-----|------------|-----------|-----------|-------|------------|-----------|-------------|
| 1 | 0.10 | 0.05 | 0.02 | 1.1442 | 63.2 | NO | -0.0022 |
| 2 | 0.08 | 0.04 | 0.02 | 1.1454 | 61.2 | NO | -0.0010 |
| 3 | 0.05 | 0.05 | 0.05 | 1.1419 | 57.4 | YES | -0.0045 |

**Best in-budget**: Run 3 with 1.1419 @ 57.4 min
**Delta vs baseline**: -0.0045 (WORSE)

## Tuning Efficiency Metrics
- **Runs executed**: 3
- **Time utilization**: 96% (57.4/60 min for best in-budget)
- **Parameter space explored**:
  - perturbation_scale_high: [0.05, 0.08, 0.10]
  - perturbation_scale_mid: [0.04, 0.05]
  - perturbation_scale_low: [0.02, 0.05]
  - rmse_high_threshold: 0.30 (fixed)
  - rmse_mid_threshold: 0.15 (fixed)
- **Pivot points**:
  - Run 1 over budget → Reduced scales
  - Run 2 still over budget → Tried fixed scale (non-adaptive)
  - Run 3 in budget but worse score → Confirmed adaptive doesn't help

## Budget Analysis
| Run | Score | Time | Budget Remaining | Decision |
|-----|-------|------|------------------|----------|
| 1 | 1.1442 | 63.2 | -3.2 min (OVER) | PIVOT - reduce scales |
| 2 | 1.1454 | 61.2 | -1.2 min (OVER) | PIVOT - try fixed scale |
| 3 | 1.1419 | 57.4 | +2.6 min | COMPLETE - adaptive disproved |

## Why Adaptive Perturbation Scale Fails

1. **Overhead without benefit**: Adaptive scaling logic adds computation without improving accuracy
2. **Wrong hypothesis**: RMSE level doesn't correlate well with "stuck in local minimum" status
3. **Fixed scale already optimal**: 0.05 perturbation scale works well regardless of RMSE
4. **Perturbation is already local**: With 0.05 scale, perturbations stay within convergence basin

## Key Findings

1. **Fixed perturbation scale (0.05) is optimal**: Baseline already found the right balance
2. **RMSE-based scaling doesn't predict optimization difficulty**: High RMSE solutions aren't necessarily stuck
3. **Simpler is better**: Adaptive logic adds complexity without benefit

## Recommendation

**DO NOT use adaptive perturbation scale.** Stick with:
- perturbed_extended_polish (1.1464 @ 51.2 min) with fixed scale 0.05

## Conclusion

**EXPERIMENT FAILED**

The hypothesis that adaptive perturbation scaling based on RMSE quality would improve results is **DISPROVED**. All configurations tested either exceeded budget or scored worse than baseline. The fixed perturbation scale of 0.05 remains optimal.

The perturbation_v2 family should be marked as EXHAUSTED.

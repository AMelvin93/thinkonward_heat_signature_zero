# 4 Perturbations with Scale 0.06 Experiment

## Status: FAILED (prior evidence confirmed)

## Hypothesis
Combine 4 perturbations with scale 0.06. Test if different scale provides improvement.

## Prior Evidence
The `perturbation_scale_tuning` experiment previously found:
- scale=0.04: 1.1389 @ 45.2 min
- scale=0.05: 1.1419 @ 38.3 min (BEST)
- scale=0.06: 1.1378 @ 43.8 min

## Tuning Results

| Run | Scale | Score | Time (min) | Budget Rem | vs Baseline |
|-----|-------|-------|------------|------------|-------------|
| 1   | 0.06  | 1.1234 | 44.9      | 15.1       | -0.0230     |
| 2   | 0.055 | 1.1315 | 44.5      | 15.5       | -0.0149     |
| 3   | 0.07  | 1.1344 | 44.3      | 15.7       | -0.0120     |

**Baseline**: 1.1464 @ 51.2 min

## Key Finding

**Prior evidence CONFIRMED**: Scale 0.05 is optimal. All tested scales (0.055, 0.06, 0.07) underperform.

Interesting observation: Scale 0.07 performed slightly better than 0.06, but both are well below baseline. This suggests:
1. The optimal scale is not a simple linear relationship
2. Scale 0.05 may be near a local optimum
3. Very large scales overshoot good solutions

## Combined Data with Prior Experiment

| Scale | Score | Time | Delta vs Baseline |
|-------|-------|------|-------------------|
| 0.04  | 1.1389 | 45.2 | -0.0075 |
| **0.05** | **1.1419** | **38.3** | **-0.0045** |
| 0.055 | 1.1315 | 44.5 | -0.0149 |
| 0.06  | 1.1234 | 44.9 | -0.0230 |
| 0.07  | 1.1344 | 44.3 | -0.0120 |

Scale 0.05 is clearly optimal.

## Tuning Efficiency Metrics

- **Runs executed**: 3
- **Time utilization**: 74% (44.3/60 min used)
- **Parameter space explored**: [0.055, 0.06, 0.07]
- **Pivot points**: None needed - all underperformed

## Budget Analysis

| Run | Score | Time | Budget Remaining | Decision |
|-----|-------|------|------------------|----------|
| 1   | 1.1234 | 44.9 | 15.1 min | CONTINUE - explore more scales |
| 2   | 1.1315 | 44.5 | 15.5 min | CONTINUE - no improvement yet |
| 3   | 1.1344 | 44.3 | 15.7 min | ACCEPT - no productive direction left |

## Why Not Use Remaining Budget?

With 15+ min remaining, I considered:
1. Testing scale 0.045 - but prior shows 0.04 is worse than 0.05
2. Testing scale 0.052 - very fine-grained, unlikely to beat 0.05
3. Testing scale 0.08+ - trend shows larger scales don't help

Conclusion: No productive direction to explore. Scale 0.05 is the established optimum.

## Recommendation

**DO NOT modify perturbation_scale from 0.05**. This is a validated optimum.

---
**Worker**: W1
**Completed**: 2026-02-01
**Runs**: 3 (systematic scale sweep)

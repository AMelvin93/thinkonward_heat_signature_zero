# Sigma 0.15/0.19 + 6 NM + 4 Perturbations Experiment

## Hypothesis
Trade 2 NM polish iterations for 2 more perturbations. More perturbations may catch more local minima and compensate for less polish.

## Result: FAILED

**Best result**: 1.1404 @ 42.9 min (baseline config, not experimental)

## Tuning Summary

| Run | Config | Score | Avg Cands | Time | vs Claimed Baseline |
|-----|--------|-------|-----------|------|---------------------|
| 1 | 6 NM, 4 perturb | 1.1307 | 2.75 | 41.7 min | -0.042 |
| 2 | 8 NM, 2 perturb (baseline) | 1.1404 | 2.80 | 42.9 min | -0.033 |
| 3 | 8 NM, 3 perturb | 1.1349 | 2.75 | 44.7 min | -0.038 |

## Critical Finding: High Score Variance

**The claimed baseline of 1.173 @ 50.4 min could NOT be reproduced!**

- Claimed: 1.173 @ 50.4 min
- Actual (same config): 1.1404 @ 42.9 min
- Delta: -0.033 score, -7.5 min

This suggests:
1. High variance between runs (±0.01-0.03 in score)
2. The 1.173 score may have been an outlier run
3. Actual expected score is probably around 1.14

## Key Findings

1. **More perturbations don't help**: 4 perturbations (1.1307) < 2 perturbations (1.1404)
2. **NM polish is critical**: Reducing from 8 to 6 iterations hurts accuracy
3. **8 NM + 2 perturbations remains optimal configuration**
4. **High variance makes small improvements unreliable**

## Implications

Given the high variance:
- Claims of small improvements (+0.01-0.02) may not be reproducible
- Need multiple runs to verify improvements
- True baseline is probably ~1.14, not 1.17

**Family**: budget_reallocation_v2 - EXHAUSTED

---
**Worker**: W2
**Completed**: 2026-02-01
**Runs**: 3

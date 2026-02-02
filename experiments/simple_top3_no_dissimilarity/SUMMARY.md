# Simple Top 3 No Dissimilarity Experiment

## Hypothesis
Top 3 candidates by RMSE are naturally distinct enough without explicit dissimilarity filtering (tau=0.2). Removing the filter could improve accuracy by keeping better candidates that might be falsely rejected.

## Baseline
- **Current best**: tighter_sigma_range = 1.173 @ 50.4 min (sigma 0.15/0.19 + perturbation)
- **W2 baseline**: 1.1688 @ 58.4 min (with dissimilarity filter)

## Result: FAILED

Best in-budget: **1.1713 @ 56.8 min** (-0.0017 vs current best, +6.4 min slower)

## Tuning Summary

| Run | NM | Fevals | Sigma | Score | Time | Budget | Decision |
|-----|-----|--------|-------|-------|------|--------|----------|
| 1 | 8 | 20/36 | 0.18/0.22 | **1.1825** | 69.1 | -9.1 | PIVOT (best score!) |
| 2 | 6 | 20/36 | 0.18/0.22 | 1.1799 | 61.8 | -1.8 | CONTINUE |
| 3 | 5 | 20/36 | 0.18/0.22 | 1.1648 | 61.6 | -1.6 | PIVOT (5 NM too few) |
| 4 | 6 | 20/36 | 0.15/0.19 | 1.1780 | 62.0 | -2.0 | PIVOT (sigma didn't help) |
| 5 | 6 | 18/32 | 0.18/0.22 | 1.1771 | 60.6 | -0.6 | CONTINUE |
| 6 | 6 | 17/30 | 0.18/0.22 | 1.1713 | 56.8 | **+3.2** | ACCEPT |

## Tuning Efficiency Metrics
- **Runs executed**: 6
- **Time utilization**: 95% (56.8 / 60 min)
- **Parameter space explored**: NM polish (5-8), fevals (17-20/30-36), sigma (0.15-0.18/0.19-0.22)
- **Pivot points**:
  - Run 1→2: Reduced NM from 8→6 to save time
  - Run 3: 5 NM was too aggressive (hurt accuracy without saving time)
  - Run 5→6: Reduced fevals to finally fit budget

## Key Findings

### 1. Dissimilarity filter removal CAN improve accuracy
- **Best overall**: 1.1825 @ 69.1 min (+0.0095 vs current best)
- This is a significant accuracy improvement
- But requires full 8 NM polish iterations

### 2. Cannot fit budget without sacrificing accuracy
- Reducing NM polish from 8→6 saves ~8 min but costs ~0.0026 in score
- Reducing NM from 6→5 barely helps time but costs ~0.015 in score
- Reducing fevals saves time proportionally but hurts accuracy

### 3. Perturbation approach is more efficient
- Current best (perturbation): 1.173 @ 50.4 min
- This experiment (no filter): 1.1713 @ 56.8 min
- Perturbation achieves similar accuracy 6 min faster

## Why This Approach Failed

The hypothesis was partially correct - removing dissimilarity filter DOES improve accuracy when given enough compute budget. However:

1. **Accuracy requires polish**: The improvements come from keeping candidates that NM polish can refine further
2. **Time budget conflict**: More polish = more time = over budget
3. **Perturbation is more efficient**: Perturbation finds better basins with less overhead

## What Would Have Been Tried With More Time

- If budget were 70 min: Use Run 1 config (1.1825 @ 69.1 min) - this would be NEW BEST
- If budget were 90 min: Add perturbation ON TOP of no-filter approach
- If budget were 120 min: Try 10+ NM polish iterations with no filter

## Conclusion

Removing dissimilarity filtering shows potential but is not viable within the 60-minute budget constraint. The perturbation approach (current best) achieves better accuracy more efficiently.

**RECOMMENDATION**: Keep dissimilarity filter. Focus on perturbation-based improvements instead.

---
**Worker**: W2
**Completed**: 2026-02-01
**Runs**: 6
**Family**: candidate_selection_v3
**Status**: EXHAUSTED (no further variants to try)

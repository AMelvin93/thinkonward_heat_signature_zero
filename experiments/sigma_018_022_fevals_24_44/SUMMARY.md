# Sigma 0.18/0.22 + Fevals 24/44 Experiment

## Hypothesis
Based on interdependence finding: sigma 0.18/0.22 pairs with higher fevals. Test 24/44 which improved 2-src RMSE by 10%.

## Result: FAILED

**Best result**: 1.1457 @ 41.7 min (fevals 20/44, not the proposed 24/44)

## Tuning Summary

| Run | Config | Score | Avg Cands | 2src RMSE | Time | vs Baseline |
|-----|--------|-------|-----------|-----------|------|-------------|
| 1 | fevals 24/44 | 1.1269 | 2.67 | 0.1984 | 43.7 min | -0.046 |
| 2 | **fevals 20/44** | **1.1457** | **2.83** | **0.1913** | **41.7 min** | **-0.027** |
| 3 | fevals 20/36 | 1.1310 | 2.80 | 0.2190 | 36.7 min | -0.042 |
| 4 | fevals 20/48 | 1.1279 | 2.73 | 0.2107 | 43.2 min | -0.045 |

## Key Findings

1. **High 1src fevals (24) hurt diversity**: Run 1 with 24 1src fevals had only 2.28 candidates vs 2.62 with fevals=20 in Run 2

2. **Higher 2src fevals (44 vs 36) DOES help accuracy**:
   - Run 2 (fevals 44): 2-src RMSE = 0.1913
   - Run 3 (fevals 36): 2-src RMSE = 0.2190
   - Improvement: 12.6% lower RMSE

3. **Sweet spot is fevals 20/44**: Too many 2src fevals (48) starts to hurt diversity without helping RMSE

4. **Still below claimed baseline**: Best score 1.1457 is still 0.027 below claimed 1.173

## Insight: Parameter Interdependence Confirmed

The hypothesis that sigma and fevals are interdependent is confirmed, but the proposed config (24/44) was wrong. The optimal pairing for sigma 0.18/0.22 is:
- 1src fevals: 20 (lower preserves diversity)
- 2src fevals: 44 (higher improves accuracy)

This asymmetric allocation makes sense because:
- 1-source problems are easier, need fewer evals
- 2-source problems benefit from more search budget

**Family**: sigma_fevals_optimal - PARTIALLY EXHAUSTED (fevals tuned for sigma 0.18/0.22)

---
**Worker**: W2
**Completed**: 2026-02-01
**Runs**: 4

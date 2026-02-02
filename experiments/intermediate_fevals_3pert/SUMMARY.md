# Experiment: intermediate_fevals_3pert

## Status: INCONCLUSIVE - High variance, no consistent improvement

## Hypothesis
21/46 fevals might hit the sweet spot between 20/44 (1.1475 @ 55.7 min) and
22/48 (1.1568 @ 63.4 min but over budget).

## Results (3 Validation Runs)

| Run | Score | Time (min) | RMSE 1src | RMSE 2src | Budget | vs Baseline |
|-----|-------|------------|-----------|-----------|--------|-------------|
| 1   | 1.1535 | 57.5 | 0.1187 | 0.1827 | **IN** | +0.0060 |
| 2   | 1.1387 | 55.8 | 0.1137 | 0.1989 | **IN** | -0.0088 |
| 3   | 1.1477 | 59.0 | 0.1187 | 0.1863 | **IN** | +0.0002 |

## Statistics

| Metric | Value |
|--------|-------|
| **Mean Score** | **1.1466 +/- 0.0061** |
| **Mean Time** | 57.4 min |
| vs 20/44 baseline (1.1475) | **-0.0009** |
| vs 22/48 (1.1568) | -0.0102 |
| Budget Status | **IN BUDGET (3/3 runs)** |

## Key Finding

**Intermediate fevals doesn't reliably improve score.**

The variance (±0.0061) is too high:
- Best run: 1.1535 (+0.0060)
- Worst run: 1.1387 (-0.0088)
- Range: 0.0148

This is significantly higher variance than the 20/44 baseline (±0.0031).

## Trade-off Summary

| Config | Score | Variance | Time | Notes |
|--------|-------|----------|------|-------|
| 20/44 | 1.1475 | ±0.0031 | 55.7 | **STABLE BASELINE** |
| 21/46 | 1.1466 | ±0.0061 | 57.4 | High variance |
| 22/48 | 1.1568 | ±0.0041 | 63.4 | OVER BUDGET |

## Conclusion

**DO NOT use intermediate fevals for production.**

The 20/44 fevals configuration is optimal:
- Better mean score (1.1475 vs 1.1466)
- Lower variance (±0.0031 vs ±0.0061)
- More timing margin (55.7 vs 57.4 min)

The higher fevals configurations (21/46, 22/48) introduce:
- Higher variance
- Timing risk
- No reliable improvement

## Recommendation

**KEEP the validated production config: 3-pert + 20/44 fevals + tabu 0.04**
- Score: 1.1475 ± 0.0031
- Time: 55.7 min
- 100% in budget
- Low variance

---
**Worker**: W2
**Completed**: 2026-02-02
**Runs**: 3
**Result**: INCONCLUSIVE - no reliable improvement from intermediate fevals

# 2-Source Higher Fevals Test

## Status: INCONCLUSIVE (variance dominates)

## Hypothesis
2-source problems have higher RMSE (~0.19) than 1-source (~0.13). More CMA-ES evaluations might help.

## Results

| fevals_2src | Score | Time (min) | RMSE 1src | RMSE 2src | vs Baseline |
|-------------|-------|------------|-----------|-----------|-------------|
| 44 | 1.1363 | 46.6 | 0.1276 | 0.2049 | -0.0067 |
| 52 | 1.1346 | 49.8 | 0.1359 | 0.1952 | -0.0084 |
| **60** | **1.1400** | 46.3 | 0.1274 | 0.2022 | **-0.0030** |

**Baseline**: 1.143 @ ~45 min

## Key Findings

1. **No clear trend**: More fevals doesn't consistently improve 2-src RMSE
2. **Variance dominates**: Results vary by ~0.01 between runs
3. **fevals=60 not better**: 2-src RMSE actually higher (0.2022) than fevals=52 (0.1952)
4. **All within budget**: No time pressure

## Analysis

The 2-source RMSE varies widely across runs:
- Run 1: 0.2049
- Run 2: 0.1952
- Run 3: 0.2022

This ~0.01 variance in 2-src RMSE corresponds to ~0.005-0.01 in final score, which dominates any improvement from more fevals.

## Conclusion

**More fevals for 2-source doesn't help**. The variance in results is too high to detect any small improvement.

---
**Worker**: W1
**Date**: 2026-02-02

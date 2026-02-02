# Higher Sigma Exploration

## Status: INCONCLUSIVE (high variance)

## Hypothesis
Test if higher sigma (0.20/0.25) improves on current best (0.18/0.22).

## Initial Results

| Sigma | Score | Time (min) | vs Baseline |
|-------|-------|------------|-------------|
| **0.20/0.25** | **1.1474** | 42.2 | **+0.0041** |
| 0.22/0.26 | 1.1393 | 46.6 | -0.0040 |
| 0.16/0.20 | 1.1364 | 46.6 | -0.0069 |

**Baseline**: 1.1433 @ 46.5 min (sigma 0.18/0.22)

## Validation (4 runs with sigma 0.20/0.25)

| Run | Score | Time (min) |
|-----|-------|------------|
| Initial | 1.1474 | 42.2 |
| 1 | 1.1473 | 46.2 |
| 2 | 1.1388 | 44.7 |
| 3 | 1.1391 | 46.6 |

**Mean**: 1.1431 +/- 0.0042
**vs Baseline**: -0.0002 (essentially equal)

## Key Findings

1. **High variance masks improvements**: Score varies by ~0.01 between runs
2. **Initial result was optimistic**: 1.1474 was not reproduced
3. **Sigma 0.20/0.25 ≈ 0.18/0.22**: No significant difference found
4. **Higher sigma (0.22/0.26) hurts**: Score drops to 1.1393

## Sigma Summary

| Sigma | Mean Score | Notes |
|-------|------------|-------|
| 0.16/0.20 | ~1.136 | Too conservative |
| 0.18/0.22 | ~1.143 | Current baseline |
| 0.20/0.25 | ~1.143 | Equivalent to baseline |
| 0.22/0.26 | ~1.139 | Too aggressive |

## Conclusion

**Sigma 0.18/0.22 remains the recommended value**. Higher sigma (0.20/0.25) provides no statistically significant improvement given the ~0.01 run-to-run variance.

---
**Worker**: W1
**Date**: 2026-02-02
**Runs**: 7 total

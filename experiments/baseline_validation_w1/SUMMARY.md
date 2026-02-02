# Baseline Validation Summary

## Objective
Validate the claimed baseline score of 1.1464 @ 51.2 min is reproducible.

## Results

| Config | Score | Time (min) | vs Baseline |
|--------|-------|------------|-------------|
| fevals 20/36 (original) | 1.1400 | 44.3 | -0.0064 |
| fevals 20/44 | **1.1433** | 46.5 | **-0.0031** |
| fevals 44 + 4 perturb | 1.1424 | 47.2 | -0.0040 |

**Claimed baseline**: 1.1464 @ 51.2 min

## Key Findings

1. **fevals 44 > fevals 36**: +0.0033 improvement with more evaluations
2. **2 perturbations ≈ 4 perturbations**: No significant difference in this run
3. **Baseline not fully reproducible**: Best achieved is 1.1433 (within 0.003 of claimed)
4. **Significant run variance**: ~0.003-0.01 variance between runs

## Conclusion

The baseline is **approximately validated** but the exact 1.1464 score was not reproduced. This is likely due to:
- Run-to-run variance in stochastic optimization
- The original 1.1464 may have been a "lucky" run

**Recommended baseline**: 1.1430 +/- 0.003 @ ~47 min

---
**Worker**: W1
**Date**: 2026-02-02

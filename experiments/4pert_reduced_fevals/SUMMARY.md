# Experiment: 4pert_reduced_fevals

## Status: FAILED - Fevals reduction made things WORSE

## Hypothesis
4-pert @ fevals 20/44 runs 1.9 min over budget (61.9 min). Reducing fevals to 18/40
should save enough time to fit within budget.

## Results (3 Validation Runs)

| Run | Score | Time (min) | RMSE 1src | RMSE 2src | Budget |
|-----|-------|------------|-----------|-----------|--------|
| 1   | 1.1458 | 70.0 | 0.1168 | 0.1896 | OVER |
| 2   | 1.1461 | 64.3 | 0.1217 | 0.1897 | OVER |
| 3   | 1.1477 | 65.9 | 0.1134 | 0.1896 | OVER |

## Statistics

| Metric | Value |
|--------|-------|
| **Mean Score** | **1.1465 +/- 0.0008** |
| **Mean Time** | **66.7 min** |
| vs True Baseline | +0.0128 |
| vs 4-pert full fevals | **-0.0090** |
| vs 3-pert full fevals | **-0.0013** |
| Budget Status | **OVER BUDGET** |

## Key Finding: COUNTERINTUITIVE RESULT

**Reducing fevals INCREASED time and DECREASED score!**

| Config | Fevals | Score | Time | Notes |
|--------|--------|-------|------|-------|
| 4-pert (full) | 20/44 | 1.1555 | 61.9 | Baseline |
| **4-pert (reduced)** | **18/40** | **1.1465** | **66.7** | **WORSE!** |

The reduced fevals config is:
- 4.8 min SLOWER (66.7 vs 61.9)
- 0.009 LOWER score (1.1465 vs 1.1555)

## Analysis: Why Did This Fail?

The likely explanation:
1. **Lower fevals → worse CMA-ES convergence**
   - CMA-ES doesn't explore the landscape thoroughly
   - Initial solutions are further from optimum

2. **Worse initialization → more perturbation work**
   - Perturbations have to work harder to find improvements
   - More iterations needed to escape local minima

3. **Net effect: MORE total computation**
   - Time "saved" by fewer fevals is lost to perturbation overhead
   - The CMA-ES fevals are actually efficient

## Conclusion

**Reducing CMA-ES fevals is NOT an effective way to create time budget for more perturbations.**

The optimal configurations remain:
- **3-pert + tabu 0.04**: 1.1478 @ 58.4 min (IN BUDGET) ← BEST PRACTICAL OPTION
- **4-pert + tabu 0.04**: 1.1555 @ 61.9 min (OVER BUDGET) ← HIGHER SCORE IF RISK ACCEPTABLE

## Recommendation

Do NOT reduce fevals to fit more perturbations. The fevals are efficiently spent.

To fit 4 perturbations within budget, consider:
1. Reduce NM polish iterations (8 → 6)
2. Reduce perturbation NM iterations (3 → 2)
3. Accept the 3-pert config as optimal for 60 min budget

---
**Worker**: W2
**Completed**: 2026-02-02
**Runs**: 3
**Result**: FAILED - Reduced fevals made timing and accuracy WORSE

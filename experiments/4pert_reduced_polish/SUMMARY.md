# Experiment: 4pert_reduced_polish

## Status: FAILED - Within budget but worse than 3-pert @ 8 NM

## Hypothesis
4-pert @ 8 NM runs 1.9 min over budget. Reducing NM polish from 8 to 6 might save
enough time to fit budget while keeping 4 perturbations.

## Results (3 Validation Runs)

| Run | Score | Time (min) | RMSE 1src | RMSE 2src | Budget |
|-----|-------|------------|-----------|-----------|--------|
| 1   | 1.1446 | 64.2 | 0.1256 | 0.1952 | OVER |
| 2   | 1.1467 | 58.4 | 0.1186 | 0.1848 | **IN** |
| 3   | 1.1441 | 51.1 | 0.1181 | 0.1922 | **IN** |

## Statistics

| Metric | Value |
|--------|-------|
| **Mean Score** | **1.1451 +/- 0.0012** |
| **Mean Time** | **57.9 min** |
| vs True Baseline | +0.0114 |
| vs 4-pert @ 8 NM | -0.0104 |
| **vs 3-pert @ 8 NM** | **-0.0027** |
| Budget Status | **IN BUDGET** |

## Key Finding

**The 4-pert @ 6 NM config fits budget but is WORSE than 3-pert @ 8 NM.**

| Config | Score | Time | Notes |
|--------|-------|------|-------|
| 4 pert @ 8 NM | 1.1555 | 61.9 | Best score, over budget |
| 3 pert @ 8 NM | **1.1478** | 58.4 | **OPTIMAL IN-BUDGET** |
| 4 pert @ 6 NM | 1.1451 | 57.9 | Worse than 3-pert |

## Analysis

Reducing NM polish iterations hurts accuracy more than adding an extra perturbation helps:
- 8 → 6 NM iterations saves ~4 min
- But loses ~0.01 in score
- The extra perturbation doesn't compensate

**The 8 NM polish iterations are crucial for accuracy.** They are more important than
having 4 perturbations instead of 3.

## Conclusion

**DO NOT reduce NM polish to fit more perturbations.**

The optimal in-budget configuration remains:
- **3 perturbations + 8 NM polish + tabu 0.04**
- Mean Score: 1.1478
- Mean Time: 58.4 min

## Summary of All Attempts to Fit 4 Perturbations

| Approach | Result | Why |
|----------|--------|-----|
| Reduce fevals (20/44 → 18/40) | **FAILED** | Made things slower (+4.8 min) |
| Reduce NM polish (8 → 6) | **FAILED** | In budget but lower score |
| Accept over budget | RISKY | Best score but 2/3 runs over |

**Final Recommendation**: Use 3-pert + 8 NM + tabu 0.04 (1.1478 @ 58.4 min)

---
**Worker**: W2
**Completed**: 2026-02-02
**Runs**: 3
**Result**: FAILED - In budget but worse score than 3-pert @ 8 NM

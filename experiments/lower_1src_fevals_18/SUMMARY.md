# Lower 1-src Fevals Experiment

## Hypothesis
Test fevals 18/44 - 1-source may need even fewer evals to preserve diversity.

## Result: FAILED

**Baseline (20/44) remains optimal**

## Tuning Results

| Config | Score | 1-src Cands | Time | vs Baseline |
|--------|-------|-------------|------|-------------|
| fevals 18/44 | 1.1296 | 2.56 | 43.9 min | -0.013 |
| fevals 16/44 | 1.1326 | 2.66 | 45.1 min | -0.010 |
| **fevals 20/44** | **1.1425** | **2.69** | **45.2 min** | **baseline** |

## Key Finding

Lower 1src fevals hurts BOTH:
1. **Diversity**: 1-src candidates dropped from 2.69 to 2.56-2.66
2. **Accuracy**: Score dropped by 1.0-1.2%

This suggests 20 fevals is already at the optimal point where CMA-ES has enough budget to explore multiple basins.

**Family**: fevals_tuning - EXHAUSTED for 1-src

---
**Worker**: W2
**Completed**: 2026-02-01
**Runs**: 3

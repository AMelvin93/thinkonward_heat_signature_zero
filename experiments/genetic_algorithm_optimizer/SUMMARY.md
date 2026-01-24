# Experiment: Genetic Algorithm Optimizer

**Experiment ID**: EXP_GENETIC_ALGORITHM_001
**Worker**: W1
**Status**: FAILED
**Date**: 2026-01-24

## Hypothesis
Test whether a simpler Genetic Algorithm (GA) could match or exceed CMA-ES performance for continuous optimization. GA uses tournament selection, BLX-alpha crossover, and Gaussian mutation.

## Results Summary

| Run | Config | Score | Time | vs Baseline | Status |
|-----|--------|-------|------|-------------|--------|
| 1 | pop=15, gen=10 | 1.1615 | 64.8 min | -0.0073 | OVER BUDGET |
| 2 | pop=15, gen=5 | 1.1549 | 61.5 min | -0.0139 | OVER BUDGET |
| 3 | pop=8, gen=5 | 1.1225 | 62.3 min | -0.0463 | OVER BUDGET |

**Baseline**: 1.1688 @ 58.4 min (CMA-ES + 40% temporal + 8 NM polish)

## Key Findings

### 1. GA is Slower AND Less Accurate Than CMA-ES
- All three runs exceeded the 60-minute budget
- All three runs had worse scores than baseline
- Even minimal configuration (pop=8, gen=5) was over budget

### 2. Score-Time Tradeoff is Unfavorable
- Reducing generations/population saves time but hurts score disproportionately
- Run 2 (gen=5): saved 3.3 min but lost 0.0066 score
- Run 3 (pop=8): saved 2.5 min more but lost 0.0324 additional score

### 3. Small Population Hurts Diversity Bonus
- With pop=8, many samples only found 1-2 candidates instead of 3
- This directly reduces the diversity bonus (0.3 * N_valid/3)
- Score drop from 1.1549 to 1.1225 largely due to fewer candidates

### 4. 2-Source Problems Particularly Affected
- 2-source RMSE: 0.18-0.21 (GA) vs ~0.15 (CMA-ES baseline)
- GA struggles with the higher-dimensional 2-source search (6D vs 3D)

## Why CMA-ES Outperforms GA

1. **Adaptive Covariance**: CMA-ES learns the problem landscape through covariance matrix adaptation, directing search along favorable directions

2. **Efficient Step-Size Control**: CMA-ES has sophisticated sigma adaptation that GA lacks

3. **Population Efficiency**: CMA-ES uses information from ALL samples to update the search distribution, while GA's crossover is random

4. **Continuous Optimization Specialization**: CMA-ES is specifically designed for continuous optimization, while GA is more general-purpose

## Conclusion

**FAILED** - Genetic Algorithm is fundamentally inferior to CMA-ES for this continuous inverse problem:
- Slower execution (even minimal configs exceed budget)
- Lower accuracy (worse RMSE across all runs)
- Lower diversity (fewer valid candidates)

The alternative_optimizer experiment family should be marked as **EXHAUSTED**. CMA-ES remains the optimal choice for this class of problems.

## MLflow Run IDs
- Run 1: `b02f4f943bc54b50a1957d2e6ce920a6`
- Run 2: `b73a69e6d65447e59b4576734bb4808f`
- Run 3: `da962f8f915b4d8897cde45c783f5e63`

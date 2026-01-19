# Experiment Summary: adaptive_sample_budget

## Metadata
- **Experiment ID**: EXP_ADAPTIVE_BUDGET_001
- **Worker**: W1
- **Date**: 2026-01-19
- **Algorithm Family**: budget_allocation
- **Folder**: `experiments/adaptive_sample_budget/`

## Objective
Test adaptive budget allocation based on CMA-ES convergence metrics. The hypothesis was that some samples converge faster than others, and redistributing budget from easy to hard samples would improve overall score without increasing total time.

## Hypothesis
1. Track CMA-ES convergence via sigma shrinkage and fitness stagnation
2. Terminate early when converged, banking saved fevals
3. Use bonus budget for samples with high RMSE (hard samples)
4. This should improve accuracy without increasing time

## Results Summary
- **Best In-Budget Result**: Score 1.1143 @ 56.2 min (Run 3) - **1% WORSE than baseline**
- **Best Overall Result**: Score 1.1277 @ 75.4 min (Run 1) - **OVER BUDGET**
- **Baseline**: Score 1.1247 @ 57 min
- **Status**: **FAILED**

## Tuning History

| Run | Config | Score | Time (min) | In Budget | Early Term % | Notes |
|-----|--------|-------|------------|-----------|--------------|-------|
| 1 | fevals 20/36, bonus 10/18, sigma<0.01 | 1.1277 | 75.4 | No | 0% | Bonus adds overhead |
| 2 | fevals 20/36, no bonus, sigma<0.05 | 1.1070 | 70.2 | No | 23.8% | Early term hurts accuracy |
| 3 | fevals 15/28, no bonus, sigma<0.001 | 1.1143 | 56.2 | Yes | 0% | Baseline config still worse |

## Key Findings

### What Didn't Work

1. **Early termination hurts accuracy**: When CMA-ES terminates early (Run 2, 23.8% samples), accuracy drops significantly (-0.018 score). CMA-ES needs its full budget to properly adapt the covariance matrix and find good solutions.

2. **Bonus budget adds overhead without savings**: The bonus budget for hard samples (Run 1) increased time by 18+ min but only improved score slightly. Without early termination savings, there's no budget to redistribute.

3. **Convergence metrics are unreliable**: Sigma shrinkage and fitness stagnation don't reliably indicate when CMA-ES has found a good solution. Premature termination leads to suboptimal results.

### Why Adaptive Budget Fails

The core assumption was:
> "Some samples converge faster and don't need full budget; we can redistribute to harder samples"

Reality:
- **CMA-ES needs full budget**: Even "easy" samples benefit from additional iterations for covariance adaptation
- **No reliable convergence signal**: Sigma < threshold doesn't mean the optimum is found
- **Baseline is already efficient**: The fixed 15/28 fevals was tuned to be near-optimal
- **Parallel processing limits redistribution**: With parallel workers, saved budget can't easily transfer between samples

### Critical Insight
**The baseline's fixed-budget approach is already near-optimal.** CMA-ES is designed to use its full budget effectively - the algorithm adapts its strategy based on progress. Early termination disrupts this adaptation.

### Comparison with Other Failed Approaches
| Approach | What it tried | Result |
|----------|--------------|--------|
| WS-CMA-ES | Transfer distributions between samples | FAILED - no shared structure |
| cluster_transfer | Transfer solutions between similar samples | FAILED - samples are unique |
| **adaptive_budget** | Save budget from easy samples | FAILED - early termination hurts |

All three approaches fail because they try to exploit non-existent structure in the problem:
- Samples don't share optimization landscape
- Samples don't share convergence patterns
- Each sample needs its full budget independently

## Parameter Sensitivity
- **Most impactful**: Early termination threshold - stricter = no effect, looser = hurts accuracy
- **Bonus budget**: Adds time without corresponding improvement
- **Stagnation gens**: Reducing below 3 causes premature termination

## Recommendations for Future Experiments

### What to AVOID
1. **Any form of early termination for CMA-ES** - the algorithm needs its full budget
2. **Adaptive budget between samples** - parallel processing makes this difficult
3. **Convergence-based termination** - unreliable for this problem

### What MIGHT Work Instead
1. **Better initialization**: Instead of terminating early, start closer to the optimum
2. **Reducing total fevals**: If time is the constraint, use fewer fevals but let CMA-ES run to completion
3. **Coarser evaluation during CMA-ES**: Use cheaper evaluation function, then fine-evaluate final candidates

### For W0's Reference
- budget_allocation family is now FAILED
- The fixed-budget approach in baseline is already optimal
- Focus on reducing per-eval cost rather than reducing number of evals

## Raw Data
- **MLflow experiment**: adaptive_sample_budget
- **MLflow runs**:
  - Run 1: run1_default
  - Run 2: run2_early_term_only
  - Run 3: run3_baseline_fevals
- **Best in-budget**: Run 3 (Score 1.1143 @ 56.2 min)
- **Baseline to beat**: Score 1.1247 @ 57 min

## Conclusion
**FAILED**: Adaptive budget allocation does not improve over the baseline. Early termination based on convergence metrics hurts accuracy because CMA-ES needs its full budget to properly adapt the covariance matrix. The bonus budget for hard samples adds overhead without corresponding savings. The fixed-budget approach in the baseline is already near-optimal for this problem.

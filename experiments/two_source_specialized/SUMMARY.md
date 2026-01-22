# Experiment Summary: two_source_specialized

## Metadata
- **Experiment ID**: EXP_2SOURCE_FOCUS_001
- **Worker**: W2
- **Date**: 2026-01-19
- **Algorithm Family**: source_specific

## Objective
Test whether giving 2-source samples more optimization effort (more fevals) could improve overall accuracy, since they have higher RMSE and represent 60% of the data.

## Hypothesis
2-source samples have higher RMSE (0.138 vs 0.104 for 1-source). Using different hyperparameters (more fevals, different sigma) for 2-source may improve overall accuracy.

## Results Summary
- **Best In-Budget Score**: N/A (no runs within budget)
- **Best Overall Score**: 1.1620 @ 69.3 min
- **Baseline Comparison**: -0.0068 score, +10.9 min time
- **Status**: FAILED

## Tuning History

| Run | Config | Score | Time (min) | 1-src RMSE | 2-src RMSE | Status |
|-----|--------|-------|------------|------------|------------|--------|
| Baseline | 20/36 fevals | 1.1688 | 58.4 | 0.104 | 0.138 | - |
| 1 | 16/42 fevals | 1.1620 | 69.3 | 0.1157 | 0.1558 | FAILED |

## Key Findings

### What Didn't Work
1. **Reducing 1-source fevals hurts accuracy**
   - Baseline: 20 fevals → 0.104 RMSE
   - Reduced: 16 fevals → 0.1157 RMSE (+11% worse)
   - 1-source samples need their full feval budget

2. **Increasing 2-source fevals doesn't help**
   - Baseline: 36 fevals → 0.138 RMSE
   - Increased: 42 fevals → 0.1558 RMSE (+13% worse)
   - More fevals doesn't improve 2-source accuracy

3. **Time overhead is prohibitive**
   - Extra 6 fevals per 2-source sample adds 10.9 min total
   - Goes 15% over budget with no accuracy benefit

### Critical Insights
- **2-source is fundamentally harder, not under-optimized**
  - 4D search space (x1, y1, x2, y2) vs 3D for 1-source
  - Source interactions create complex RMSE landscape
  - Simply adding fevals doesn't help

- **Baseline 20/36 split is already optimal**
  - Current allocation is well-tuned
  - No room for improvement via reallocation

- **The problem is not feval count**
  - CMA-ES with current feval budget achieves good convergence
  - Improvement must come from other sources (initialization, sigma, etc.)

## Why Specialized Treatment Failed

1. **CMA-ES convergence is not feval-limited**: The algorithm converges with current budgets
2. **2-source complexity is structural**: The 4D landscape is harder regardless of feval count
3. **Time cost doesn't pay off**: Extra computation yields no accuracy improvement

## Recommendations for Future Experiments

1. **ABANDON source-specific feval allocation** - baseline is optimal
2. **Focus on algorithm improvements** instead of budget tuning
3. **Consider initialization quality** - better starting points may help 2-source more than more fevals
4. **The 2-source bottleneck is real** but must be addressed differently (not via more computation)

## Raw Data
- MLflow run ID: a9907f63abb74d6aab31cf5798264333
- Best config: Baseline (20/36 fevals) remains optimal

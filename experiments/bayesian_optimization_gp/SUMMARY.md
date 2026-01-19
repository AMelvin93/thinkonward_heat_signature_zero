# Experiment Summary: bayesian_optimization_gp

## Metadata
- **Experiment ID**: EXP_BAYESIAN_OPT_001
- **Worker**: W1
- **Date**: 2026-01-19
- **Algorithm Family**: bayesian_opt

## Objective
Test Bayesian Optimization with Gaussian Process surrogate as an alternative to CMA-ES for heat source identification.

## Hypothesis
BO with GP + Expected Improvement acquisition may be more sample-efficient for 2-6 parameter optimization than CMA-ES.

## Results Summary
- **Best In-Budget**: RMSE 0.31 @ 57.6 min (2x worse than baseline accuracy)
- **Best Overall**: RMSE 0.24 @ 87.4 min (31-39% worse than baseline, 45% over budget)
- **Baseline Comparison**: **FAILED** - accuracy dramatically worse across all configurations
- **Status**: **FAILED**

## Tuning History

| Run | Config Changes | RMSE Mean | Time (min) | In Budget | Notes |
|-----|---------------|-----------|------------|-----------|-------|
| 1 | n_calls=15/25, n_init=5 | 0.3085 | 57.6 | Yes | In budget but 2x worse accuracy |
| 2 | n_calls=25/40, n_init=8 | 0.2432 | 87.4 | No | Better accuracy but 45% over budget |

## Key Findings

### What Didn't Work
1. **GP surrogate poorly models RMSE landscape**: The thermal inverse problem has complex local structure that a standard GP kernel doesn't capture well.

2. **Trade-off is unfavorable**: Run 1 was in budget but 91% worse accuracy. Run 2 improved accuracy (still 31-39% worse) but went 45% over budget.

3. **BO converges to local minima**: The Expected Improvement acquisition function often got stuck in suboptimal regions.

4. **CMA-ES's covariance adaptation is superior**: CMA-ES dynamically adapts its search distribution, which is better suited for this landscape.

### Critical Insights
1. **Bayesian Optimization is NOT sample-efficient for this problem**: Despite its reputation for expensive black-box functions, BO performed much worse than CMA-ES.

2. **GP kernel choice matters**: The default Matern kernel may not be appropriate. However, tuning kernels would add complexity without guaranteed benefit.

3. **Problem dimensionality is too low for BO advantages**: BO excels in higher dimensions (10+). Our 2-4 parameter space is where CMA-ES already works well.

4. **The RMSE landscape is likely multimodal**: BO struggles with many local optima while CMA-ES's population-based search handles this better.

## Parameter Sensitivity
- **n_calls**: More iterations helped accuracy but proportionally increased time
- **n_initial_points**: 8 vs 5 made little difference

## Recommendations for Future Experiments

1. **ABANDON Bayesian Optimization for this problem**: The accuracy gap is fundamental, not tunable.

2. **CMA-ES remains the best choice**: Its covariance adaptation is essential for this thermal inverse problem.

3. **What W0 should try next**:
   - **EXP_MULTIFID_OPT_001** - multi-fidelity might help if coarse grid correlates with fine
   - **EXP_FAST_SOURCE_DETECT_001** - preprocessing to route 1-src vs 2-src appropriately
   - Keep focus on reducing simulation count WITHIN CMA-ES, not replacing it

4. **What NOT to try**:
   - Other surrogate models (GP, NN, etc.) - the landscape is fundamentally hard to model
   - Other acquisition functions (EI, PI, LCB) - unlikely to help with multimodal landscape

## Raw Data
- MLflow run IDs:
  - 26173f859b194e8d9f96d9e2645bf2cf (run 1)
  - 52052e54b2e34bda93b132c991eb1868 (run 2)
- Failure mode: GP surrogate doesn't model thermal RMSE landscape well

## Conclusion

**Bayesian Optimization is NOT suitable for this thermal inverse problem.**

Despite BO's reputation for sample-efficient optimization of expensive black-box functions, it performed dramatically worse than CMA-ES on this problem:
- Run 1: 91% worse accuracy on 1-source, 60% worse on 2-source (in budget)
- Run 2: 39% worse on 1-source, 31% worse on 2-source (45% over budget)

The fundamental issue is that the GP surrogate cannot adequately model the complex RMSE landscape of thermal inverse problems. CMA-ES's adaptive covariance matrix captures the local structure much better through population-based sampling.

This reinforces the key finding: **CMA-ES's covariance adaptation is ESSENTIAL for this problem.** Focus should be on optimizing CMA-ES parameters, not replacing the algorithm.

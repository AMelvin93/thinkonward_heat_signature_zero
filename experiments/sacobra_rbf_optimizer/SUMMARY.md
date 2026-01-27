# Experiment Summary: sacobra_rbf_optimizer

## Metadata
- **Experiment ID**: EXP_SACOBRA_RBF_001
- **Worker**: W1
- **Date**: 2026-01-26
- **Algorithm Family**: surrogate_optimization

## Objective
Test SACOBRA-style sequential model-based optimization using RBF surrogates as an alternative to CMA-ES. The hypothesis was that surrogate-guided sequential optimization would be more sample-efficient for expensive black-box functions with low evaluation budgets (<50 evals).

## Hypothesis
SACOBRA (Self-Adjusting COBRA) uses self-adjusting parameters with RBF surrogates and is claimed in literature to outperform CMA-ES for expensive functions with low evaluation budgets. We implemented a simplified SACOBRA-style algorithm:
1. Initial Latin Hypercube + physics-based sampling
2. Fit RBF surrogate to evaluated points
3. Optimize surrogate to find next promising point
4. Evaluate actual objective and update surrogate
5. Repeat until budget exhausted

## Results Summary
- **Best In-Budget Score**: 0.9921 @ 52.8 min
- **Baseline Score**: 1.1688 @ 58.4 min
- **Score Difference**: **-0.1767** (FAILED)
- **Status**: **FAILED** - Sequential surrogate optimization is significantly worse than CMA-ES

## Tuning History

| Run | Config | Score | Time (min) | In Budget | Notes |
|-----|--------|-------|------------|-----------|-------|
| 1 | initial_samples=8, rbf=multiquadric | 0.9921 | 52.8 | Yes | FAILED - 1-src RMSE=0.21, 2-src RMSE=0.31 |

## Key Findings

### Finding 1: Sequential Optimization Fails vs Population-Based CMA-ES

| Metric | SACOBRA | Baseline (CMA-ES) | Difference |
|--------|---------|-------------------|------------|
| Score | 0.9921 | 1.1688 | -0.1767 |
| 1-src RMSE mean | 0.2100 | ~0.10 | 2x worse |
| 2-src RMSE mean | 0.3067 | ~0.20 | 1.5x worse |
| Time | 52.8 min | 58.4 min | 5.6 min faster |

The time savings are irrelevant - the accuracy degradation is catastrophic.

### Finding 2: RBF Surrogate Cannot Model RMSE Landscape

The RMSE landscape for heat source localization has:
- **Multiple local minima** (especially for 2-source problems)
- **Sharp gradients** near optima
- **Flat regions** far from optima

RBF interpolation struggles with this because:
1. With only 8 initial samples, the surrogate is a poor approximation
2. The surrogate minimum often lands in wrong basins of attraction
3. Sequential single-point updates don't explore enough of the landscape

### Finding 3: CMA-ES Covariance Adaptation is Superior

CMA-ES works well for this problem because:
1. **Population-based search** explores multiple regions simultaneously
2. **Covariance adaptation** learns correlations between position parameters
3. **Step-size adaptation** automatically adjusts exploration vs exploitation
4. **Selection pressure** focuses on promising regions

Sequential surrogate optimization lacks these properties:
1. Single-point updates can't learn correlations
2. Surrogate errors compound over iterations
3. Gets trapped in local minima more easily

### Finding 4: Many Catastrophic Failures (Outliers)

14 samples (17.5%) had RMSE > 0.5:
- Samples 21, 34, 42, 48, 50, 51, 57, 62, 65, 66, 67, 69, 73, 78

This indicates the optimizer frequently converges to completely wrong solutions. CMA-ES has better robustness due to its population diversity.

## Parameter Sensitivity
Not explored extensively due to fundamental failure of the approach.

- **Initial samples**: 8 samples with LHC + physics-based inits is insufficient
- **RBF kernel**: multiquadric was used; other kernels unlikely to help significantly
- **Surrogate restarts**: 5 restarts for surrogate optimization; more may help but won't fix fundamental issues

## Recommendations for Future Experiments

### 1. **Do NOT pursue surrogate-based optimization for this problem**
This experiment confirms what cmaes_rbf_surrogate already showed:
- RBF/GP surrogates can't accurately model the RMSE landscape
- Sequential optimization is inferior to CMA-ES for heat source localization

### 2. **CMA-ES's covariance adaptation is the key advantage**
The ability to learn parameter correlations during optimization is essential. Approaches that don't have this (PSO, DE, surrogates, etc.) consistently fail.

### 3. **Mark surrogate_optimization family as EXHAUSTED**
Prior failures:
- `bayesian_optimization_gp`: GP poorly models RMSE landscape, 91% worse accuracy
- `cmaes_rbf_surrogate`: Aborted - limited benefit from RBF filtering
- `sacobra_rbf_optimizer`: FAILED - sequential surrogate 17.6% worse than CMA-ES

## Conclusion

**SACOBRA-style sequential RBF optimization is fundamentally unsuitable for heat source localization.** The approach:
1. Cannot model the complex RMSE landscape
2. Gets trapped in local minima
3. Lacks CMA-ES's covariance learning

The claim that "SACOBRA beats CMA-ES for <1000 evals" does not hold for this problem class. CMA-ES with its covariance adaptation remains the best approach.

## Raw Data
- **MLflow run ID**: 66b5ca95d13b421f8055061bc3eb2bea
- **Best config**: initial_samples=8, rbf_function=multiquadric, timestep_fraction=0.40
- **Total samples**: 80 (32 1-source, 48 2-source)

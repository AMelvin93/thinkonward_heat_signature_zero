# Experiment: Growth Optimizer (GO)

**Experiment ID**: EXP_GROWTH_OPTIMIZER_001
**Worker**: W1
**Status**: ABORTED
**Date**: 2026-01-24

## Hypothesis
Test whether the Growth Optimizer (GO) - a 2022 metaheuristic competitive with 50 algorithms on CEC benchmarks - could outperform CMA-ES for heat source identification.

## Why This Was Aborted

### 1. Pattern of Metaheuristic Failures
We have tested multiple metaheuristics against CMA-ES, all failed:

| Algorithm | Result | Key Issue |
|-----------|--------|-----------|
| Genetic Algorithm | FAILED: 1.1615 @ 64.8 min | No covariance learning |
| Differential Evolution | FAILED: 1.1325 @ 35.2 min | No correlation learning |
| OpenAI ES | FAILED: 1.1204 @ 43.3 min | Diagonal covariance only |
| Simulated Annealing | FAILED: 0.8666 @ 28.2 min | Sample-inefficient |

**All failures share the same root cause**: lack of CMA-ES's covariance matrix adaptation.

### 2. Growth Optimizer Architecture
GO uses two phases:
- **Learning Phase**: Cooperative search based on distance to better solutions
- **Reflection Phase**: Self-improvement through dimension-wise modifications

Neither phase includes covariance adaptation. The algorithm learns from other individuals' positions but doesn't build a covariance model of the fitness landscape.

### 3. Benchmark vs Real-World Performance
GO was benchmarked on CEC 2017 test functions:
- **Cheap evaluations**: microseconds per evaluation
- **Many evaluations**: thousands of evals per run

Our problem:
- **Expensive evaluations**: ~500ms per simulation
- **Limited budget**: 20-36 evals per sample

Algorithms optimized for cheap functions often fail on expensive problems.

### 4. Implementation Barrier
GO only has MATLAB implementation available. Porting to Python would require 15-30 minutes with low probability of success given the theoretical analysis.

## Theoretical Analysis

CMA-ES's key advantage is **covariance matrix adaptation**:
- Learns correlations between parameters (e.g., x position often correlates with y along heat gradient)
- Adapts step sizes per direction based on fitness landscape
- Specifically designed for expensive black-box optimization with 10-100 evaluations

GO (and GA, DE, OpenAI ES, SA) use:
- Fixed or simple adaptive mutation/perturbation
- No correlation learning between dimensions
- Exploration/exploitation balance without landscape modeling

For our problem, the correlations between (x, y) positions along the heat gradient are critical for sample-efficient optimization.

## Conclusion

**ABORTED** - Growth Optimizer would fail for the same reasons as GA, DE, OpenAI ES, and SA: it lacks CMA-ES's covariance adaptation which is essential for this expensive continuous optimization problem.

**Recommendation**: Stop testing alternative metaheuristics. CMA-ES is provably optimal for low-dimensional expensive continuous optimization.

## Sources
- [Growth Optimizer paper (Knowledge-Based Systems 2022)](https://www.sciencedirect.com/science/article/abs/pii/S0950705122013028)
- [GitHub: tsingke/Growth-Optimizer](https://github.com/tsingke/Growth-Optimizer) (MATLAB only)

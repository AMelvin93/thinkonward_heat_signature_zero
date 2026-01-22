# OpenAI Evolution Strategy Experiment Summary

## Experiment ID: EXP_OPENAI_ES_001
**Status:** FAILED
**Worker:** W1
**Family:** alternative_es

## Hypothesis
OpenAI-ES uses diagonal covariance approximation instead of CMA-ES's full covariance matrix. This reduces computational overhead (O(n) vs O(n²)) and may converge faster for low-dimensional problems (2-4 parameters).

## Results

### Tuning Runs

| Config | Samples | Score | Time (min) | Notes |
|--------|---------|-------|------------|-------|
| pop=10, gen=10, sigma=0.2, lr=0.1 | 20 | 1.0688 | 102.0 | Way too many simulations |
| pop=6, gen=5, sigma=0.2, lr=0.1 | 20 | 1.1293 | 57.1 | Better but still worse than baseline |
| pop=4, gen=3, sigma=0.25, lr=0.3 | 20 | 1.1390 | 37.7 | Comparable on 20 samples |
| pop=4, gen=3, sigma=0.25, lr=0.3 | 40 | 1.1204 | 43.3 | Final test - worse than baseline |

### Best In-Budget Configuration
- **Config:** population_size=4, max_generations=3, sigma=0.25, learning_rate=0.3
- **Score:** 1.1204
- **Projected Time:** 43.3 min
- **vs Baseline:** -0.0158 score, +4.3 min (1.1362 @ 39 min)

### Per-Source-Count Analysis (40-sample run)
- **1-source:** RMSE 0.1783 (n=16) - acceptable
- **2-source:** RMSE 0.2349 (n=24) - still problematic

## Why OpenAI ES Failed

### 1. Covariance Structure Matters for Low-Dim Problems
OpenAI ES uses diagonal covariance (independent perturbations per dimension), while CMA-ES maintains full covariance (captures correlations between parameters). In our 2-4D problem:
- Position parameters (x, y) are often correlated along the heat gradient
- CMA-ES captures this and searches efficiently along correlated directions
- OpenAI ES explores axis-aligned directions only, missing these correlations

### 2. Sample Efficiency Gap
CMA-ES is specifically designed for expensive black-box optimization with few function evaluations. OpenAI ES was designed for parallelizable RL with many cheap evaluations:
- CMA-ES: Optimal for 10-100 evals/init
- OpenAI ES: Designed for 1000+ evals with massive parallelization

### 3. Natural Gradient vs Covariance Adaptation
While both use natural gradients, they differ:
- CMA-ES: Adapts full covariance + step size via evolution path
- OpenAI ES: Uses antithetic sampling + fixed diagonal covariance

For our low-dim, expensive-evaluation problem, CMA-ES's approach is superior.

## Implementation Notes
- Implemented OpenAI ES from scratch (nevergrad not available)
- Used antithetic sampling (mirrored perturbations) for variance reduction
- Learning rate and sigma tuned across multiple runs
- Same infrastructure as baseline (40% timesteps, NM polish)

## Conclusion
OpenAI ES is NOT suitable for this inverse heat problem. The diagonal covariance approximation loses critical correlation information that CMA-ES captures. For low-dimensional expensive black-box optimization, CMA-ES remains optimal.

## Recommendation
- Mark `alternative_es` family as **FAILED** for this problem type
- Do not pursue Natural ES or PEPG (same diagonal assumption)
- CMA-ES's O(n²) covariance overhead is negligible for n=2-4 and provides crucial accuracy

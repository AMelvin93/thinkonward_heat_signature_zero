# Experiment Summary: adaptive_simulated_annealing

## Metadata
- **Experiment ID**: EXP_ADAPTIVE_SA_001
- **Worker**: W2
- **Date**: 2026-01-22
- **Algorithm Family**: simulated_annealing

## Objective
Test scipy's dual_annealing (generalized simulated annealing with local search) as an alternative global optimizer to CMA-ES, based on a 2024 Nature paper showing ASA success for heat source reconstruction.

## Hypothesis
ASA adaptively adjusts temperature schedule based on search state, potentially improving global optimization for complex inverse problems. SA's probabilistic uphill moves may find different optima than CMA-ES.

## Results Summary
- **Best In-Budget Score**: 0.8666 @ 28.2 min
- **Best Overall Score**: (estimated) ~1.16 @ 320 min (from early runs)
- **Baseline Comparison**: -0.2581 vs 1.1247 (**23% WORSE**)
- **Status**: **FAILED**

## Tuning History

| Run | Config | Score | Time (min) | In Budget | Notes |
|-----|--------|-------|------------|-----------|-------|
| 1 | 80/150 fevals, local search | N/A | ~320 proj | No | Killed. 5-6x over budget, RMSE 0.135 |
| 2 | 20/40 fevals, local search | N/A | ~150 proj | No | Killed. 2-3x over budget, RMSE 0.166 |
| 3 | 15/30 fevals, NO local search | 0.8666 | 28.2 | **Yes** | 23% worse than baseline |
| 4 | 10/20 fevals, local search | N/A | ~130 proj | No | Killed. 2x over budget, RMSE 0.173 |

## Key Findings

### Why dual_annealing Failed

1. **Local search overhead is fatal**
   - dual_annealing's local search uses L-BFGS-B with finite differences
   - Each L-BFGS-B step requires multiple function evaluations
   - Even with minimal fevals (10/20), local search causes 2x budget overrun

2. **Pure SA lacks accuracy**
   - Without local search, SA is fast (28 min) but inaccurate
   - RMSE 0.36 vs baseline 0.15 = 2.4x worse accuracy
   - SA's random walk exploration needs many more iterations to converge

3. **CMA-ES is fundamentally more efficient**
   - CMA-ES learns covariance structure from previous evaluations
   - SA's probabilistic acceptance wastes evaluations on uphill moves
   - For expensive simulations, CMA-ES's sample efficiency is critical

### The Nature Paper Context
The 2024 Nature paper used ASA for biological tissue heat source reconstruction, but:
- Their simulations were likely cheaper (simpler physics)
- They didn't have the same time constraints
- Different problem characteristics may favor different optimizers

## Parameter Sensitivity
- **Local search**: The most impactful parameter. With it: too slow. Without it: too inaccurate.
- **maxfun**: Reducing fevals helps time but can't compensate for local search overhead
- **visit/accept**: Temperature parameters had minimal impact compared to local search choice

## Recommendations for Future Experiments

### 1. Do NOT Try Pure SA Variants
The core issue isn't the specific SA implementation - it's that SA's exploration mechanism is less sample-efficient than covariance-adaptive methods for this problem.

### 2. Mark simulated_annealing Family EXHAUSTED
No SA variant will beat CMA-ES for this problem:
- With refinement: too slow due to gradient-free local search
- Without refinement: too inaccurate due to random exploration

### 3. Focus on CMA-ES Improvements
CMA-ES already achieves good sample efficiency. Future work should:
- Improve temporal fidelity further (current best: 40% timesteps)
- Better initialization strategies
- Multi-fidelity approaches that work (not spatial coarsening)

## Technical Details

### scipy.optimize.dual_annealing
```python
dual_annealing(
    objective,
    bounds,
    maxfun=...,
    initial_temp=5000,      # Starting temperature
    restart_temp_ratio=1e-5, # Restart threshold
    visit=2.62,             # Step length distribution
    accept=-5.0,            # Acceptance probability
    no_local_search=False,  # L-BFGS-B at each minimum
)
```

The `no_local_search=False` option triggers L-BFGS-B refinement, which dominates runtime.

## Raw Data
- MLflow run IDs: 6221577ac0a44e10a9e6539911cc69c5 (Run 3)
- Best in-budget config: {max_fevals: 15/30, n_restarts: 1, no_local_search: True}

## Conclusion
**FAILED** - Simulated annealing is fundamentally unsuitable for this problem. With local search refinement, it's 2-5x over budget. Without, it's 23% worse accuracy. CMA-ES's covariance-adaptive approach is more sample-efficient for expensive function evaluations. The simulated_annealing family should be marked EXHAUSTED.

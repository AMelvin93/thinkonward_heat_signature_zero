# Experiment Summary: ensemble_voting

## Metadata
- **Experiment ID**: EXP_ENSEMBLE_001
- **Worker**: W1
- **Date**: 2026-01-18
- **Algorithm Family**: ensemble

## Objective
Run multiple optimizers (CMA-ES and Nelder-Mead) in parallel and take the best result per sample to leverage the strengths of each algorithm.

## Hypothesis
Different optimizers excel on different samples; an ensemble approach would capture the best of each, improving accuracy without significantly increasing time.

## Results Summary
- **Best In-Budget Score**: N/A (never achieved budget compliance)
- **Best Overall Score**: 1.0972 @ 347.9 min
- **Baseline Comparison**: -0.0275 vs 1.1247 (WORSE)
- **Status**: FAILED

## Configuration Tested
```
CMA-ES: 15/24 fevals, sigma=0.15/0.2
Nelder-Mead: 30/50 maxiter, 3 inits
```

## Detailed Results
| Metric | Value |
|--------|-------|
| Score | 1.0972 |
| Projected 400 samples | 347.9 min |
| RMSE Mean | 0.1810 |
| RMSE 1-src | 0.1334 (32 samples) |
| RMSE 2-src | 0.2128 (48 samples) |
| Success Rate | 80/80 samples |

### Optimizer Win Distribution
| Optimizer | Wins | Percentage |
|-----------|------|------------|
| Nelder-Mead | 59 | 73.8% |
| CMA-ES | 21 | 26.2% |

## Key Findings

### What We Learned
1. **Nelder-Mead dominates but adds overhead**: Nelder-Mead won 74% of samples, confirming it finds good local optima. However, running both optimizers doubles simulation count.

2. **Massive time overhead**: 2-source samples required 900-1100 simulations each (vs ~500 for baseline CMA-ES alone), leading to 5.8x budget overrun.

3. **Score is WORSE than baseline**: Even with more compute, the ensemble achieved 1.0972 vs baseline's 1.1247. The extra candidates from multiple optimizers didn't improve the final selection.

4. **Outliers persist**: Three samples had RMSE > 0.4 (samples 57, 63, 10), suggesting the ensemble doesn't help with hard cases.

### Why This Failed
1. **Computational redundancy**: Running two optimizers means roughly 2x the simulations per sample
2. **No synergy**: Taking "best of two" doesn't outperform a well-tuned single optimizer
3. **Nelder-Mead inefficiency**: While it wins more often, its 3 random inits each doing 30-50 iterations is very expensive
4. **No early stopping**: Both optimizers run to completion rather than stopping when one finds a good solution

### Why Tuning Won't Help
- Even cutting all evals in HALF would project to ~174 min (2.9x over budget)
- Even cutting to QUARTER would project to ~87 min (45% over budget)
- Reducing evals would ALSO hurt the already-below-baseline score
- Fundamental approach is flawed for this time budget

## Recommendations for Future Experiments

### DO NOT TRY:
1. Ensemble approaches with multiple full optimizers - too expensive
2. Running Nelder-Mead from multiple random inits - use single init from CMA-ES best instead
3. Any approach that runs optimizers in parallel rather than sequentially

### CONSIDER TRYING:
1. **Sequential handoff**: Use CMA-ES for exploration (10 fevals), then hand off best to single Nelder-Mead run (not multiple inits)
2. **Cheap pre-filter**: Use coarse grid or surrogate to select ONE optimizer to run per sample
3. **Adaptive switching**: Start with CMA-ES, switch to Nelder-Mead only if convergence detected

## Abort Reason
**Abort criteria triggered**: Projected time 347.9 min > 70 min threshold. No amount of parameter tuning can bring this within budget while maintaining competitive accuracy.

## Raw Data
- MLflow run ID: (logged automatically by run.py)
- Configuration:
```json
{
  "cmaes_max_fevals_1src": 15,
  "cmaes_max_fevals_2src": 24,
  "cmaes_sigma0_1src": 0.15,
  "cmaes_sigma0_2src": 0.2,
  "nelder_mead_maxiter_1src": 30,
  "nelder_mead_maxiter_2src": 50,
  "nelder_mead_n_inits": 3
}
```

## Lessons for W0
1. **Ensemble = expensive**: Multi-optimizer ensembles fundamentally conflict with time budget
2. **Nelder-Mead is strong locally**: Should be used for REFINEMENT, not full optimization
3. **The baseline is well-tuned**: Simple CMA-ES with good parameters beats complex ensembles
4. **Focus on reducing simulations per sample**, not adding more optimization paths

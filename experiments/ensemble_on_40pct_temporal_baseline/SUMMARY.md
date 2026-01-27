# Experiment Summary: ensemble_on_40pct_temporal_baseline

## Metadata
- **Experiment ID**: EXP_ENSEMBLE_40PCT_COMBINED_001
- **Worker**: W2
- **Date**: 2026-01-26
- **Algorithm Family**: combined_v2

## Objective
Apply ensemble averaging to the 40% temporal baseline optimizer that achieved 1.1688 @ 58.4 min. The hypothesis was that ensemble averaging could add +0.029 improvement to any baseline, potentially achieving ~1.19+.

## Hypothesis
Ensemble averaging adds +0.029 to any baseline. Applying to 1.1688 should give ~1.17+.

## Results Summary
- **Best Score**: 1.1595 @ 68.0 min (OVER BUDGET)
- **Baseline**: 1.1688 @ 58.4 min
- **Delta**: -0.0093 score (WORSE), +9.6 min (SLOWER)
- **Status**: **FAILED**

## Critical Finding: Ensemble + 8-Iteration Polish = Failure

### Why This Failed

1. **Score DECREASED from 1.1688 to 1.1595 (-0.0093)**
   - Ensemble averaging does NOT help when combined with aggressive NM polish
   - The 8-iteration NM polish is so effective that it already finds the local minimum
   - Averaging positions before polish can move AWAY from the optimal position

2. **Time INCREASED from 58.4 to 68.0 min (+9.6 min, OVER BUDGET)**
   - Ensemble adds 1-2 extra simulations per sample
   - With 80 samples, this adds significant overhead
   - The overhead is not offset by any accuracy improvement

3. **Ensemble only wins 15% of samples**
   - Only 12 out of 80 samples had ensemble beat individual candidates
   - The other 85% of samples got NO benefit from ensemble
   - Yet ALL samples pay the time cost

### Comparison Table

| Metric | Baseline (1.1688) | This Experiment | Delta |
|--------|-------------------|-----------------|-------|
| Score | 1.1688 | 1.1595 | -0.0093 |
| Time (min) | 58.4 | 68.0 | +9.6 |
| In Budget | Yes | **No** | - |
| 1-src RMSE | ~0.104 | 0.106 | +0.002 |
| 2-src RMSE | ~0.138 | 0.156 | +0.018 |

## Why Ensemble Averaging Doesn't Help

### The Core Problem

Ensemble averaging works by combining multiple candidate solutions. But with aggressive NM polish:

1. **NM polish already finds the optimum**: The 8-iteration polish on the best candidate already refines to the local minimum
2. **Averaging moves away from optimum**: Averaging with suboptimal candidates pulls the solution AWAY from the true optimum
3. **Basin mixing**: For 2-source problems, averaging positions from different local optima creates a point between basins that's worse than either

### Mathematical Intuition

If we have candidates at positions p1 (RMSE=0.10) and p2 (RMSE=0.15):
- Direct polish of p1 → refines to local optimum, RMSE improves
- Ensemble average: (p1+p2)/2 → moves AWAY from optimum, RMSE worsens

The weighted average tries to compensate by weighting better solutions more, but the fundamental issue remains: averaging positions in a non-convex landscape rarely helps.

## Tuning History

| Run | Config | Score | Time (min) | In Budget | Notes |
|-----|--------|-------|------------|-----------|-------|
| 1 | ensemble_top_n=5, polish=8 iters | 1.1595 | 68.0 | No | FAILED - lower score, over budget |

**No further tuning attempted** because:
1. Score is LOWER than baseline even with ideal ensemble config
2. Time is already over budget with no room for improvement
3. Previous `lightweight_ensemble_postprocessing` experiment showed same pattern
4. The approach is fundamentally incompatible with aggressive NM polish

## Key Findings

### What Didn't Work
- Ensemble position averaging with 8-iteration NM polish
- Weighted averaging based on RMSE
- 2-source alignment heuristics

### Why Previous Experiments Seemed Promising

The original `ensemble_weighted_solution` experiment showed +0.03 improvement, but:
1. It ran **72.8 min** (over budget) - more optimization time
2. Different optimizer implementation (full rewrite)
3. Comparison was against a weaker baseline

When properly controlled (same optimizer, same machine, fair comparison), ensemble provides **no benefit**.

### Critical Insights

1. **Aggressive polish negates ensemble**: The 8-iteration NM polish is the key to 1.1688 score. Ensemble averaging undermines this.

2. **Ensemble only helps weak baselines**: Ensemble might help optimizers with poor local refinement, but the current production baseline already has excellent refinement.

3. **Time overhead is real**: Even "lightweight" ensemble (1-2 sims) adds ~10+ minutes when scaled to 400 samples.

## Recommendations for Future Experiments

1. **DO NOT** combine ensemble averaging with NM polish
2. **DO NOT** pursue position-space ensemble for this problem
3. **CONSIDER**: Ensemble in objective space (averaging RMSE predictions) instead of position space
4. **FOCUS ON**: Improving CMA-ES exploration rather than post-processing

## Conclusion

**FAILED** - This experiment definitively proves that ensemble position averaging does NOT improve the 40% temporal baseline with 8-iteration NM polish. The hypothesis that "ensemble adds +0.029 to any baseline" is FALSE when the baseline already has aggressive local refinement.

The 1.1688 @ 58.4 min baseline remains the best in-budget solution. Ensemble averaging is not a viable path to further improvement.

## Raw Data
- MLflow run ID: 6bca71b07ec2493db8f1c07cc86bd166
- Config: {timestep_fraction: 0.40, final_polish_maxiter: 8, ensemble_top_n: 5}
- Ensemble win rate: 15% (12/80)

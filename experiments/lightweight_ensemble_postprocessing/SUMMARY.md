# Experiment Summary: lightweight_ensemble_postprocessing

## Metadata
- **Experiment ID**: EXP_LIGHTWEIGHT_ENSEMBLE_001
- **Worker**: W2
- **Date**: 2026-01-26
- **Algorithm Family**: postprocessing_v2

## Objective
Add minimal ensemble averaging step to the production baseline optimizer, avoiding the overhead of a full optimizer rewrite that caused the previous `ensemble_weighted_solution` experiment to exceed budget.

## Hypothesis
Integrating ensemble as a lightweight postprocessing step (adding only 1-2 extra simulations per sample) should capture the accuracy improvement demonstrated in `ensemble_weighted_solution` while fitting within the 60-minute budget.

## Results Summary
- **Ensemble Score**: 1.1542 @ 56.9 min (Run 1), 1.1538 @ 49.9 min (Run 2)
- **Verified Baseline**: 1.1557 @ 49.3 min (same machine, same conditions)
- **Actual Comparison**: -0.0015 vs verified baseline (within noise)
- **Status**: **NEUTRAL** - Matches baseline, no improvement

## Critical Finding: Baseline Comparison Issue

The initial comparison used an outdated baseline (1.1247 from robust_fallback) instead of the actual current production baseline. When comparing against the verified baseline run on the same machine:

| Run | Score | Time (min) | Ensemble Wins |
|-----|-------|------------|---------------|
| Ensemble Run 1 | 1.1542 | 56.9 | 11.2% |
| Ensemble Run 2 | 1.1538 | 49.9 | 15.0% |
| **Production Baseline** | 1.1557 | 49.3 | N/A |

**The scores are essentially identical (1.154 vs 1.156), within random noise.**

## Key Findings

### What Didn't Work: Ensemble Position Averaging

1. **Ensemble only wins 11-15% of samples**: Despite weighted averaging with 2-source alignment, the ensemble solution beats individual candidates only ~13% of the time.

2. **Aggressive NM polish dominates**: The production baseline's 5-iteration NM polish on the best candidate is so effective that the ensemble average rarely beats the polished individual solution.

3. **No net benefit**: The ensemble step adds complexity (1-2 extra sims) without improving the score.

### Why Previous Experiment Showed Improvement

The `ensemble_weighted_solution` experiment showed +0.03 improvement because:
1. It used a DIFFERENT optimizer implementation (full rewrite)
2. Different timing/scheduling may have caused different baseline comparison
3. The high overhead (72 min) may have allowed more optimization

### Comparison: Full Rewrite vs Lightweight

| Metric | ensemble_weighted_solution | lightweight_ensemble |
|--------|---------------------------|---------------------|
| Score | 1.1552 | 1.1542 |
| Time (min) | 72.8 (over budget) | 56.9 (in budget) |
| Ensemble Wins | 98.8% | 11.2% |
| **vs Verified Baseline** | N/A | -0.0015 (within noise) |

The full rewrite's high ensemble win rate (98.8%) may be due to different optimizer parameters or the longer runtime allowing more exploration.

## Technical Analysis

### Why Ensemble Averaging Has Limited Benefit

1. **CMA-ES already averages solutions**: CMA-ES updates the mean based on all evaluated solutions, providing implicit ensemble benefits.

2. **NM polish is highly effective**: The 5-iteration Nelder-Mead polish already finds the local minimum near the best candidate.

3. **Position averaging mixes basins**: As W1 found, averaging positions from different local optima can produce a point between basins that's worse than either.

4. **2-source permutation alignment helps but not enough**: While alignment prevents catastrophic averaging of swapped sources, it doesn't help when sources are near different local minima.

## Recommendations

1. **DO NOT promote to production**: The ensemble step provides no measurable improvement over baseline.

2. **Mark postprocessing_v2 family as exhausted**: All ensemble averaging approaches (weighted, unweighted, median) have been tested with no success.

3. **Key insight**: Ensemble averaging works in theory (wins 11-15% of samples) but the baseline NM polish is already so effective that the net benefit is zero.

## Conclusion

**NEUTRAL** - The lightweight ensemble approach achieves the same score as the production baseline (1.154 vs 1.156, within noise). The ensemble wins only 11-15% of samples, which is not enough to improve the overall score.

This confirms that:
1. Position averaging has limited benefit when combined with aggressive local refinement (NM polish)
2. The production baseline's CMA-ES + NM polish is already near-optimal
3. Further improvements likely require fundamentally different approaches, not postprocessing variants

## Raw Data
- MLflow run IDs: 3d67dc22ad624a9aa70db00bbf179349, fa0e290c9e0b4c1181627f6338205f6e
- Best config: {ensemble_top_n: 5}
- Ensemble win rate: 11-15%
- Verified baseline: 1.1557 @ 49.3 min (early_timestep_filtering)

# Experiment Summary: CMA-ES to Nelder-Mead Sequential Handoff

## Metadata
- **Experiment ID**: EXP_SEQUENTIAL_HANDOFF_001
- **Worker**: W2
- **Date**: 2026-01-19
- **Algorithm Family**: hybrid

## Objective
Test sequential handoff from CMA-ES exploration to Nelder-Mead refinement, where CMA-ES finds a good region and NM polishes the solution, without the parallel overhead of the ensemble approach.

## Hypothesis
Sequential handoff captures CMA-ES exploration + NM local refinement without doubling simulation count. Unlike the failed ensemble experiment (parallel, 2x simulations), this is sequential with minimal overhead.

## Results Summary
- **Best In-Budget Score**: 1.1132 @ 56.6 min (Run 4)
- **Best Overall Score**: 1.1439 @ 126.5 min (Run 2)
- **Baseline Comparison**: -0.0115 vs baseline 1.1247 (IN-BUDGET RESULT IS WORSE)
- **Status**: **FAILED**

## Tuning History

| Run | Config Changes | Score | Time (min) | In Budget | Notes |
|-----|---------------|-------|------------|-----------|-------|
| 1 | NM maxiter=40/50, top_n=3, fine grid | - | - | No | ABORTED - 500+ sec per 2-src sample |
| 2 | NM maxiter=8/12, top_n=2, fine grid | 1.1439 | 126.5 | No | Best score but 2x over budget |
| 3 | NM maxiter=12/18, top_n=2, coarse grid | 1.1331 | 108.3 | No | Coarse helped but still 48 min over |
| 4 | NM maxiter=3/5, top_n=1, coarse grid | 1.1132 | 56.6 | **Yes** | IN BUDGET but BELOW baseline |

## Key Findings

### What Worked (Academically)
- NM does improve solution quality when given enough iterations
- Run 2 achieved +0.0192 over baseline with fine-grid NM
- Coarse-grid NM saves time vs fine-grid

### What Didn't Work (Practically)
- **Time overhead is prohibitive**: Even minimal NM adds too much time
- **Accuracy-time trade-off is unfavorable**: Reducing NM to fit budget loses the accuracy benefit
- **CMA-ES already uses most budget**: Baseline CMA-ES takes ~57 min, leaving no room for NM

### Critical Insights
1. **Sequential handoff is fundamentally flawed for this problem**: The time budget is fully consumed by CMA-ES optimization. Any additional refinement step pushes us over budget.

2. **NM's value is in fine-grid iterations**: When we use coarse-grid NM (for speed), we lose the precision benefit. When we use fine-grid NM (for quality), we blow the budget.

3. **The ensemble experiment insight was wrong**: Ensemble found "NM won 74%" but that's because NM was given MORE simulations. The lesson isn't "add NM" but rather "NM wins when given more compute budget."

4. **Diminishing returns curve**: Score improvement per NM iteration is steep initially but drops off. By the time we reduce NM enough to fit budget, we're in the flat part of the curve.

## Parameter Sensitivity
- **Most impactful**: nm_grid (fine vs coarse) - massive time difference
- **Time-sensitive**: nm_maxiter, nm_top_n - each additional iteration costs ~2-3 sec per sample
- **Threshold-insensitive**: RMSE thresholds had minimal impact on outcome

## Time Budget Analysis

| Component | Time Cost | Notes |
|-----------|-----------|-------|
| Baseline CMA-ES | ~57 min | Already at budget limit |
| NM fine-grid (8/12 iters) | +70 min | 2x budget |
| NM coarse-grid (12/18 iters) | +51 min | Still 1.8x budget |
| NM minimal coarse (3/5 iters) | +0 min | Fits budget but no benefit |

## Recommendations for Future Experiments

### DO NOT TRY
1. Any variant of CMA-ES + NM sequential - the math doesn't work
2. Any post-processing refinement on the fine grid
3. Adding optimizers after CMA-ES

### CONSIDER INSTEAD
1. **Multi-fidelity WITHIN CMA-ES**: Run more CMA-ES generations on coarse grid, fewer on fine
2. **Better initialization**: Reduce CMA-ES iterations needed by starting closer to solution
3. **Surrogate filtering**: Use cheap proxy to filter before expensive simulation
4. **Source count detection**: Skip 2-src optimization on 1-src samples

### Why This Failed But Others Might Work
- This experiment tried to ADD work (NM) after CMA-ES
- Successful approaches need to REPLACE expensive work with cheaper alternatives
- Multi-fidelity replaces fine simulations with coarse ones (cheaper)
- Surrogate filtering replaces simulations with ML predictions (cheaper)

## Conclusion

**FAILED**: Sequential CMA-ES to NM handoff does not work within the 60-minute budget. The best in-budget configuration (Run 4) achieves 1.1132, which is 0.0115 WORSE than baseline 1.1247.

The core problem is that CMA-ES already consumes the full time budget. Any post-processing, no matter how minimal, either:
1. Pushes us over budget (Runs 1-3)
2. Adds negligible value that doesn't compensate for reduced CMA-ES budget (Run 4)

This experiment conclusively rules out the "CMA-ES exploration + local refinement" hybrid approach for this problem.

## Raw Data
- Run 1: ABORTED
- Run 2: Score 1.1439, RMSE 0.1516, Time 126.5 min
- Run 3: Score 1.1331, RMSE 0.1647, Time 108.3 min
- Run 4: Score 1.1132, RMSE 0.1934, Time 56.6 min

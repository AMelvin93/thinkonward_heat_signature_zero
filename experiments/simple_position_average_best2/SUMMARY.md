# Experiment Summary: simple_position_average_best2

## Metadata
- **Experiment ID**: EXP_SIMPLE_POSITION_AVERAGE_001
- **Worker**: W1
- **Date**: 2026-01-26
- **Algorithm Family**: postprocessing_v4

## Status: FAILED

## Objective
Test whether averaging the positions of the top-2 CMA-ES candidates and recomputing optimal intensity produces a better solution than either individual candidate.

## Hypothesis
When CMA-ES finds multiple good candidates, they may be on opposite sides of the true optimum. Averaging their positions could yield a better estimate that's closer to the global minimum.

## Approach
1. Run standard CMA-ES optimization with early timestep filtering (baseline)
2. After getting candidate pool, sort by RMSE and take top-2
3. Average their positions: `(x1+x2)/2, (y1+y2)/2`
4. For 2-source problems, align sources before averaging using minimum distance matching
5. Compute optimal intensity for the averaged position
6. Add averaged candidate to pool before dissimilarity filtering

## Results Summary
- **Best In-Budget Score**: 1.1297 @ 33.6 min
- **Baseline**: 1.1688 @ 58.4 min
- **Delta**: -0.0391 (-3.3%)
- **Status**: FAILED

## Tuning History

| Run | Config | Score | Time (min) | In Budget | Notes |
|-----|--------|-------|------------|-----------|-------|
| 1 | enable_avg=True, n_avg=2 | 1.1297 | 33.6 | Yes | Averaged selected 18.8% of time |

## Detailed Results

### RMSE Analysis
| Metric | 1-source | 2-source | Combined |
|--------|----------|----------|----------|
| RMSE mean | 0.1405 | 0.1981 | 0.1751 |
| Time mean | 19.3s | 43.1s | - |

### Averaged Candidate Selection
- Samples where averaged candidate had best RMSE: **15/80 (18.8%)**
- Despite being selected in ~19% of cases, overall score decreased

## Key Findings

### What Didn't Work

1. **Position Averaging Doesn't Preserve Optimality**
   - Averaging two local minima does NOT produce a point closer to the global minimum
   - The heat equation loss landscape is not convex - averaging positions typically moves to a worse region
   - Example: If two candidates are at RMSE 0.10 and 0.12, their average might be at RMSE 0.20

2. **Averaging Can Hurt When Candidates Are Far Apart**
   - For 2-source problems, averaging sources that are far apart creates artificial intermediate positions
   - The averaged position has no physical meaning in the context of the inverse problem

3. **Selection Bias Creates False Positives**
   - Averaged candidate was selected 18.8% of the time
   - However, in many of these cases, the averaging might have replaced a better candidate in the pool due to dissimilarity filtering
   - The "selection" of the averaged candidate doesn't mean it improved the solution

### Why The Hypothesis Failed

The hypothesis assumed the loss landscape has a convex-like structure near optima where averaging would help. In reality:

1. **Multi-modal Landscape**: The heat source identification problem has multiple local minima
2. **Non-convex Basins**: Even within a single basin, the optimal point is rarely at the geometric center of good solutions
3. **Position-Intensity Coupling**: Changing position also changes optimal intensity - averaging breaks this coupling

## Critical Insight

**Position averaging is fundamentally incompatible with this inverse problem.**

Unlike ensemble methods in classification (where averaging predictions often helps due to reduced variance), position averaging in optimization:
- Destroys the relationship between position and optimal intensity
- Moves solutions away from local minima into potentially worse regions
- Only helps if candidates happen to be equidistant from the true solution (rarely true)

## Recommendations for Future Experiments

1. **DO NOT pursue position averaging approaches** - This includes top-3, top-5, or weighted averaging variants. The fundamental problem is the non-convex landscape.

2. **If ensemble approaches are desired, try:**
   - Voting/selection from multiple candidates (not averaging)
   - Running CMA-ES multiple times from different initializations
   - Using the diversity bonus in scoring instead of averaging

3. **Better alternatives for ensemble-like behavior:**
   - Keep more diverse candidates in the pool (increase candidate_pool_size)
   - Use dissimilarity filtering more aggressively to maintain diversity
   - Trust the best individual solution rather than averaging

## Raw Data
- MLflow run ID: 559a7d4fc57e4471a825e19ee1d5bd87
- Best config: `{"enable_averaging": true, "n_to_average": 2}`
- Files: `optimizer.py`, `run.py`, `STATE.json`

## Conclusion

**Simple position averaging of top-2 candidates is not beneficial for heat source identification.**

The approach reduces accuracy by 3.3% while adding complexity. The heat equation inverse problem has a non-convex loss landscape where averaging positions moves solutions away from optima, not toward them.

This definitively rules out all position-averaging ensemble approaches (top-2, top-3, top-5, weighted, etc.) for this problem class.

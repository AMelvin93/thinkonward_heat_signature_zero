# Experiment Summary: stratified_sample_processing

## Metadata
- **Experiment ID**: EXP_SAMPLE_STRATIFIED_001
- **Worker**: W2
- **Date**: 2026-01-26
- **Algorithm Family**: sampling_v2

## Objective
Use different optimization settings for 1-source vs 2-source problems to better match problem complexity:
- 1-source: fewer fevals (15), more polish (10 iter) - simpler landscape, refinement helps
- 2-source: standard fevals (36), less polish (6 iter) - complex landscape, exploration matters

## Hypothesis
1-source and 2-source problems may benefit from different optimization settings.

## Results Summary
- **Score**: 1.1605 @ 69.1 min (OVER BUDGET)
- **Baseline**: 1.1688 @ 58.4 min
- **Delta**: -0.0083 score (WORSE), +10.7 min (SLOWER, OVER BUDGET)
- **Status**: **FAILED**

## RMSE Breakdown

| Source Type | Stratified | Baseline | Delta |
|-------------|------------|----------|-------|
| 1-source | 0.1037 | ~0.104 | Similar |
| 2-source | 0.1606 | ~0.138 | **+0.022 (WORSE)** |

## What Went Wrong

### 1. 2-Source Polish Reduction Hurt Accuracy
- Reducing polish from 8 to 6 iterations for 2-source problems significantly hurt accuracy
- 2-source RMSE went from ~0.138 to 0.1606 (+0.022 increase)
- The complex 4D landscape (x1, y1, x2, y2) needs MORE refinement, not less

### 2. 1-Source More Polish Didn't Help
- Increasing polish from 8 to 10 iterations for 1-source showed minimal improvement
- 1-source RMSE went from ~0.104 to 0.1037 (negligible difference)
- The 8-iteration baseline was already near-optimal for the simpler 2D landscape

### 3. Time Budget Violated
- Projected time: 69.1 min (vs 60 min budget)
- The 10-iteration polish for 1-source added more time than the 6-iteration polish saved

## Why the Hypothesis Was Wrong

The hypothesis assumed:
1. 1-source problems are "too easy" and don't need as many CMA-ES iterations
2. 2-source problems are "too hard" and polish iterations are wasted

Reality:
1. 1-source problems benefit from exploration (fevals) just as much
2. 2-source problems NEED the polish iterations to find the local minimum in a complex landscape
3. The baseline's uniform approach is actually optimal

## Key Insight

**NM Polish is More Valuable for Complex Problems**

The 2-source problem has a 4D search space with potential multimodality. NM polish helps refine the position after CMA-ES finds a good region. Reducing polish iterations causes the optimizer to stop before reaching the local minimum.

## Configuration Comparison

| Parameter | Stratified | Baseline | Winner |
|-----------|------------|----------|--------|
| 1-src fevals | 15 | 20 | Baseline |
| 2-src fevals | 36 | 36 | Tie |
| 1-src polish | 10 | 8 | Tie (minimal diff) |
| 2-src polish | 6 | **8** | **Baseline** |
| Overall score | 1.1605 | **1.1688** | **Baseline** |

## Recommendations

1. **Keep uniform polish iterations**: The baseline's 8 iterations for both 1-src and 2-src is optimal

2. **DO NOT reduce 2-source polish**: The complex landscape needs refinement

3. **Mark sampling_v2 family as explored**: Stratified processing doesn't help

4. **Focus elsewhere**: Improvements should target:
   - Better initialization (already optimized with triangulation)
   - Earlier timestep filtering (already at 40%)
   - Alternative loss functions (tested and failed)

## Conclusion

**FAILED** - The stratified approach produced worse results than the uniform baseline. The key insight is that 2-source problems need MORE refinement (NM polish), not less. The baseline's 8-iteration polish for both source types is already optimal.

## Raw Data
- MLflow run ID: 318fec91ab164abb868ce339c41de64f
- Config: {fevals_1src: 15, fevals_2src: 36, polish_1src: 10, polish_2src: 6}
- Samples: 80 (32 1-source, 48 2-source)

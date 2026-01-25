# ICA-Seeded Initialization Experiment

## Experiment ID: EXP_ICA_SEEDED_INIT_001
**Status**: FAILED
**Worker**: W1
**Date**: 2026-01-24

## Hypothesis

FastICA achieved the best accuracy (RMSE 1.0422) in earlier experiments but took 87 minutes. Using ICA only for initialization (~1-2 seconds per sample) should provide better starting positions for CMA-ES while staying within the 60-minute budget.

## Implementation

Added FastICA initialization for 2-source problems:
1. Decompose sensor readings using sklearn's FastICA (n_components=2)
2. Extract mixing matrix columns (weight vectors)
3. Estimate source positions via weighted centroid of sensor positions
4. Use ICA-derived positions as additional starting points for CMA-ES

## Results

| Run | ICA | ICA max_iter | Score | Time (min) | In Budget | Notes |
|-----|-----|--------------|-------|------------|-----------|-------|
| 1   | Yes | 50           | 1.1601 | 69.2      | NO        | -0.0087 vs baseline |

### Detailed Metrics (Run 1)
- **1-src RMSE**: 0.1130 (n=32)
- **2-src RMSE**: 0.1582 (n=48)
- **ICA used**: 22 samples (out of 48 2-source samples)

### Comparison to Baseline
| Metric | Baseline | ICA-Seeded | Delta |
|--------|----------|------------|-------|
| Score  | 1.1688   | 1.1601     | -0.0087 |
| Time   | 58.4 min | 69.2 min   | +10.8 min |

## Analysis

### Why ICA Initialization Failed

1. **Significant Overhead**: ICA adds ~10+ minutes for 80 samples (~7.5s per 2-source sample), pushing total time well over budget.

2. **No Accuracy Improvement**: The ICA-derived source positions are NOT better starting points than the random/grid initializations already used by CMA-ES. The score actually decreased by 0.0087.

3. **CMA-ES Already Sufficient**: The CMA-ES optimizer is robust enough to find good solutions from random initialization. Adding ICA as a starting point doesn't help because CMA-ES explores the search space effectively regardless of initial position.

4. **ICA Limitations**: FastICA assumes linear mixing of independent sources, which doesn't match the physics of heat diffusion. The heat equation creates non-linear, spatially-varying sensor responses that ICA cannot properly decompose.

## Key Finding

**ICA initialization is not beneficial for this problem.** The overhead of running ICA (~7.5s per sample) provides no accuracy benefit because:
- CMA-ES is already effective at finding global optima
- The ICA-derived positions are not more accurate than random initialization
- The physics of heat diffusion violates ICA's linear mixing assumption

## Recommendation

Do NOT use ICA initialization. The baseline approach (40% temporal fidelity + sigma 0.15/0.20 + 8 NM polish) remains optimal:
- **Score**: 1.1688
- **Time**: 58.4 min

## Files
- `optimizer.py`: Modified optimizer with ICA initialization
- `run.py`: Experiment runner with MLflow logging
- `STATE.json`: Tuning history and configuration

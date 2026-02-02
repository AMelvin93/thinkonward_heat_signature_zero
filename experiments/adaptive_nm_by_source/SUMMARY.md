# Adaptive NM by Source Count Experiment

## Status: FAILED (uniform NM is better)

## Hypothesis
Use different NM polish iterations based on source count:
- 1-source problems: fewer iterations (converge faster)
- 2-source problems: more iterations (higher complexity)

## Results

| Run | nm_1src | nm_2src | Score | Time (min) | vs Baseline |
|-----|---------|---------|-------|------------|-------------|
| 1 | 4 | 8 | 1.1422 | 43.7 | -0.0042 |
| 2 | 4 | 10 | 1.1322 | 45.4 | -0.0142 |
| 3 | 6 | 8 | 1.1291 | 40.8 | -0.0173 |

**Baseline** (uniform 8 NM): 1.1464 @ 51.2 min

## RMSE Analysis

| Config | RMSE 1src | RMSE 2src | Time 1src | Time 2src |
|--------|-----------|-----------|-----------|-----------|
| nm4/nm8 | 0.1457 | 0.1915 | 20.2s | 58.5s |
| nm4/nm10 | 0.1298 | 0.2042 | 22.1s | 60.6s |
| nm6/nm8 | 0.1286 | 0.2128 | 23.7s | 52.6s |

## Key Findings

1. **Uniform 8 NM is optimal**: All adaptive configs underperformed baseline
2. **More NM for 2-src doesn't help**: nm_2src=10 had WORSE 2-src RMSE (0.2042 vs 0.1915)
3. **Less NM for 1-src hurts score**: Lower 1-src RMSE doesn't compensate for worse 2-src
4. **Both sources need similar polish**: The complexity difference doesn't justify different NM

## Why Adaptive Doesn't Work

The hypothesis assumed:
- 1-source converges faster → needs less polish
- 2-source is more complex → needs more polish

Reality:
- **Both benefit from 8 iterations equally**
- Reducing 1-src iterations hurts accuracy without saving much time
- Increasing 2-src iterations doesn't improve accuracy (may overfit)

## Tuning Efficiency

- **Runs executed**: 3
- **Time utilization**: 73% (43.7/60 min best case)
- **Parameter space explored**: nm_1src [4, 6], nm_2src [8, 10]

## Recommendation

**KEEP uniform refine_maxiter=8 for all samples**. Adaptive allocation does not improve results.

---
**Worker**: W1
**Completed**: 2026-02-02
**Runs**: 3 (systematic exploration)

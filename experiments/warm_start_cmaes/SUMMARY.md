# Experiment Summary: warm_start_cmaes (WS-CMA-ES)

## Metadata
- **Experiment ID**: EXP_WS_CMAES_001
- **Worker**: W1
- **Date**: 2026-01-19
- **Algorithm Family**: meta_learning
- **Folder**: `experiments/warm_start_cmaes/`

## Objective
Test Warm Start CMA-ES (WS-CMA-ES) from CyberAgentAILab's cmaes library to improve optimization efficiency through learned distribution transfer.

## Hypothesis
Warm starting CMA-ES from a probing phase would:
1. Use multiple short probing runs to explore the search space
2. Combine information via get_warm_start_mgd() to estimate an initial distribution
3. Run main optimization from this warm start for faster convergence

## Results Summary
- **Best In-Budget Result**: RMSE 0.4599 @ 23.3 min (Run 2) - **170% WORSE than baseline**
- **Best Overall Result**: RMSE 0.2759 @ 65.8 min (Run 1) - **62% WORSE than baseline**
- **Baseline**: RMSE ~0.17 @ 57 min
- **Status**: **FAILED**

## Tuning History

| Run | Config Changes | RMSE | Time (min) | In Budget | Notes |
|-----|---------------|------|------------|-----------|-------|
| 1 | 3 probing starts, 5 fevals, main 15/25 | 0.2759 | 65.8 | No | 62% worse, 10% over budget |
| 2 | 2 probing starts, 3 fevals, main 20/32 | 0.4599 | 23.3 | Yes | 170% worse, fast but useless |

### Detailed Results by Source Type
| Run | 1-src RMSE | Baseline 1-src | Degradation | 2-src RMSE | Baseline 2-src | Degradation |
|-----|------------|----------------|-------------|------------|----------------|-------------|
| 1 | 0.2917 | 0.14 | 108% worse | 0.2654 | 0.21 | 26% worse |
| 2 | 0.3366 | 0.14 | 140% worse | 0.5420 | 0.21 | 158% worse |

## Key Findings

### What Didn't Work

1. **Probing phase wastes budget**: Running short CMA-ES probing runs from random/smart initializations doesn't provide useful information for warm starting. The budget would be better spent on a single well-optimized CMA-ES run.

2. **get_warm_start_mgd() doesn't help**: The warm start distribution computed from probing solutions doesn't improve convergence. In fact, it may be directing the main optimization toward suboptimal regions.

3. **CyberAgentAILab's cmaes library is not well-tuned for this problem**: The library is designed for hyperparameter optimization, not simulation-based inverse problems. The baseline using pycma is more mature and well-tuned.

4. **Each sample is independent**: Unlike hyperparameter optimization where tasks are similar, each thermal inverse problem sample has unique heat source positions. There's no meaningful covariance structure to transfer.

### Why WS-CMA-ES Fails for This Problem

The core assumption of WS-CMA-ES is that:
> "The optimization landscape is similar across tasks, so learned covariance transfers meaningfully"

For heat source identification:
- **Each sample is a different optimization problem** with unique source positions
- **No shared structure**: The covariance learned on one sample doesn't help another
- **Probing adds overhead**: The probing phase consumes budget without benefit
- **pycma is already well-initialized**: The baseline uses triangulation + smart init, which is more effective than statistical warm starting

### Critical Insight
**WS-CMA-ES is designed for transfer learning across SIMILAR tasks.** Heat source samples are NOT similar tasks - each has unique source positions. The approach is fundamentally mismatched to this problem.

### Comparison with Failed cluster_transfer
| Approach | What it transfers | Result |
|----------|-------------------|--------|
| cluster_transfer | Solution positions | FAILED - positions don't transfer |
| WS-CMA-ES | Distribution (mean, sigma, cov) | FAILED - distributions don't transfer |

Both approaches fail for the same reason: **each sample is an independent optimization problem with no shared structure**.

## Parameter Sensitivity
- **Most impactful parameter**: Total fevals (more = better, but still worse than baseline)
- **Probing overhead**: More probing = more wasted budget
- **Warm start hyperparameters (gamma, alpha)**: No tuning helped

## Recommendations for Future Experiments

### What to AVOID
1. **Any form of transfer learning between samples** - fundamentally doesn't work for this problem
2. **Probing phases** - waste budget without useful information
3. **Statistical warm starting** - each sample is unique

### What MIGHT Work Instead
1. **Better single-sample initialization**: Improve triangulation/smart init within each sample
2. **Adaptive budget allocation**: Spend more time on harder samples
3. **Problem-specific covariance**: Use physics-informed covariance structure (e.g., x and y are independent)

### For W0's Reference
- meta_learning family is now EXHAUSTED (cluster_transfer + WS-CMA-ES both failed)
- Transfer between samples is NOT viable for inverse heat problems
- Focus on within-sample optimizations instead

## Raw Data
- **MLflow run IDs**:
  - Run 1: `85600c5927634c699bf5cf84ffa91ccc`
  - Run 2: `8ac369226bd14898b96239b089aa0194`
- **Best in-budget config**: Run 2 (RMSE 0.4599 - useless)
- **Baseline to beat**: RMSE 0.17 @ 57 min

## Conclusion
**FAILED**: Warm Start CMA-ES does not work for thermal inverse problems. Each sample has unique source positions with no shared optimization landscape structure. Probing phases waste budget without providing useful warm start information. The meta_learning approach family is now exhausted.

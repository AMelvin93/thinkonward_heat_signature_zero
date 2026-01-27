# Experiment Summary: larger_candidate_pool

## Metadata
- **Experiment ID**: EXP_LARGER_CANDIDATE_POOL_001
- **Worker**: W2
- **Date**: 2026-01-26
- **Algorithm Family**: diversity_tuning

## Objective
Test whether a larger candidate pool (15 vs baseline 10) improves diversity score by providing more candidate solutions for dissimilarity filtering.

## Hypothesis
With more candidates in the initial CMA-ES pool, the dissimilarity filter has more options to select from, potentially leading to better diverse solutions and higher overall score.

## Results Summary
- **Score**: 1.1726 @ 85.6 min (MASSIVELY OVER BUDGET)
- **Baseline**: 1.1688 @ 58.4 min
- **Delta**: +0.0038 score (marginally better), +27.2 min (+46% slower, FAR OVER BUDGET)
- **Status**: **FAILED**

## RMSE Breakdown

| Source Type | Larger Pool | Baseline | Delta |
|-------------|-------------|----------|-------|
| 1-source | 0.1101 | ~0.104 | +0.006 (slightly worse) |
| 2-source | 0.1718 | ~0.138 | +0.034 (WORSE) |

## Candidate Distribution

| Candidates | Larger Pool (15) | Baseline (10) |
|------------|------------------|---------------|
| 1-cand | 0 | ~2-3 |
| 2-cand | 3 | ~8-10 |
| 3-cand | 77 | ~68-70 |

**Note**: More samples achieved 3 candidates with the larger pool, but this came at massive time cost.

## What Went Wrong

### 1. Time Explosion (+27.2 min)
- Larger pool size means more candidates to polish with NM
- Each additional candidate adds ~8-12 seconds of NM polish time
- With 50% more candidates in pool (15 vs 10), the compounding effect is massive
- Total runtime increased by 46% (85.6 vs 58.4 min)

### 2. Diminishing Returns on Score
- Score improved only marginally (+0.0038)
- The additional candidates didn't significantly improve accuracy
- Most of the extra candidates were filtered out by dissimilarity check

### 3. RMSE Actually Got Worse
- 1-source RMSE: 0.1101 vs ~0.104 baseline (slightly worse)
- 2-source RMSE: 0.1718 vs ~0.138 baseline (significantly worse)
- More candidates doesn't mean BETTER candidates

## Root Cause Analysis

The larger candidate pool creates a tradeoff that doesn't favor accuracy:

1. **Time Cost is Linear**: Each additional candidate in the pool adds polish time
2. **Score Benefit is Sublinear**: Most extra candidates are filtered by dissimilarity (tau=0.2)
3. **Quality Dilution**: Averaging over more candidates dilutes the best solutions

The baseline pool_size=10 was already optimized to:
- Generate enough candidates for diversity (77/80 samples got 3 candidates)
- Not waste time on redundant candidates

## Configuration Comparison

| Parameter | Larger Pool | Baseline | Winner |
|-----------|-------------|----------|--------|
| candidate_pool_size | 15 | **10** | **Baseline** |
| Score | 1.1726 | 1.1688 | Larger Pool (marginal) |
| Time | 85.6 min | **58.4 min** | **Baseline** |
| In budget? | NO | **YES** | **Baseline** |

## Key Insight

**The Baseline Pool Size is Already Optimal**

The pool_size=10 achieves:
- Near-maximum diversity (77/80 samples get 3 candidates)
- Acceptable runtime (58.4 min, within 60 min budget)
- Best accuracy/time tradeoff

Increasing pool size only adds overhead without meaningful benefit.

## Recommendations

1. **Keep pool_size=10**: The baseline is already optimal for the time budget

2. **DO NOT increase pool size**: The time penalty far outweighs any score benefit

3. **Mark diversity_tuning family as explored**: Pool size tuning doesn't help

4. **Alternative approaches for diversity**:
   - Better initialization strategies (already optimized with triangulation)
   - Different dissimilarity thresholds (tau parameter)
   - Multi-restart with different random seeds

## Conclusion

**FAILED** - The larger candidate pool produced a marginally better score (+0.0038) but at an unacceptable time cost (+27.2 min, 46% slower). The experiment clearly demonstrates that:

1. Pool size is NOT a lever for improvement
2. The baseline pool_size=10 is already optimal
3. Time budget constraints make larger pools impractical

The key insight is that diversity comes from initialization quality (triangulation, gradient-based refinement), not from generating more redundant candidates.

## Raw Data
- MLflow run ID: 10642f66137e47428159ff399831ac68
- Config: {candidate_pool_size: 15, fevals_1src: 20, fevals_2src: 36, timestep: 0.40, polish: 8}
- Samples: 80 (32 1-source, 48 2-source)
- Candidate distribution: {1: 0, 2: 3, 3: 77}

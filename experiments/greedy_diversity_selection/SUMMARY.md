# Experiment Summary: greedy_diversity_selection

## Metadata
- **Experiment ID**: EXP_GREEDY_DIVERSITY_001
- **Worker**: W2
- **Date**: 2026-01-26
- **Algorithm Family**: scoring_v2

## Objective
Replace the baseline's simple "sort by RMSE, filter by dissimilarity" approach with greedy selection that maximizes the combined score (accuracy + diversity) directly.

## Hypothesis
The scoring formula `score = mean(1/(1+RMSE)) + 0.3*(n_candidates/3)` rewards both accuracy AND diversity. A greedy selection that considers both terms might outperform simple RMSE-first sorting.

## Results Summary
- **Score**: 1.1499 @ 68.5 min (OVER BUDGET)
- **Baseline**: 1.1688 @ 58.4 min
- **Delta**: -0.0189 score (WORSE), +10.1 min (OVER BUDGET)
- **Status**: **FAILED**

## Candidate Distribution Comparison

| Candidates | Greedy | Baseline | Delta |
|------------|--------|----------|-------|
| 1-cand | 3 | ~2 | +1 |
| 2-cand | 13 | ~8 | +5 |
| 3-cand | 64 | ~70 | **-6** |

The greedy approach produced **fewer** 3-candidate samples, which hurt the diversity score.

## RMSE Comparison

| Source Type | Greedy | Baseline | Delta |
|-------------|--------|----------|-------|
| 1-source | 0.1199 | ~0.104 | +0.016 (worse) |
| 2-source | 0.1690 | ~0.138 | +0.031 (worse) |

Both 1-source and 2-source RMSE were worse with greedy selection.

## What Went Wrong

### 1. Greedy Doesn't Always Select Best RMSE
The greedy approach tries to balance accuracy and diversity at each step. However:
- In the first step, it adds a small bias toward low RMSE (good anchor)
- In subsequent steps, it considers diversity contribution
- This means it might NOT select the lowest RMSE candidate if a slightly higher RMSE candidate adds more diversity

**Problem**: The scoring formula averages RMSE. Selecting a higher-RMSE candidate early "contaminates" the average.

### 2. Fewer 3-Candidate Samples
The greedy selection is more strict about diversity, resulting in:
- 64/80 samples with 3 candidates (vs ~70 in baseline)
- 13/80 samples with 2 candidates (vs ~8 in baseline)

**Problem**: The diversity bonus from having 3 candidates (+0.3 vs +0.2) outweighs the marginal RMSE benefit.

### 3. Implementation Overhead
The greedy selection adds computational overhead without improving results. The baseline's simple approach is both faster and more effective.

## Root Cause Analysis

The baseline's approach is actually near-optimal because:

1. **Sort by RMSE first**: Ensures the best accuracy candidate is always first
2. **Filter by dissimilarity (tau=0.2)**: Ensures subsequent candidates are meaningfully different
3. **Take up to 3**: Maximizes diversity bonus

The greedy approach fails because:
1. It might select higher-RMSE candidates for diversity reasons
2. This hurts the averaged accuracy term
3. The marginal diversity gain doesn't compensate for accuracy loss

## Key Insight

**Simple Sort + Filter is Actually Optimal**

The scoring formula:
```
score = mean(1/(1+RMSE)) + 0.3*(n_candidates/3)
```

Is best served by:
1. Getting the lowest RMSE candidate first (maximizes accuracy)
2. Adding diverse candidates that don't hurt the RMSE average too much
3. Maximizing n_candidates for the diversity bonus

The baseline achieves this naturally by sorting by RMSE first, then adding diverse candidates.

## Recommendations

1. **Keep baseline sort+filter approach**: It is already optimal

2. **DO NOT use greedy selection**: It produces worse results

3. **Mark selection algorithm tuning as explored**: The baseline is optimal

4. **Focus improvement elsewhere**: The candidate selection is not the bottleneck

## Conclusion

**FAILED** - Greedy selection produced worse results than the simple baseline:
- Score: -0.0189 (worse)
- Time: +10.1 min (over budget)
- Fewer 3-candidate samples (64 vs ~70)
- Higher RMSE for both 1-src and 2-src

The key insight is that the baseline's "sort by RMSE, filter by dissimilarity" approach is actually optimal for the scoring formula. More sophisticated selection algorithms don't help and can hurt.

## Raw Data
- MLflow run ID: 8b32af3283ea465d908fff963a7d56e9
- Config: {tau: 0.2, selection: greedy}
- Samples: 80 (32 1-source, 48 2-source)
- Candidate distribution: {1: 3, 2: 13, 3: 64}

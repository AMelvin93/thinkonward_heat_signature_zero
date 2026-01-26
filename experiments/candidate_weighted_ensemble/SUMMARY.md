# Experiment Summary: candidate_weighted_ensemble

## Metadata
- **Experiment ID**: EXP_CANDIDATE_ENSEMBLE_001
- **Worker**: W1
- **Date**: 2026-01-25
- **Algorithm Family**: ensemble_final

## Status: ABORTED (Prior Evidence Conclusive)

## Objective
Create a weighted ensemble of top-3 candidates instead of selecting the single best, to better exploit the diversity bonus (0.3 * N_valid/3) in the scoring formula.

## Why Aborted

This experiment is based on **two misconceptions** that have been **definitively disproven**:

### Misconception 1: Diversity Is the Bottleneck
The **niching_cmaes_diversity** experiment proved diversity is already saturated:

| Metric | Value |
|--------|-------|
| Average N_valid | 2.75 / 3 |
| Samples with 3 candidates | 80% |
| Max diversity gain possible | ~0.025 |

Quote: "Diversity is not the bottleneck: With N_valid already at 2.75/3, improving diversity can only gain 0.025 in the diversity term"

### Misconception 2: Weighted Ensemble Would Help

The scoring formula structure makes weighted ensembling counterproductive:

```
score = (1/N) * sum(1/(1+L_i)) + 0.3 * (N/3)
```

**The first term AVERAGES accuracy over ALL candidates.**

Example from niching_cmaes_diversity:
- 1 candidate with RMSE=0: score = 1.0 + 0.1 = **1.1**
- 3 candidates with RMSE=[0, 0.5, 0.5]: score = 0.78 + 0.3 = **1.08**

Adding two worse candidates **DECREASED** the score from 1.1 to 1.08!

## Prior Evidence Summary

### 1. niching_cmaes_diversity (EXP_NICHING_CMAES_001)
**Result**: FAILED - Score 1.0622 vs baseline 1.1688 (-0.1066)

Key findings:
1. "Baseline already achieves near-maximum diversity"
2. "The scoring formula penalizes diverse but worse candidates"
3. "Accuracy is the bottleneck" - improving RMSE is the only viable path
4. "Diversity optimization is NOT a viable path"

### 2. cmaes_restart_from_best
**Result**: FAILED - Diversity loss hurt score

Key finding: "Diversity is critical for scoring: The scoring formula heavily weights candidate diversity (0.3 * N_valid/3)"

Implication: The CURRENT baseline already optimizes for diversity. Trying to add MORE diversity hurts accuracy.

## Technical Explanation

### Why Weighted Ensemble Would NOT Help

The proposed approach has no path to improvement:

1. **If weights favor low-RMSE candidates**:
   - Result is essentially the same as taking the best candidate
   - No improvement over baseline

2. **If weights try to increase diversity**:
   - Worse candidates get included in average
   - Score decreases due to accuracy term averaging
   - Same failure mode as niching_cmaes_diversity

3. **The current approach is already optimal**:
   - CMA-ES produces 2-3 candidates naturally
   - NM polish improves accuracy of top-3
   - No room for improvement via ensemble

### Diversity Component Analysis

| Approach | N_valid | Diversity Term | Accuracy Term |
|----------|---------|---------------|---------------|
| Baseline | 2.75 | 0.275 | Optimized |
| Force 3 | 3.0 | 0.300 | Worse (average includes bad candidates) |
| **Delta** | +0.25 | +0.025 | **Negative** (more than -0.025) |

The math shows: forcing more candidates is a net negative.

## Algorithm Family Status

- **diversity (all variants)**: **EXHAUSTED**
- **ensemble_final**: **EXHAUSTED**

Key insight: The problem typically has ONE global optimum per sample. Trying to find multiple distinct solutions means finding suboptimal solutions.

## Recommendations

1. **Do NOT pursue diversity optimization** - it's already saturated at 2.75/3
2. **Focus on accuracy** - the scoring formula rewards low RMSE more than diversity
3. **Single best solution matters** - finding the ONE global optimum is more important
4. **ensemble_final family is EXHAUSTED** - no viable ensemble approach exists

## Raw Data
- MLflow run IDs: None (experiment not executed)
- Prior evidence: niching_cmaes_diversity, cmaes_restart_from_best, weighted_centroid_nm

## Conclusion

This experiment would fail because (1) diversity is already saturated at 2.75/3 candidates, (2) the scoring formula's accuracy term averages over all candidates so adding worse candidates hurts more than the diversity bonus helps, and (3) the thermal inverse problem has one global optimum per sample, not multiple distinct local optima.

The diversity and ensemble families are EXHAUSTED. No further diversity optimization experiments should be created.

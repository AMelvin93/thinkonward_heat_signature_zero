# Experiment Summary: median_ensemble_solution

## Status: ABORTED (Prior Evidence)

## Experiment ID: EXP_MEDIAN_ENSEMBLE_001
## Worker: W1
## Date: 2026-01-26

## Hypothesis
Use coordinate-wise median instead of mean for position averaging, which may be more robust to outliers.

## Why Aborted

### Ensemble Averaging Already Failed
Prior experiment `top3_ensemble_averaging` (EXP_TOP3_ENSEMBLE_001) tested position averaging:

**Result: FAILED** - Score 1.1129 vs baseline 1.1688 (-4.8%)

### Median vs Mean Won't Fix Root Cause
The failure of ensemble averaging was NOT due to outliers corrupting the mean. The fundamental issues are:

1. **Source permutation problem**: For 2-source cases, sources may be inconsistently ordered across candidates
2. **Mixing different local optima**: Candidates from different basins should not be averaged
3. **Destroying precise refinement**: NM polish positions are already optimized; any averaging degrades them

Using median instead of mean:
- Still averages positions (just with different weighting)
- Still suffers from permutation issues
- Still mixes potentially distinct solutions
- May be slightly more robust to single outlier, but outliers are not the problem

### Recommendation
**Do NOT pursue any position-averaging ensemble approach.** The problem is not outliers vs central tendency - it's that averaging optimized positions is fundamentally the wrong approach for this inverse problem.

## Related Experiments
- `top3_ensemble_averaging`: FAILED with score 1.1129 (-4.8%)

## Conclusion
**Median ensemble would fail for the same reasons as mean ensemble.** Keep candidates as discrete solutions rather than trying to combine them spatially.

# Experiment Summary: micro_restart_polish

## Metadata
- **Experiment ID**: EXP_HYBRID_RESTART_001
- **Worker**: W2
- **Date**: 2026-01-25
- **Algorithm Family**: restart_v3

## Status: ABORTED

## Why Aborted
Prior evidence shows **ALL polish modifications FAIL**:

1. **extended_nm_polish (12 iters)**: FAILED - over budget by 37% for only +0.0015 score
2. **powell_polish_instead_nm**: FAILED - 5-8x slower due to line search overhead
3. **adaptive_nm_coefficients**: FAILED - 45% slower, -1.7% score
4. **progressive_polish_fidelity**: ABORTED - truncated polish overfits to noise
5. **split_polish_fidelity**: ABORTED - same issue as above
6. **weighted_centroid_nm**: FAILED - biased NM toward single local minimum
7. **L-BFGS-B polish**: FAILED - finite difference overhead

**The baseline (NM x8 with default coefficients and full timesteps) is OPTIMAL.**

## Conclusion
Any modification to the polish method results in either:
- Slower convergence (more time)
- Worse accuracy (lower score)
- Or both

The restart_v3 family is EXHAUSTED. No further experiments on polish method modifications should be attempted.

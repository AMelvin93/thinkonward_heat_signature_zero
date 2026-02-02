# Sigma 0.15/0.19 + Fevals 22/40 Experiment

## Hypothesis
Combine optimal sigma (0.15/0.19) with sweet spot fevals (22/40). Prior tests optimized these independently:
- tighter_sigma_range: sigma 0.15/0.19 + 20/36 fevals = 1.173
- higher_fevals_test: sigma 0.18/0.22 + 22/40 fevals = 1.1640

If both optimizations are independent, combining them should yield even better results.

## Result: FAILED

Best result: **1.1358 @ 44.9 min** (-0.0372 vs baseline)

## Tuning Summary

| Run | Sigma | Fevals | Perturb | Score | Avg Cands | Time | Decision |
|-----|-------|--------|---------|-------|-----------|------|----------|
| 1 | 0.15/0.19 | 22/40 | 2 | 1.1268 | 2.26 | 41.3 | FAILED - diversity hurt |
| 2 | 0.15/0.19 | 22/40 | 3 | 1.1358 | 2.76 | 44.9 | FAILED - still worse |
| 3 | 0.16/0.20 | 22/40 | 2 | 1.1281 | 2.79 | 45.3 | FAILED - intermediate also bad |

## Key Finding: Parameter Interdependence

**Sigma and fevals are NOT independent parameters!**

The optimal fevals depends on the sigma value:
- **Sigma 0.18/0.22**: Optimal with 22/40 fevals (larger exploration needs more evaluations)
- **Sigma 0.15/0.19**: Optimal with 20/36 fevals (tighter exploration converges faster)

When sigma is tighter (0.15/0.19), higher fevals (22/40) causes:
1. CMA-ES converges too quickly to a single basin
2. All population members become similar
3. Dissimilarity filter removes duplicate candidates
4. Fewer diverse candidates = lower score (scoring formula penalizes low diversity)

## Evidence of Diversity Loss

| Run | Avg N_valid | Expected | Impact |
|-----|-------------|----------|--------|
| Baseline | ~2.75 | 3.0 | Normal |
| Run 1 | 2.26 | 3.0 | **-0.5 candidates lost** |
| Run 2 | 2.76 | 3.0 | Recovered with 3 perturbations |
| Run 3 | 2.79 | 3.0 | Still not optimal |

## Conclusion

Cannot combine sigma and fevals optimizations independently. They must be tuned together.

**Optimal configurations remain:**
- sigma 0.15/0.19 + 20/36 fevals = 1.173 (current best)
- sigma 0.18/0.22 + 22/40 fevals = 1.1640 (alternative)

**Family**: combined_sigma_fevals - EXHAUSTED

---
**Worker**: W2
**Completed**: 2026-02-01
**Runs**: 3

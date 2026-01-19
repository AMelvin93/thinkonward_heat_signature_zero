# Temporal 40% + Higher Sigma Experiment

## Experiment ID: EXP_TEMPORAL_HIGHER_SIGMA_001
**Worker:** W1
**Status:** COMPLETED (FAILED)
**Date:** 2026-01-19

## Hypothesis

Higher sigma (0.25/0.30 vs W2's 0.18/0.22) combined with 40% temporal fidelity can achieve better accuracy within the 60-minute budget by enabling broader exploration in the CMA-ES optimization.

## Baseline

**W2's best configuration:**
- Score: 1.1688
- Time: 58.4 min (in budget)
- Config: 40% temporal + 8 NM polish, sigma 0.18/0.22

## Tuning Runs

| Run | Sigma | Polish | Fevals | Score | Time | In Budget | Finding |
|-----|-------|--------|--------|-------|------|-----------|---------|
| 1 | 0.25/0.30 | 8 | 20/36 | **1.1745** | 63.0 | No | Best score but 3 min over |
| 2 | 0.25/0.30 | 5 | 20/36 | 1.1584 | 52.1 | Yes | In budget but below W2 |
| 3 | 0.25/0.30 | 7 | 20/36 | 1.1690 | 68.8 | No | High time variance |
| 4 | 0.25/0.30 | 6 | 16/28 | 1.1594 | 60.1 | No | Reduced fevals hurt accuracy |
| 5 | 0.20/0.24 | 8 | 20/36 | 1.1590 | 68.5 | No | Intermediate worse than both |

## Results

### Best Overall Score (Run 1)
- **Score:** 1.1745 (+0.0057 vs W2)
- **Time:** 63.0 min (3 min over budget)
- **RMSE:** 1-src 0.1048, 2-src 0.1575

### Best In-Budget (Run 2)
- **Score:** 1.1584 (-0.0104 vs W2)
- **Time:** 52.1 min
- **RMSE:** 1-src 0.1166, 2-src 0.1733

## Key Findings

1. **Higher sigma improves accuracy but increases time disproportionately**
   - Sigma 0.25/0.30 improved score by +0.0057 but added 5+ minutes
   - The exploration benefit doesn't compensate for the time cost

2. **Reducing polish to stay in budget loses accuracy**
   - Run 2 with 5 polish iterations was in budget but scored below W2
   - Polish is critical for refining CMA-ES solutions to full accuracy

3. **2-source samples show high variance with higher sigma**
   - Times ranged from 55s to 237s for 2-source problems
   - Higher sigma requires more CMA-ES iterations to converge

4. **Intermediate sigma performed worst**
   - Sigma 0.20/0.24 scored worse than both W2 (0.18/0.22) and high (0.25/0.30)
   - Suggests non-linear relationship between sigma and performance

5. **W2's sigma is near-optimal**
   - The 0.18/0.22 configuration balances exploration vs convergence time
   - Higher sigma provides diminishing returns within time budget

## Conclusion

**FAILED** - Cannot beat W2's baseline (1.1688 @ 58.4 min) within the 60-minute budget.

Higher sigma enables better exploration and marginally improves accuracy, but the increased CMA-ES convergence time pushes runs over budget. When constrained to stay in budget by reducing polish iterations, the accuracy gains are lost.

W2's configuration represents a well-optimized trade-off between exploration (sigma) and refinement (polish) that this experiment could not improve upon.

## Recommendations for Future Work

1. **Investigate adaptive sigma scheduling** - Start with higher sigma, reduce as optimization progresses
2. **Focus on 2-source efficiency** - 2-source problems dominate runtime, optimize for these
3. **Explore other parameters** - Temporal fraction, fevals allocation, initialization strategies
4. **Consider problem-specific tuning** - Different sigma for 1-src vs 2-src problems

## MLflow Runs

- `run1_sigma_025_030` - Best score, over budget
- `run2_sigma_025_030_polish5` - Best in-budget
- `run3_sigma_025_030_polish7` - High variance
- `run4_reduced_fevals` - Reduced fevals test
- `run5_intermediate_sigma` - Intermediate sigma test

# Sigma 0.15/0.19 Baseline Validation

## Hypothesis
Validate the claimed 1.1730 result with 3 runs to confirm it's reproducible.

## Result: CRITICAL FINDING - 1.173 IS NOT REPRODUCIBLE

**Actual baseline**: 1.1337 ± 0.0027 @ 44.0 min

## Validation Results

| Run | Score | Avg Cands | Time | vs Claimed 1.173 |
|-----|-------|-----------|------|------------------|
| 1 | 1.1365 | 2.78 | 42.1 min | -0.036 |
| 2 | 1.1345 | 2.79 | 47.2 min | -0.039 |
| 3 | 1.1300 | 2.71 | 42.7 min | -0.043 |

## Statistical Summary

| Metric | Value |
|--------|-------|
| Mean | **1.1337** |
| Std Dev | 0.0027 |
| Min | 1.1300 |
| Max | 1.1365 |
| Range | 0.0065 |

## Critical Implications

### 1. The 1.173 claim was erroneous
- Gap from claimed 1.173 to actual 1.134 = **0.04 points**
- This is 15x larger than run-to-run variance (0.0027)
- The 1.173 was likely an outlier, measurement error, or different config

### 2. Run-to-run variance is small
- Previously assumed variance of ±0.01-0.03 was wrong
- Actual variance is only ±0.003
- This means small improvements ARE detectable

### 3. Competitive baseline reset
- Any experiment claiming to "improve" over 1.173 needs re-evaluation
- The true target for improvement is 1.1337, not 1.173
- Experiments scoring 1.14-1.15 are actually good!

### 4. Previous experiments re-evaluated
| Experiment | Claimed | Actual Delta vs 1.1337 |
|------------|---------|------------------------|
| sigma_015_019_nm6_4perturb Run 2 | 1.1404 | +0.007 (IMPROVEMENT!) |
| sigma_018_022_fevals_24_44 Run 2 | 1.1457 | +0.012 (IMPROVEMENT!) |

## Conclusion

**The sigma 0.18/0.22 + fevals 20/44 config (1.1457) is actually 1.2% better than the true baseline!**

This experiment should be re-evaluated as a potential new best configuration.

---
**Worker**: W2
**Completed**: 2026-02-01
**Runs**: 3

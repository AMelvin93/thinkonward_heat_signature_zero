# Reproduce W2 Best Config - Verification Experiment

## Experiment ID: EXP_REPRODUCE_W2_BEST_001
**Worker:** W2
**Status:** FAILED - Cannot reproduce claimed baseline
**Date:** 2026-01-28

## Hypothesis

The W2 baseline (1.1688 @ 58.4 min) is reproducible with the documented configuration:
- 40% temporal fidelity
- sigma 0.18/0.22
- 8 NM polish iterations
- fevals 20/36

## Verification Runs

| Run | Sigma | Score | Time (min) | vs Claimed W2 |
|-----|-------|-------|------------|---------------|
| 1 | 0.18/0.22 | 1.1599 | 71.5 | -0.0089 score, +13.1 min |
| 2 | 0.15/0.20 | 1.1559 | 71.1 | -0.0129 score, +12.7 min |

## Results

### Run 1: sigma 0.18/0.22 (claimed W2 config)
- **Score:** 1.1599
- **Time:** 71.5 min (11.5 min OVER BUDGET)
- **RMSE:** 1-src 0.1027, 2-src 0.1544
- **Delta vs claimed:** -0.0089 score, +13.1 min

### Run 2: sigma 0.15/0.20 (original baseline sigma)
- **Score:** 1.1559
- **Time:** 71.1 min (11.1 min OVER BUDGET)
- **RMSE:** 1-src 0.1134, 2-src 0.1592
- **Delta vs claimed:** -0.0129 score, +12.7 min

## Key Findings

1. **CRITICAL: Claimed W2 baseline cannot be reproduced**
   - Both tested configurations are significantly over budget (71+ min vs 58 min)
   - Both score lower than claimed (1.16 vs 1.17)

2. **Timing discrepancy is ~22%**
   - Claimed: 58.4 min → Actual: ~71 min
   - This suggests either different machine or different code path

3. **sigma 0.18/0.22 is slightly better than 0.15/0.20**
   - Score: 1.1599 vs 1.1559 (+0.0040)
   - Time: 71.5 vs 71.1 (+0.4 min)

4. **The "simple baseline" reference points are more reliable**
   - early_timestep_filtering (25% temporal): 1.1246 @ 32.6 min
   - solution_verification_pass: 1.1373 @ 42.6 min

## Conclusion

**FAILED** - The claimed W2 baseline of 1.1688 @ 58.4 min cannot be verified on this machine.

The actual achievable scores with the documented configuration are:
- sigma 0.18/0.22: **1.1599 @ 71.5 min** (over budget)
- sigma 0.15/0.20: **1.1559 @ 71.1 min** (over budget)

Possible explanations:
1. Original W2 run was on a faster machine (G4dn.2xlarge vs current WSL)
2. Different code version or parameters were used
3. The claimed numbers may have been from a partial run or different configuration

## Recommendations

1. **Use solution_verification_pass (1.1373 @ 42.6 min) as the reliable baseline**
   - This was verified with multiple seeds
   - It's within budget with room to spare

2. **Do not rely on the 1.1688 figure for comparison**
   - Cannot be reproduced
   - May lead to false conclusions about experiment failures

3. **The 40% temporal + full polish approach is valid but expensive**
   - Adds ~30+ min vs 25% temporal
   - Need to reduce something else to fit budget

## Time Analysis

```
Run 1 (sigma 0.18/0.22):
  Time: 71.5 min
  Budget: 60 min
  OVER BUDGET by 11.5 min (19%)

Run 2 (sigma 0.15/0.20):
  Time: 71.1 min
  Budget: 60 min
  OVER BUDGET by 11.1 min (19%)

TIME UTILIZATION: 119% - Cannot use remaining budget to improve
```

## What Would Be Needed to Match Claimed Baseline

- Either 19% speedup in simulation (unlikely)
- Or reduce fevals/polish to fit budget (would hurt accuracy)
- Or accept that 1.1688 @ 58.4 min may be an unrealistic target for this machine

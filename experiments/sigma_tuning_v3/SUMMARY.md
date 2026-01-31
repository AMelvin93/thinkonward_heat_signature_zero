# Experiment: sigma_tuning_v3

## Objective
Test if different sigma values can improve accuracy on top of the new best config (fevals_22_40).

## Hypothesis
Current sigma 0.18/0.22 may not be optimal. Tighter sigma could converge faster, while looser sigma could explore more broadly.

## Baselines
| Config | Score | Time (min) | In Budget | sigma 1src | sigma 2src |
|--------|-------|------------|-----------|------------|------------|
| fevals_22_40 | 1.1640 | 57.1 | YES | 0.18 | 0.22 |

## Results

| Config | Score | Time (min) | In Budget | RMSE 1-src | RMSE 2-src | Delta |
|--------|-------|------------|-----------|------------|------------|-------|
| tighter_sigma | 1.1523 | 56.6 | YES | 0.1309 | 0.2157 | **-0.0117** |
| looser_sigma | 1.1626 | 56.8 | YES | 0.1236 | 0.1949 | -0.0014 |
| asymmetric_sigma | 1.1624 | 57.5 | YES | 0.1284 | 0.1906 | -0.0016 |

## Analysis

### Tighter Sigma Hurts Significantly
Reducing sigma from 0.18/0.22 to 0.16/0.20:
- **Score drops by 0.0117** (1% relative)
- 2-source RMSE increases from 0.189 to 0.216
- The tighter exploration doesn't find as good basins

### Looser Sigma Slightly Worse
Increasing sigma from 0.18/0.22 to 0.20/0.25:
- Score drops by 0.0014 (negligible but consistent)
- 1-source RMSE improves, but 2-source slightly worse
- More exploration doesn't help for this problem

### Asymmetric Sigma Also Fails
Keeping 1src at 0.18 but increasing 2src to 0.25:
- Score drops by 0.0016
- 2-source RMSE improves, but 1-source worsens
- No net benefit from asymmetric approach

## Key Findings

1. **Current sigma 0.18/0.22 is optimal** - Both directions hurt accuracy
2. **Tighter sigma is dangerous** - Largest negative impact (-0.0117)
3. **Looser sigma doesn't help** - More exploration yields diminishing returns
4. **sigma_v2 family is EXHAUSTED** - No further tuning needed

## Comparison with Prior Sigma Experiments
| Experiment | sigma 1src | sigma 2src | Score |
|------------|------------|------------|-------|
| sigma_tuning_v3 tighter | 0.16 | 0.20 | 1.1523 |
| fevals_22_40 (current best) | 0.18 | 0.22 | 1.1640 |
| sigma_tuning_v3 looser | 0.20 | 0.25 | 1.1626 |

The pattern is clear: 0.18/0.22 sits at the optimal exploration/exploitation balance.

## Recommendations

1. **Keep sigma 0.18/0.22** - Already optimal
2. **fevals_22_40 remains best config** - Score 1.1640 @ 57.1 min
3. **Mark sigma_v2 family as EXHAUSTED** - No further sigma tuning needed

## Conclusion
**FAILED** - No sigma variation improves on the baseline. Current sigma 0.18/0.22 is optimal.

## Family Status
`sigma_v2` - **EXHAUSTED** - Sigma 0.18/0.22 is optimal for this problem.

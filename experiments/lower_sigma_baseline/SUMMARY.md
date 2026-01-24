# Lower Sigma Baseline Experiment

## Experiment ID: EXP_LOWER_SIGMA_001

## Status: FAILED

## Hypothesis
Lower sigma (0.15/0.18) may converge faster and more precisely than baseline (0.15/0.20).

## Results

### Run 1: Lower Sigma (0.15/0.18)
| Metric | Value | Baseline | Delta |
|--------|-------|----------|-------|
| Score | 1.1335 | 1.1688 | -0.0353 |
| Projected (400) | 39.6 min | 58.4 min | -18.8 min |
| RMSE (1-src) | 0.1261 | ~0.10 | +0.03 |
| RMSE (2-src) | 0.2001 | ~0.16 | +0.04 |

**Result:** WORSE - Lower sigma is too local, doesn't explore the search space sufficiently.

### Run 2: Higher Sigma (0.18/0.22)
| Metric | Value | Baseline | Delta |
|--------|-------|----------|-------|
| Score | 1.1379 | 1.1688 | -0.0309 |
| Projected (400) | 38.5 min | 58.4 min | -19.9 min |
| RMSE (1-src) | 0.1354 | ~0.10 | +0.04 |
| RMSE (2-src) | 0.2076 | ~0.16 | +0.05 |

**Result:** WORSE - Higher sigma explores too broadly, converges to worse local optima.

## Key Findings

1. **Baseline sigma (0.15/0.20) is locally optimal** - Both lower and higher sigma variants perform worse.

2. **2-source problems are most affected** - The RMSE difference is larger for 2-source (0.04-0.05) than 1-source (0.03-0.04), indicating the 4D search space is more sensitive to sigma tuning.

3. **Time improvement doesn't compensate** - While lower sigma runs faster (39.6 min vs 58.4 min), the score drop far outweighs the time savings.

4. **Sigma sensitivity is symmetric** - Both directions from baseline produce similar degradation (~0.03 score drop), suggesting the baseline is at or near the optimal sigma configuration.

## Sigma Sensitivity Analysis

| Sigma Config | Score | vs Baseline |
|--------------|-------|-------------|
| 0.15/0.18 (lower) | 1.1335 | -0.0353 |
| **0.15/0.20 (baseline)** | **1.1688** | — |
| 0.18/0.22 (higher) | 1.1379 | -0.0309 |

## Conclusion

**The baseline sigma configuration (0.15/0.20) is optimal.** Further experiments in the `cmaes_tuning` family should focus on other parameters (fevals, thresholds, population size) rather than sigma.

## Recommendation

Mark the sigma tuning space as EXHAUSTED. The current sigma values are at a local optimum and should not be changed.

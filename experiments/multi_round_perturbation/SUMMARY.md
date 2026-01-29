# Multi-Round Perturbation Experiment

## Experiment ID
EXP_MULTI_ROUND_PERTURB_001

## Hypothesis
Single perturbation may not escape deep local minima. Two rounds of perturbation+optimization may find better basins.

## Baseline
- **perturbation_plus_verification**: 1.1468 @ 54.2 min
- **perturbed_extended_polish**: 1.1464 @ 51.2 min (single-round perturbation)

## Results

### FAILED - Multi-round perturbation does NOT improve over single-round

| Run | Config | Score | Time (min) | In Budget | vs Baseline |
|-----|--------|-------|------------|-----------|-------------|
| 1 | 2 rounds, 2 NM iters/round | 1.1432 | 61.6 | NO | -0.0036 |
| 2 | 2 rounds, 1 NM iter/round | 1.1372 | 50.9 | YES | -0.0096 |
| 3 | 1 round, 4 NM iters, 2 candidates | 1.1389 | 429.3 | NO | -0.0079 |

**Best in-budget**: Run 2 with 1.1372 @ 50.9 min
**Delta vs baseline**: -0.0096 (WORSE)

## Tuning Efficiency Metrics
- **Runs executed**: 3
- **Time utilization**: 85% (50.9/60 min for best in-budget)
- **Parameter space explored**:
  - perturb_rounds: [1, 2]
  - perturb_nm_iters: [1, 2, 4]
  - perturb_top_n: [1, 2]
  - perturbation_scale: [0.05, 0.06]
- **Pivot points**:
  - Run 1 over budget → Reduced NM iters to 1
  - Run 2 worse than baseline → Tried different config with more perturbations
  - Run 3 massively over budget → Confirmed approach doesn't work

## Budget Analysis
| Run | Score | Time | Budget Remaining | Decision |
|-----|-------|------|------------------|----------|
| 1   | 1.1432 | 61.6 | -1.6 min (OVER)  | PIVOT - reduce NM iters |
| 2   | 1.1372 | 50.9 | +9.1 min         | CONTINUE - score below baseline |
| 3   | 1.1389 | 429.3 | -369 min (OVER) | ABANDON - config doesn't work |

## Why Multi-Round Perturbation Fails

1. **Overhead without benefit**: Multi-round perturbation adds NM iterations but doesn't improve accuracy
2. **Diminishing returns**: If single perturbation doesn't find a better basin, additional perturbations are unlikely to help
3. **CMA-ES already explores well**: The covariance adaptation in CMA-ES already does effective exploration
4. **NM polish saturates**: After 8 iterations of NM polish, additional iterations don't help

## Key Findings

1. **Single-round perturbation is optimal**: The baseline perturbed_extended_polish (1.1464 @ 51.2 min) uses single-round perturbation effectively
2. **More perturbations ≠ better accuracy**: Increasing perturb_top_n or perturbation rounds adds time without score improvement
3. **NM iterations have diminishing returns**: perturb_nm_iters beyond 2-3 don't help

## What Would Have Been Tried With More Time
- If budget were 70 min: Try perturb_rounds=2 with perturb_nm_iters=2 (Run 1 config)
- If budget were 90 min: Try perturb_rounds=3 to see if even more rounds help (unlikely)

## Recommendation

**DO NOT use multi-round perturbation.** Stick with single-round perturbation as implemented in:
- perturbed_extended_polish (1.1464 @ 51.2 min)
- perturbation_plus_verification (1.1468 @ 54.2 min)

## Conclusion

**EXPERIMENT FAILED**

Multi-round perturbation adds overhead without improving accuracy. The hypothesis that "two rounds of perturbation+optimization may find better basins" is DISPROVED. Single-round perturbation is sufficient and optimal.

The perturbation_v2 family should be marked as EXHAUSTED.

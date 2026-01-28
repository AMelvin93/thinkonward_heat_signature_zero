# Experiment Summary: perturbation_plus_verification

## Metadata
- **Experiment ID**: EXP_PERTURB_PLUS_VERIFY_001
- **Worker**: W2
- **Date**: 2026-01-28
- **Algorithm Family**: hybrid_v3

## Status: MARGINAL SUCCESS

## Objective
Combine two successful approaches:
1. Basin hopping via perturbation (from perturbed_extended_polish - 1.1464 @ 51.2 min)
2. Gradient verification (from solution_verification_pass - 1.1373 @ 42.6 min)

## Hypothesis
Perturbation (+0.0079 on simpler config) and verification (+0.0127 on simpler config) might stack for additional improvement.

## Baseline
- **Current best (perturbed_extended_polish)**: 1.1464 @ 51.2 min
- **solution_verification_pass**: 1.1373 @ 42.6 min

## Results Summary

| Run | perturb_top_n | n_perturbations | Score | Time (min) | Budget | Status |
|-----|---------------|-----------------|-------|------------|--------|--------|
| 1   | 1             | 2               | **1.1468** | **54.2** | **IN** | **Best in-budget** |
| 2   | 2             | 2               | 1.1525 | 66.8 | OVER   | Better score but over budget |
| 3   | 2             | 1               | 1.1436 | 64.8 | OVER   | Worse score, still over budget |

**Best in-budget**: Run 1 with score **1.1468 @ 54.2 min** (+0.0004 vs baseline)

## Tuning Efficiency Metrics
- **Runs executed**: 3
- **Time utilization**: 90% (54.2/60 min used at best in-budget)
- **Parameter space explored**: perturb_top_n=[1,2], n_perturbations=[1,2]
- **Pivot points**:
  - Run 1→2: Increased perturb_top_n to 2 (over budget but better score)
  - Run 2→3: Reduced n_perturbations to 1 (still over budget, worse score)

## Key Findings

### 1. Perturbation and Verification Help Different Samples
Run 1 statistics:
- Perturbed candidates selected: 17/80 samples (21%)
- Verified candidates selected: 11/80 samples (14%)
- Only 1 sample had BOTH selected

This confirms the two mechanisms address different failure modes:
- Perturbation escapes shallow local minima
- Verification catches incomplete convergence

### 2. Marginal Improvement Only (+0.0004)
Despite helping 28 samples (35%), the final score improvement is only +0.0004:
- Perturbation helps 21% of samples
- Verification helps 14% of samples
- But the improvements per sample are small

### 3. More Perturbations = More Time Without Proportional Benefit
Run 2 (perturb_top_n=2) showed:
- +0.0061 score improvement (better)
- +15.6 min overhead (over budget)
- The extra time isn't worth the score gain

### 4. Coarse Grid Verification is Key
Using coarse grid for verification (instead of fine grid) made the approach feasible:
- Fine grid verification: ~200 min projected (3.5x over budget)
- Coarse grid verification: ~54 min projected (in budget)

## Budget Analysis

| Run | Score | Time | Budget Remaining | Decision |
|-----|-------|------|------------------|----------|
| 1   | 1.1468| 54.2 | 5.8 min         | TRY MORE |
| 2   | 1.1525| 66.8 | -6.8 min        | PIVOT    |
| 3   | 1.1436| 64.8 | -4.8 min        | ACCEPT Run 1 |

## Optimal Configuration

```python
config = {
    "enable_perturbation": True,
    "perturb_top_n": 1,
    "n_perturbations": 2,
    "perturbation_scale": 0.05,
    "perturb_nm_iters": 3,
    "enable_verification": True,  # On COARSE grid
    "gradient_eps": 0.02,
    "gradient_threshold": 0.1,
    "step_size": 0.05,
    "max_fevals_1src": 20,
    "max_fevals_2src": 36,
    "timestep_fraction": 0.4,
    "refine_maxiter": 8,
    "refine_top_n": 2,
    "sigma0_1src": 0.18,
    "sigma0_2src": 0.22,
}
```

## Comparison to Previous Results

| Approach | Score | Time | Delta vs this |
|----------|-------|------|---------------|
| perturbation_plus_verification (this) | **1.1468** | 54.2 min | - |
| perturbed_extended_polish | 1.1464 | 51.2 min | -0.0004, -3.0 min |
| solution_verification_pass | 1.1373 | 42.6 min | -0.0095, -11.6 min |

## Recommendations

1. **DO NOT ADOPT** - Marginal improvement (+0.0004) not worth the +3 min overhead
   - perturbed_extended_polish remains the best approach

2. **Key Insight**: Perturbation and verification help DIFFERENT samples but the improvements DON'T stack significantly
   - Perturbation: 21% of samples benefit
   - Verification: 14% of samples benefit
   - But combined improvement is only 0.03% better

3. **The improvements are orthogonal but NOT additive**
   - Each mechanism provides small per-sample improvements
   - Combining them helps more samples but doesn't improve individual samples

## What Would Have Been Tried With More Time
- If budget were 70 min: perturb_top_n=2 would have achieved 1.1525 score
- If budget were 90 min: Try verify_top_n=2 (verify multiple candidates)

## Conclusion

**MARGINAL SUCCESS** - The combined perturbation + verification approach achieves a marginally better score (1.1468 vs 1.1464, +0.0004) but costs 3 more minutes (54.2 vs 51.2 min). The improvement is so small it may be within noise.

**Key insight**: Perturbation and verification address different failure modes but the improvements don't compound significantly. The baseline perturbed_extended_polish (1.1464 @ 51.2 min) remains the best approach for practical use.

## Raw Data
- Experiment directory: `experiments/perturbation_plus_verification/`
- Files: `optimizer.py`, `run.py`, `STATE.json`

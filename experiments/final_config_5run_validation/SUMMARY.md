# Experiment: final_config_5run_validation

## Status: FAILED - Config does NOT reliably beat baseline

## Hypothesis
Validate final optimal config (sigma 0.14/0.19, fevals 20/44, 2 perturb, scale 0.06) with 5 runs to establish confidence intervals.

## Bug Fix Note
**W1's original run.py had `enable_tabu_hopping=False`**, which disabled perturbations entirely.
- Previous (buggy) results: 1.1267-1.1358 (all below baseline)
- W2 fixed this to `enable_tabu_hopping=True` and re-ran all 5 validation runs.

## Configuration Tested (FIXED)
```python
config = {
    'sigma0_1src': 0.14,
    'sigma0_2src': 0.19,
    'max_fevals_1src': 20,
    'max_fevals_2src': 44,
    'timestep_fraction': 0.40,
    'refine_maxiter': 8,
    'enable_tabu_hopping': True,  # FIXED: Was False in W1's version
    'n_perturbations': 2,
    'perturb_nm_iters': 3,
    'perturbation_scale': 0.06,
    'tabu_distance': 0.03,
    'max_tabu_attempts': 10,
}
```

## Results (5 Runs, Fixed Config)

| Run | Score | Time (proj 400) | RMSE 1src | RMSE 2src | vs Baseline |
|-----|-------|-----------------|-----------|-----------|-------------|
| 1   | 1.1430 | 51.5 min | 0.1195 | 0.1760 | -0.0034 |
| 2   | **1.1471** | 59.0 min | 0.1167 | 0.1898 | **+0.0007** |
| 3   | 1.1349 | 51.1 min | 0.1311 | 0.1852 | -0.0115 |
| 4   | 1.1238 | 52.2 min | 0.1293 | 0.2101 | -0.0226 |
| 5   | 1.1412 | 51.8 min | 0.1120 | 0.1908 | -0.0052 |

**Baseline**: 1.1464 @ 51.2 min

## Statistics
| Metric | Value |
|--------|-------|
| **Mean Score** | **1.1380 +/- 0.0081** |
| **Mean Time** | 53.1 +/- 3.0 min |
| Score Range | [1.1238, 1.1471] |
| **vs Baseline** | **-0.0084** |
| Runs above baseline | **1 out of 5 (20%)** |

## Key Findings

### 1. Config Does NOT Beat Baseline Reliably
- Only **1 out of 5 runs** (20%) exceeded baseline
- Mean score (1.1380) is **0.0084 below baseline**
- The config underperforms the claimed baseline

### 2. High Run-to-Run Variance
- Score std: 0.0081 (significant)
- Range: 0.0233 (from 1.1238 to 1.1471)
- This variance makes it difficult to trust single-run results

### 3. Asymmetric Sigma 0.14/0.19 Not Validated
- Previous claims of 1.17+ scores with this sigma were likely lucky runs
- The true expected value is closer to 1.138

### 4. Perturbation Scale 0.06 Effect Unclear
- With high variance, cannot determine if scale 0.06 is optimal
- The perturbation effect is masked by noise

## Comparison to Original Baseline (W2's 1.1688)

The original W2 baseline claimed:
- Score: 1.1688 @ 58.4 min
- Config: sigma 0.18/0.22, fevals 20/40, 40% temporal

This validation config (sigma 0.14/0.19):
- Mean: 1.1380 @ 53.1 min
- **0.0308 below the claimed baseline**

## RMSE Analysis

| Metric | 1-source | 2-source |
|--------|----------|----------|
| Run 1 | 0.1195 | 0.1760 |
| Run 2 | 0.1167 | 0.1898 |
| Run 3 | 0.1311 | 0.1852 |
| Run 4 | 0.1293 | 0.2101 |
| Run 5 | 0.1120 | 0.1908 |
| **Mean** | **0.1217** | **0.1904** |

2-source RMSE remains the bottleneck (~0.19).

## Conclusion

**RESULT: FAILED - Config does NOT reliably beat baseline**

The asymmetric sigma 0.14/0.19 configuration:
1. Does NOT provide consistent improvement over baseline
2. Has high variance making comparisons unreliable
3. Should NOT be used for production

## Recommendations

1. **Revert to original baseline config**: sigma 0.18/0.22 appears more reliable
2. **More validation runs needed**: Consider 10+ runs for reliable comparisons
3. **Focus on reducing variance**: Explore techniques to stabilize results
4. **Re-evaluate prior "improvements"**: Many claimed improvements may be noise

## Tuning Efficiency Metrics
- **Runs executed**: 5 (as planned)
- **Time utilization**: 88.5% (53.1/60 min average)
- **Parameter space explored**: Fixed config validation only
- **Bug fixed**: enable_tabu_hopping corrected from False to True

## Budget Analysis
| Run | Score | Time | Budget Remaining | In Budget |
|-----|-------|------|------------------|-----------|
| 1   | 1.1430 | 51.5 | 8.5 min | YES |
| 2   | 1.1471 | 59.0 | 1.0 min | YES |
| 3   | 1.1349 | 51.1 | 8.9 min | YES |
| 4   | 1.1238 | 52.2 | 7.8 min | YES |
| 5   | 1.1412 | 51.8 | 8.2 min | YES |

All runs within budget, mean ~7.0 min remaining.

---
**Worker**: W2 (fixed bug, re-ran validation)
**Completed**: 2026-02-02
**Runs**: 5 (systematic validation)

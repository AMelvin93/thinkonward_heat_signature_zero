# Experiment Summary: informative_sensor_subset

## Metadata
- **Experiment ID**: EXP_SENSOR_SUBSET_001
- **Worker**: W2
- **Date**: 2026-01-24
- **Algorithm Family**: sensor_optimization

## Objective
Use only most informative sensors (by variance/SNR) for optimization to reduce noise and potentially improve accuracy.

## Hypothesis
Not all sensors contribute equally to source localization. Using a subset of most informative sensors may improve signal-to-noise ratio and reduce computation.

## Results Summary
- **Best In-Budget Score**: N/A (not implemented)
- **Best Overall Score**: N/A
- **Baseline Comparison**: N/A
- **Status**: **ABORTED** - Subset RMSE is poor proxy for full RMSE (r=0.68)

## Key Findings

### Finding 1: Sensor Count Distribution

| Sensors | Samples | Percentage |
|---------|---------|------------|
| 2 | 8 | 10% |
| 3 | 24 | 30% |
| 4 | 24 | 30% |
| 5 | 16 | 20% |
| 6 | 8 | 10% |

Most samples have 3-6 sensors, so subset selection is technically possible.

### Finding 2: Subset RMSE Correlation is LOW

| Metric | Value |
|--------|-------|
| Full vs Top-2 RMSE correlation | r = 0.6845 |
| p-value | 4.2e-08 |

A Spearman correlation of 0.68 means subset RMSE is NOT a reliable proxy for full RMSE. Candidates that look good under subset RMSE may be poor under full RMSE.

### Finding 3: Same Problem as Weighted Loss

```
EXP_WEIGHTED_LOSS_001 (FAILED):
  - Optimized weighted RMSE instead of unweighted
  - Score: 1.0131 vs baseline 1.1688 (-0.1557)
  - Different objective → different optimum

Sensor Subset Selection:
  - Would optimize subset RMSE instead of full RMSE
  - Same problem: different objective → different optimum
  - Expected similar or worse outcome
```

### Finding 4: No Computational Benefit

| Operation | Time |
|-----------|------|
| RMSE computation (4 sensors) | ~0.1 ms |
| RMSE computation (2 sensors) | ~0.05 ms |
| Simulation (bottleneck) | ~1200 ms |

Reducing sensors from 4 to 2 saves ~0.05 ms per evaluation. This is negligible compared to simulation time (0.004% savings).

## Why Sensor Subset Selection Doesn't Help

### The Fundamental Problem

```
We are SCORED on: RMSE using ALL sensors
We would OPTIMIZE on: RMSE using SUBSET of sensors

This is proxy optimization!

Proxy optimization has consistently FAILED:
- Weighted RMSE (EXP_WEIGHTED_LOSS_001): -14% score
- Log RMSE (EXP_LOG_RMSE_LOSS_001): same accuracy, +12 min overhead
- Any loss modification finds DIFFERENT optimum than scoring
```

### Correlation Analysis

With Spearman r = 0.68:
- ~50% of rankings preserved (r² ≈ 0.46)
- CMA-ES may converge to wrong optimum
- Final evaluation on full RMSE will show degradation

### Variance Ratio Variation

| Metric | Value |
|--------|-------|
| Mean variance ratio (sensor 0/1) | 18.61 |
| Std deviation | 57.83 |
| Min | 0.02 |
| Max | 375.58 |

Huge variation in sensor informativeness across samples makes any fixed selection strategy unreliable.

## Abort Criteria Met

From experiment specification:
> "Sensor selection overhead exceeds savings OR subset loses critical source information"

Actual abort reason:
> **Subset RMSE has only r=0.68 correlation with full RMSE (scoring criterion). This is equivalent to proxy optimization, which failed catastrophically in EXP_WEIGHTED_LOSS_001. No computational benefit since RMSE is not the bottleneck.**

## Recommendations

### 1. sensor_optimization Family Should Be Marked EXHAUSTED
Any approach that optimizes a modified objective will find different optimum than the scoring criterion.

### 2. Don't Modify the Objective Function
Lessons learned:
- Weighted RMSE: FAILED (-14%)
- Log RMSE: FAILED (no benefit)
- Subset RMSE: Would FAIL (r=0.68 proxy)
- Tikhonov regularization: ABORTED (same issue)

**Always optimize exactly what you're scored on.**

### 3. RMSE Computation is Not Bottleneck
Simulation (~1200 ms) dominates. Any optimization of RMSE computation (<1 ms) is pointless.

## Conclusion

**ABORTED** - Sensor subset selection is proxy optimization (optimizing subset RMSE when scored on full RMSE). The correlation of 0.68 is insufficient - this would cause CMA-ES to converge to wrong solutions. The sensor_optimization family fails for the same reason as loss_function modifications: optimizing a different objective finds a different (wrong) optimum.

## Files
- `feasibility_analysis.py`: Initial sensor count analysis
- `detailed_analysis.py`: Correlation and timing analysis
- `STATE.json`: Experiment state tracking

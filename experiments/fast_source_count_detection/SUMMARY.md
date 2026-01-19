# Experiment Summary: Fast Source Count Detection

## Metadata
- **Experiment ID**: EXP_FAST_SOURCE_DETECT_001
- **Worker**: W2
- **Date**: 2026-01-19
- **Algorithm Family**: preprocessing

## Objective
Fast detection of 1-source vs 2-source BEFORE optimization to save time by routing samples to appropriate optimizers.

## Hypothesis
Simple peak detection on thermal images can determine n_sources, saving time by not running 2-src optimizer on 1-src samples.

## Results Summary
- **Status**: **ABORTED** - Invalid premise
- **Key Finding**: n_sources is already provided in sample data; detection is unnecessary

## Why This Experiment Was Aborted

### Reason 1: INVALID PREMISE
The baseline optimizer already has direct access to `n_sources` from the sample data:

```python
# From robust_fallback/optimizer.py:359
n_sources = sample['n_sources']
```

The hypothesis assumed we "waste time trying 2-source optimization on 1-source samples." In reality, the baseline ALREADY routes samples correctly based on the provided n_sources value.

### Reason 2: Detection Not Feasible
Even if detection were needed, it's not achievable with the available data:

**Data Limitation**: We don't have thermal images. We only have:
- `sensors_xy`: Sparse sensor positions (2-6 sensors)
- `Y_noisy`: Temperature time series at sensor locations

**Feature Analysis Results**:
| Feature | 1-source | 2-source | Separable? |
|---------|----------|----------|------------|
| var_sensors | 4.7 +/- 8.4 | 5.5 +/- 4.5 | NO |
| temp_range | 3.8 +/- 3.6 | 5.4 +/- 2.7 | NO |
| avg_corr | 0.69 +/- 0.31 | 0.85 +/- 0.17 | NO |
| onset_spread | 101 +/- 129 | 109 +/- 108 | NO |
| entropy | 0.89 +/- 0.42 | 1.17 +/- 0.27 | NO |

**Classification Accuracy**:
- Simple heuristic: 57.5% (barely better than majority baseline)
- Random Forest CV: 67.5% +/- 10.8%
- Success criteria: >= 95%
- **Gap**: 28% below target

## Why Detection Fails
1. **Sparse sensor data**: Only 2-6 sensor readings per sample - not enough to characterize the full thermal field
2. **Similar dynamics**: 1-source and 2-source samples produce overlapping feature distributions
3. **Noise**: 10% noise in temperature readings masks subtle differences
4. **Problem degeneracy**: Different source configurations can produce similar sensor readings

## Test Data Analysis
- Total samples: 80
- 1-source samples: 32 (40%)
- 2-source samples: 48 (60%)
- Sensor count: varies 2-6 per sample (no strong correlation with n_sources)

## Recommendations for Future Work

### DO NOT TRY
1. Any feature-based detection from sensor readings
2. Statistical methods on sparse sensor data
3. Peak detection without full thermal field

### VERIFY BEFORE ANY SIMILAR EXPERIMENT
1. Check if the metadata already provides the needed information
2. Confirm that the problem actually exists before trying to solve it
3. Test with a simple feasibility analysis before full implementation

## Conclusion

**ABORTED**: This experiment's premise was invalid. The n_sources value is already available in the sample data, and the baseline correctly uses it. Even if detection were needed, achieving 95% accuracy is not feasible with sparse sensor readings (achievable accuracy: ~67%).

This experiment took ~15 minutes of analysis time and saved significant implementation effort by identifying the flawed premise early.

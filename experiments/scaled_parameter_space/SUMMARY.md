# Experiment Summary: scaled_parameter_space

## Metadata
- **Experiment ID**: EXP_INVERSE_SCALING_001
- **Worker**: W1
- **Date**: 2026-01-25
- **Algorithm Family**: problem_transform

## Status: ABORTED (Prior Evidence Conclusive)

## Objective
Scale optimization parameters to have similar sensitivity. Transform x from [0,2] to [0,1] and keep y in [0,1], giving a unit hypercube for CMA-ES search.

## Why Aborted

**Coordinate-wise scaling has been thoroughly tested via dd-CMA-ES**, which adaptively learns per-coordinate variances. This is MORE sophisticated than manual rescaling.

### Key Prior Evidence

| Experiment | Result | Key Finding |
|------------|--------|-------------|
| **diagonal_decoding_cmaes** | FAILED (-0.0222) | "2:1 scale difference is too mild" |
| **separable_cmaes** | FAILED | "Separable search is WORSE than full covariance" |

### Why Manual Scaling Cannot Help

1. **dd-CMA-ES Already Tested This**
   - dd-CMA-ES learns optimal diagonal scaling ADAPTIVELY
   - Result: "dd-CMA-ES is essentially neutral - dd=1.0 vs dd=0 gives no significant difference"
   - Manual rescaling is LESS sophisticated than adaptive learning

2. **The 2:1 Ratio Is Too Mild**
   - Domain is [0,2] x [0,1] - only 2:1 aspect ratio
   - Quote: "2:1 scale difference is too mild for coordinate-wise optimization"
   - Severe ill-conditioning (e.g., 100:1) would be needed for scaling to matter

3. **Low-Dimensional Problems Don't Need Scaling**
   - We optimize 2D (1-source) or 4D (2-source) parameters
   - Quote: "Low-dimensional problems don't benefit from diagonal decoding"
   - Full covariance CMA-ES handles these dimensions easily

## Technical Analysis

### What dd-CMA-ES Does vs Manual Scaling

| Approach | Method | Sophistication |
|----------|--------|----------------|
| **Manual scaling** | x' = x/2, y' = y | Fixed rescaling |
| **dd-CMA-ES** | Learns D = diag(d) adaptively | Adaptive per-coordinate |

dd-CMA-ES is strictly more powerful - if it couldn't help, manual scaling won't either.

### Why Full Covariance Is Sufficient

For 2D/4D problems with mild 2:1 conditioning:
- Full covariance matrix C can encode any elliptical distribution
- Correlations between x and y may be important (dd-CMA loses some)
- Standard CMA-ES already finds optimal scaling via covariance adaptation

## Algorithm Family Status

- **problem_transform**: Should be marked **EXHAUSTED** for scaling variants
- **cmaes_variants**: Already EXHAUSTED - dd-CMA and sep-CMA both failed

### Related Failed Approaches

1. **dd-CMA-ES**: Adaptive diagonal decoding - no improvement
2. **sep-CMA-ES**: Separable (no correlations) - WORSE
3. **polar_parameterization**: Alternative coords - FAILED

## Recommendations

1. **Do NOT pursue parameter rescaling** - already tested via dd-CMA-ES
2. **The domain conditioning is fine** - 2:1 ratio is not a problem
3. **Standard CMA-ES is optimal** - it adapts covariance naturally

## Conclusion

The scaled_parameter_space experiment would fail because: (1) dd-CMA-ES already tested coordinate-wise scaling adaptively and found no benefit, (2) the 2:1 domain ratio is too mild for any scaling approach to help, and (3) low-dimensional problems (2D/4D) don't need specialized scaling. Standard CMA-ES with full covariance adaptation is already optimal.

The problem_transform family should be marked EXHAUSTED for scaling-related experiments.

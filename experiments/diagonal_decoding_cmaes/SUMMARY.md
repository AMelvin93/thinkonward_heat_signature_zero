# Experiment Summary: diagonal_decoding_cmaes

## Metadata
- **Experiment ID**: EXP_DD_CMAES_001
- **Worker**: W1
- **Date**: 2026-01-24
- **Algorithm Family**: cmaes_variants

## Objective
Test whether dd-CMA-ES (Diagonal Decoding CMA-ES) improves optimization for the heat source inverse problem. The hypothesis is that the domain's different scales (Lx=2.0, Ly=1.0) could benefit from coordinate-wise rescaling.

## Hypothesis
dd-CMA introduces adaptive diagonal decoding that learns coordinate-wise variances WITHOUT losing correlation information. Unlike sep-CMA-ES (which drops correlations), dd-CMA adds diagonal learning ON TOP of full covariance. This could help with our coordinate-wise ill-conditioned domain where Lx=2.0, Ly=1.0.

Based on: "Diagonal Acceleration for CMA-ES" (Akimoto & Hansen, Evolutionary Computation 2020)

## Results Summary
- **Best In-Budget Score**: 1.1466 @ 50.3 min
- **Best Overall Score**: 1.1466 @ 50.3 min
- **Documented Baseline**: 1.1688 @ 58.4 min
- **Delta**: -0.0222 score, -8.1 min (faster but less accurate)
- **Status**: **FAILED** - dd-CMA-ES does NOT improve accuracy

## Tuning History

| Run | Config Changes | Score | Time (min) | In Budget | Notes |
|-----|---------------|-------|------------|-----------|-------|
| 1 | dd=1.0, sigma 0.18/0.22 | 1.1394 | 51.3 | Yes | Baseline dd-CMA-ES config |
| 2 | dd=1.0, sigma 0.22/0.26 | 1.1466 | 50.3 | Yes | Higher sigma helped slightly |
| 3 | dd=0, sigma 0.18/0.22 | 1.1389 | 49.1 | Yes | Control - no diagonal decoding |

## Key Findings

### What Didn't Work
1. **dd-CMA-ES is essentially neutral**: dd=1.0 (1.1394) vs dd=0 (1.1389) - no significant difference
2. **2:1 scale difference is too mild**: The Lx=2.0, Ly=1.0 domain ratio is only 2:1, not severe enough for diagonal decoding to provide benefit
3. **Low-dimensional problems don't benefit**: 2D (1-source) and 4D (2-source) are already small enough for standard CMA-ES to handle well

### Analysis
- dd-CMA-ES is designed for problems with severe coordinate-wise ill-conditioning
- Our problem has:
  - Positions bounded to [0, 2.0] x [0, 1.0] - only 2:1 ratio
  - Intensity analytically computed - not part of CMA-ES search
  - Full covariance already captures any x-y correlations

### Surprising Finding
- Higher sigma (0.22/0.26 vs 0.18/0.22) with dd=1.0 gave +0.0072 score improvement
- This suggests sigma tuning is more impactful than CMA-ES variant selection

## Parameter Sensitivity
- **Most impactful**: sigma (higher = slightly better accuracy)
- **Least impactful**: CMA_diagonal_decoding (0 vs 1 makes no difference)

## Recommendations for Future Experiments

### Do NOT Try
1. sep-CMA-ES (EXP_SEP_CMAES_001) - if dd-CMA doesn't help, sep-CMA (which drops correlations) will likely hurt
2. Other CMA-ES variants for coordinate scaling - 2:1 ratio is too mild

### What Might Help
1. Focus on the documented baseline's exact configuration to understand why it achieves 1.1688
2. Consider that the baseline may use different NM polish or initialization strategies

## cmaes_variants Family Status
Based on this experiment and prior results:
- dd-CMA-ES: FAILED (no improvement)
- sep-CMA-ES (W2 is testing): Likely to fail (loses correlations)
- Standard CMA-ES: Remains optimal for this problem

**Recommendation**: Mark cmaes_variants family as likely EXHAUSTED after sep-CMA-ES completes.

## Raw Data
- MLflow run IDs:
  - Run 1: f9e55d458cdd4eaba90dd0a30a0c2f81
  - Run 2: 80be1c380d4a450b82cb47fafa9b72f3
  - Run 3: 5041edf6f98043b2b9d7897d6e5ade1a
- Best config: diagonal_decoding=1.0, sigma=0.22/0.26, timestep_fraction=0.40, fevals=20/36, refine_maxiter=8

## Conclusion
**EXPERIMENT FAILED** - dd-CMA-ES provides no improvement for the heat source inverse problem. The 2:1 domain aspect ratio is insufficient to benefit from coordinate-wise diagonal decoding. Standard CMA-ES remains optimal.

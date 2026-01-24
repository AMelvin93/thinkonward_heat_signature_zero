# Coordinate-wise Sigma

## Experiment ID: EXP_COORD_SIGMA_001

## Status: ABORTED (Flawed Premise + Prior Evidence)

## Original Hypothesis
Use different initial sigma for positions (x,y) vs intensity (q). The hypothesis was that positions bounded [0,2]x[0,1] and intensity [0.5,2.0] have different scales that might benefit from coordinate-specific exploration rates.

## Why Aborted

### 1. Flawed Premise: CMA-ES Does Not Optimize Intensity
The experiment description suggests using different sigma for "positions (x,y) vs intensity (q)". However, **CMA-ES only optimizes position parameters**. Intensity (q) is computed analytically via least squares regression after each position evaluation.

From the baseline optimizer:
- 1-source: CMA-ES searches (x, y) → q computed analytically
- 2-source: CMA-ES searches (x1, y1, x2, y2) → q1, q2 computed analytically

The premise of "different sigma for position vs intensity" is invalid.

### 2. Prior Evidence: dd-CMA-ES Failed
**EXP_DD_CMAES_001** tested coordinate-wise scaling via Diagonal Decoding CMA-ES. Results:

| Config | Score | Notes |
|--------|-------|-------|
| dd=1.0 | 1.1394 | With diagonal decoding |
| dd=0 | 1.1389 | Without diagonal decoding |

Key finding: **The 2:1 domain ratio (Lx=2.0, Ly=1.0) is too mild for coordinate-wise approaches to help.**

The dd-CMA-ES paper (Akimoto & Hansen, 2020) shows that diagonal decoding helps when there's severe coordinate-wise ill-conditioning. Our 2:1 ratio is not severe enough.

### 3. Low-Dimensional Problems Don't Benefit
With only 2 parameters (1-source) or 4 parameters (2-source), the search space is small enough for standard CMA-ES to handle without coordinate-specific treatment.

## Recommendation

Do not pursue coordinate-wise sigma approaches. The cmaes_tuning family should focus on other parameters.

Alternative experiments to consider:
- Lower initial sigma (tighter initial search)
- Different fevals/polish budget allocation
- Different threshold tuning

## Prior Experiment Reference
See `experiments/diagonal_decoding_cmaes/SUMMARY.md` for detailed analysis of why coordinate-wise scaling fails for this problem.

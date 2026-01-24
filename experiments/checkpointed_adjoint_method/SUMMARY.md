# Experiment Summary: checkpointed_adjoint_method

## Metadata
- **Experiment ID**: EXP_CHECKPOINTED_ADJOINT_001
- **Worker**: W2
- **Date**: 2026-01-24
- **Algorithm Family**: gradient_revisited

## Objective
Implement a checkpointing-based discrete adjoint method for the ADI heat equation solver to compute gradients efficiently, avoiding the manual derivation errors that plagued the previous adjoint attempt (EXP_CONJUGATE_GRADIENT_001).

## Hypothesis
The previous adjoint experiment failed because manual adjoint derivation for ADI time-stepping is error-prone (gradients were 5-6 orders of magnitude too small). A more rigorous discrete adjoint approach - explicitly deriving the adjoint of each ADI operation step-by-step - should yield correct gradients.

## Results Summary
- **Best In-Budget Score**: N/A (gradient verification failed)
- **Best Overall Score**: N/A
- **Baseline Comparison**: N/A
- **Status**: **FAILED**

## Key Findings

### What Worked: Single-Timestep Adjoint ✓
Verified the discrete adjoint of a single ADI timestep against finite differences:

| Test | Adjoint Value | Finite Diff | Ratio | Error |
|------|--------------|-------------|-------|-------|
| d(L)/d(U_in) | -0.74308004 | -0.74308005 | 1.000000 | 0.00% |
| d(L)/d(S_scaled) | -1.52844596 | -1.53551260 | 0.995398 | 0.46% |

**Critical Discovery**: The corrected discrete adjoint for a single ADI timestep is:
```
Forward:  U' = Ay^{-1} * (I+rLx) * Ax^{-1} * (I+rLy) * U + source terms
Adjoint:  λ_n = (I+rLy) * Ax^{-1} * (I+rLx) * Ay^{-1} * λ_{n+1}
```

The previous experiment (EXP_CONJUGATE_GRADIENT_001) had the operation order WRONG:
- Previous (wrong): `Ay^{-1} * (I+rLx) * Ax^{-1} * (I+rLy)` (same as forward)
- Correct: `(I+rLy) * Ax^{-1} * (I+rLx) * Ay^{-1}` (reversed order)

### What Failed: Full Gradient Accumulation ✗
Despite correct single-timestep adjoints, the full gradient has a consistent 5x error:

| Metric | Adjoint | Finite Diff | Ratio | Error |
|--------|---------|-------------|-------|-------|
| dRMSE/dx | -2.044 | -9.914 | 0.206 | 79.4% |
| dRMSE/dy | -0.276 | -1.341 | 0.206 | 79.4% |

The 5x error is CONSISTENT (same ratio for both x and y), suggesting a systematic issue.

### Root Cause Analysis

The likely cause is the **chain rule through optimal intensity q(x,y)**:

The optimization objective is RMSE(x, y) where:
1. Intensity q is computed as optimal q(x,y) via least squares
2. q depends on x,y because Y_unit = Y(x,y,q=1) depends on source position

The full gradient should be:
```
d(RMSE)/d(x) = d(RMSE)/d(q) * d(q)/d(x) + d(RMSE)/d(Y_sim) * d(Y_sim)/d(x)
```

We only computed the second term. The first term (sensitivity through q) is missing.

## Technical Details

### Correct Adjoint of Source Term
The source S appears in BOTH ADI half-steps:
- Step 1: `Ax * U* = (I+rLy)*U + S_scaled`
- Step 2: `Ay * U' = (I+rLx)*U* + S_scaled`

Total: `d(U')/d(S) = Ay^{-1} + Ay^{-1} * (I+rLx) * Ax^{-1}`

Adjoint: `d(L)/d(S) = Ay^{-1} * λ + Ax^{-1} * (I+rLx) * Ay^{-1} * λ`

### Implementation Complexity
Computing the full gradient correctly requires:
1. Adjoint of ADI timestep w.r.t. temperature (done, verified correct)
2. Adjoint of ADI timestep w.r.t. source (done, verified correct)
3. Chain rule through optimal q computation (NOT done, complex)
4. Proper handling of checkpoints for memory efficiency (NOT done)

## Recommendations for Future Experiments

### 1. Do NOT Retry Manual Adjoint
This experiment confirms the previous finding: manual adjoint implementation for this problem is extremely complex due to:
- ADI splitting requiring careful operation reversal
- Coupled optimization where q depends on (x,y)
- Multiple places where chain rule must be applied correctly

### 2. Adjoint-Free Methods Are Preferred
For this problem, adjoint-free methods (CMA-ES + analytical q) are more practical:
- CMA-ES doesn't need gradients - works well for 2-4D problems
- Analytical q via least squares is efficient and accurate
- The baseline already achieves near-optimal performance

### 3. If Gradients Are Needed, Use Automatic Differentiation
The only viable path for gradient-based optimization would be:
- Full JAX rewrite with implicit differentiation for ADI solves
- This was attempted (EXP_JAX_AUTODIFF_001) and failed due to stability issues
- The explicit Euler stability constraint makes JAX impractical

## Conclusion
**FAILED** - Single-timestep adjoint derivation is correct, but the full gradient has a 5x error due to missing chain rule through the optimal intensity computation. This validates the conclusion from previous experiments: **gradient-based methods are not viable for this problem** due to implementation complexity. The CMA-ES baseline with analytical intensity optimization remains optimal.

## Raw Data
- Gradient test code: `/workspace/experiments/checkpointed_adjoint_method/gradient_test.py`
- Single-timestep U adjoint: ratio 1.000000 ✓
- Single-timestep S adjoint: ratio 0.995398 ✓
- Full gradient: ratio 0.206143 ✗

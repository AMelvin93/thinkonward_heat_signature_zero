# Experiment Summary: conjugate_gradient_adjoint

## Metadata
- **Experiment ID**: EXP_CONJUGATE_GRADIENT_001
- **Worker**: W2
- **Date**: 2026-01-22
- **Algorithm Family**: adjoint_gradient

## Objective
Implement the adjoint method for O(1) gradient computation to enable fast L-BFGS-B optimization, bypassing the O(2n) finite differences overhead that made previous gradient-based optimization too slow.

## Hypothesis
L-BFGS-B achieved 1.1627 accuracy but took 202 min due to finite differences (9 simulations per gradient for 2-source). Adjoint method computes exact gradients in O(1) (forward + adjoint solve), enabling gradient-based optimization within the 60-minute budget.

## Results Summary
- **Best In-Budget Score**: 0.9006 @ 56.5 min
- **Best Overall Score**: 0.9006 @ 56.5 min
- **Baseline Comparison**: -0.2241 vs 1.1247 (**20% WORSE**)
- **Status**: **FAILED**

## Root Cause Analysis

### Critical Finding: Incorrect Gradient Implementation
Verification test comparing adjoint gradients to finite differences:

| Metric | Adjoint | Finite Diff | Ratio |
|--------|---------|-------------|-------|
| dRMSE/dx | -0.000017 | -9.079 | 0.000002 |
| dRMSE/dy | +0.000004 | +1.980 | 0.000002 |

**The adjoint gradient is 5-6 orders of magnitude too small.**

This explains why L-BFGS-B showed "0 iterations" for most optimization runs - the optimizer sees near-zero gradients and assumes immediate convergence.

### Why the Implementation Failed

1. **Complex PDE discretization**: The heat equation uses ADI (Alternating Direction Implicit) time-stepping, which requires careful adjoint derivation. The forward/backward time symmetry is broken by the split scheme.

2. **Sensor interpolation**: Converting point sensor observations to field sources for the adjoint requires careful bilinear interpolation weights that must be consistent with the forward problem.

3. **Scaling issues**: The integration over space and time requires proper discretization weights (dx, dy, dt) that may not be correctly applied.

4. **Missing terms**: The optimal intensity (q) depends on position (x, y), creating coupled gradients that may not be fully captured.

## Tuning History

| Run | Config Changes | Score | Time (min) | In Budget | Notes |
|-----|---------------|-------|------------|-----------|-------|
| 1 | max_iter=10, restarts=2/3 | 0.9006 | 56.5 | Yes | 20% worse than baseline |
| 2 | Gradient verification | N/A | N/A | N/A | Gradient 1e-6 of correct value |

## Key Findings

### What Didn't Work
1. **Manual adjoint derivation** - Error-prone for complex PDEs with ADI time-stepping
2. **Assuming implementation correctness** - Should have verified gradients against finite differences FIRST
3. **L-BFGS-B with bad gradients** - Sees near-zero gradient, immediately "converges" to wrong solution

### What the Experiment Revealed
1. **Adjoint method is theoretically sound** but implementation is extremely difficult for non-trivial PDE discretizations
2. **Gradient verification is essential** before any gradient-based optimization experiment
3. **Manual adjoint derivation requires domain expertise** in numerical methods for PDEs

## Recommendations for Future Experiments

### 1. JAX Autodiff (EXP_JAX_AUTODIFF_001) - HIGH PRIORITY
Instead of manual adjoint derivation, rewrite the heat solver in JAX and use automatic differentiation. This:
- Eliminates manual derivation errors
- Handles complex discretizations automatically
- Provides GPU acceleration as a bonus
- JAX's JIT compilation can be faster than Python loops

### 2. Finite Difference Gradient Refinement
If JAX is too complex, try L-BFGS-B with finite differences but:
- Use only 1-2 iterations (very limited budget)
- Start from CMA-ES best, only polish
- May still be too expensive (9 sims per grad for 2-source)

### 3. Abandon Manual Adjoint
Do NOT retry manual adjoint implementation. The complexity of deriving and debugging is not worth the potential benefit when autodiff alternatives exist.

## Technical Details

### Implementation Attempted
```python
# Forward solve: T(x,y,t) given sources
# Adjoint solve: λ(x,y,t) backward in time with sensor residuals as source
# Gradient: ∂J/∂x0 = ∫∫∫ λ * q * ∂G/∂x0 dt dx dy
```

### Likely Error Sources
1. Adjoint PDE boundary conditions (should match forward but transposed)
2. Time-reversal of ADI steps (x-sweep and y-sweep order should reverse)
3. Source term scaling in adjoint equation
4. Missing chain rule terms from q(x,y) dependence

## Raw Data
- MLflow run IDs: 6e1f20e9be72493eb414d38add47423f
- Best config: {n_restarts_1src: 2, n_restarts_2src: 3, max_iter: 10, timestep_fraction: 0.4}

## Conclusion
**FAILED** - The adjoint gradient implementation is fundamentally incorrect, producing gradients 5-6 orders of magnitude too small. Manual adjoint derivation for complex PDE discretizations is error-prone and not recommended. Future gradient-based experiments should use automatic differentiation (JAX) instead.

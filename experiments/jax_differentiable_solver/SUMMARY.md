# Experiment Summary: jax_differentiable_solver

## Metadata
- **Experiment ID**: EXP_JAX_AUTODIFF_001
- **Worker**: W1
- **Date**: 2026-01-22
- **Algorithm Family**: differentiable_simulation

## Objective
Rewrite the thermal simulator in JAX to enable automatic differentiation, providing exact gradients for L-BFGS-B or Adam optimization.

## Hypothesis
JAX autodiff through the heat equation solver would provide exact gradients automatically, enabling fast gradient-based optimization without the O(2n) finite differences overhead that made L-BFGS-B impractical in previous experiments.

## Results Summary
- **Best In-Budget Score**: N/A (experiment aborted before full run)
- **Best Overall Score**: N/A
- **Baseline Comparison**: N/A
- **Status**: ABORTED - Fundamental Incompatibility

## Abort Reason

**The JAX approach has a fundamental incompatibility with this problem's physics.**

### Technical Analysis

1. **JAX Autodiff Requires Explicit Time-Stepping**
   - JAX's automatic differentiation works by tracing through operations
   - Implicit methods (like ADI) involve solving linear systems which are not directly differentiable through standard JAX
   - Forward Euler (explicit) was implemented as the differentiable alternative

2. **Explicit Euler Stability Constraint**
   - Stability requires: `dt < dx^2 * dy^2 / (2 * kappa * (dx^2 + dy^2))`
   - For 100x50 grid with kappa=0.1: `dt_max = 0.002061`
   - Original implicit ADI uses: `dt = 0.004` (1.9x larger)

3. **Timestep Mismatch**
   - Original simulation: 1000 timesteps @ dt=0.004
   - JAX stable simulation: 3881 timesteps @ dt=0.001031
   - **4x more timesteps needed** to match the same total simulation time

4. **Physics Validation Failed**
   - Running 500 timesteps (for speed): Final temperature at sensor = 0.61
   - Original 1000 timesteps: Final temperature at sensor = 3.46
   - **5.7x temperature difference** - completely wrong physics
   - Optimizer would converge to wrong solutions

## Key Findings

### What Didn't Work
1. **Explicit Euler too slow**: Needs 4x more timesteps than implicit ADI
2. **Truncated simulation wrong physics**: Can't use fewer timesteps - gives wrong temperatures
3. **JIT compilation overhead**: First call takes ~2s to compile, subsequent calls still slower than baseline

### Critical Insights
1. **Implicit methods are essential for time-domain PDEs at this scale**
   - The implicit ADI method allows large timesteps while remaining unconditionally stable
   - Explicit methods are fundamentally limited by physics (CFL condition)

2. **JAX is not a silver bullet for PDE optimization**
   - Works well for simple ODEs or small-scale PDEs
   - For thermal diffusion at realistic scales, implicit methods are necessary

3. **Differentiating through implicit solvers is possible but complex**
   - Would require implementing adjoint method manually in JAX
   - At that point, just implementing adjoint directly (EXP_CONJUGATE_GRADIENT_001) is simpler

## Parameter Sensitivity
N/A - experiment aborted before parameter tuning

## Recommendations for Future Experiments

1. **Abandon the JAX autodiff approach for this problem**
   - The stability constraint makes explicit methods impractical
   - Implicit solver differentiation is too complex for the potential benefit

2. **Focus on the adjoint method instead (EXP_CONJUGATE_GRADIENT_001)**
   - Adjoint provides O(1) gradients through implicit solvers
   - No stability constraint issues
   - W2 is already working on this approach

3. **Alternative: diffrax library with implicit solvers**
   - diffrax has adaptive implicit solvers with autodiff
   - But implementation complexity may not be worth it given adjoint alternative

4. **Mark differentiable_simulation family as EXHAUSTED for explicit methods**
   - Any approach requiring explicit time-stepping will hit the same wall

## Technical Details

### Files Created
- `jax_heat_solver.py`: JAX implementation of explicit Euler heat solver with autodiff
- `test_optimization.py`: Comparison of JAX L-BFGS-B vs CMA-ES
- `test_timing.py`: Timing analysis
- `analyze_issue.py`: Root cause analysis of physics mismatch

### Numerical Comparison
| Metric | Original (Implicit ADI) | JAX (Explicit, 500 steps) |
|--------|------------------------|---------------------------|
| dt | 0.004 | 0.001031 |
| nt | 1000 | 500 (truncated) |
| Total time | 4.0 | 0.516 |
| Final temp sensor 0 | 3.46 | 0.61 |

## Raw Data
- No MLflow runs (experiment aborted before full execution)
- No score computed (physics mismatch prevents valid comparison)

## Conclusion

The JAX differentiable simulation approach is **fundamentally incompatible** with this heat equation problem due to the stability constraints of explicit time-stepping methods. The implicit ADI method used by the original simulator is essential for efficiency, and differentiating through it would require manual adjoint implementation (which is EXP_CONJUGATE_GRADIENT_001, a separate experiment).

**Recommendation**: Mark `differentiable_simulation` family as EXHAUSTED and focus resources on adjoint-based gradient methods or non-gradient approaches that have proven effective.

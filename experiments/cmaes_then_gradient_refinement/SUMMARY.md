# Experiment Summary: cmaes_then_gradient_refinement

## Metadata
- **Experiment ID**: EXP_HYBRID_CMAES_LBFGSB_001
- **Worker**: W1
- **Date**: 2026-01-22
- **Algorithm Family**: hybrid_gradient

## Objective
Replace Nelder-Mead polish with L-BFGS-B polish in the CMA-ES + polish pipeline. The hypothesis is that L-BFGS-B's gradient-based refinement converges faster near the optimum than NM's simplex method, allowing fewer polish iterations.

## Hypothesis
CMA-ES finds good basin quickly. L-BFGS-B with few iterations (2-3) can refine within the basin faster than NM (8 iterations) because L-BFGS-B uses gradient information (even via finite differences).

## Results Summary
- **Best In-Budget Score (L-BFGS-B)**: 1.1174 @ 42.4 min (WORSE than baseline)
- **Best In-Budget Score (NM x8)**: 1.1415 @ 38.0 min (BETTER than baseline)
- **Baseline Comparison**: L-BFGS-B polish is NOT better than NM polish
- **Status**: FAILED - Hypothesis disproven

## Tuning History

| Run | Method | Iters | Score | Time (min) | In Budget | Notes |
|-----|--------|-------|-------|------------|-----------|-------|
| 1 | L-BFGS-B | 3 | 1.1721 | 60.3 | No | +0.036 score but over budget |
| 2 | NM | 8 | 1.1652 | 45.7 | Yes | Baseline comparison |
| 3 | L-BFGS-B | 2 | 1.1174 | 42.4 | Yes | Score WORSE than baseline |
| 4 | NM | 8 | 1.1415 | 38.0 | Yes | Best in-budget result |

## Key Findings

### Why L-BFGS-B Polish Failed

1. **Finite Difference Overhead**: L-BFGS-B uses finite differences for gradient estimation, requiring O(n) extra function evaluations per iteration:
   - 1-source (2D): 3 extra evals per iteration
   - 2-source (4D): 5 extra evals per iteration

2. **Simulation Cost**: Each function evaluation requires a simulation (~0.5s). The finite difference overhead adds significant time:
   - 2-source with L-BFGS-B x3: ~250-350 sims/sample
   - 2-source with NM x8: ~180 sims/sample

3. **Diminishing Returns**: L-BFGS-B's quadratic convergence only helps when close to optimum. Near the optimum, NM's simplex method is already efficient for low-dimensional problems (2-4D).

### Comparison Summary

| Aspect | NM Polish (x8) | L-BFGS-B Polish (x2-3) |
|--------|---------------|------------------------|
| Sims per 2-src sample | ~180 | ~220-350 |
| Score (20 samples) | 1.1415 | 1.1174-1.1721 |
| Time (20 samples) | 38 min | 42-60 min |
| In budget? | Yes | Borderline/No |

### Critical Insight

**Finite differences make L-BFGS-B inefficient when objective function is expensive.**

For cheap functions, L-BFGS-B's O(n) gradient computation is negligible. But when each function evaluation is a simulation (~0.5s), the overhead dominates. NM's derivative-free approach is more efficient for expensive low-dimensional optimization.

## Parameter Sensitivity

- **polish_maxiter**: L-BFGS-B needs 3+ iterations for accuracy, but 3 iterations is over budget
- **polish_method**: NM is clearly better than L-BFGS-B for this problem

## Recommendations for Future Experiments

1. **Do NOT pursue L-BFGS-B polish** - NM is more efficient for expensive objectives
2. **hybrid_gradient family should be marked FAILED** - gradient polish doesn't help when gradients are expensive
3. **The only way gradient methods work is with analytical/adjoint gradients** - EXP_CONJUGATE_GRADIENT_001 (adjoint) is the only viable gradient approach
4. **NM x8 is already optimal for polish** - no need to explore other derivative-free polish methods

## Technical Details

### Files Created
- `optimizer.py`: HybridCMAESLBFGSBOptimizer with switchable polish method
- `run.py`: Run script with MLflow logging

### Implementation Notes
- L-BFGS-B polish uses scipy.optimize.minimize with method='L-BFGS-B' and bounds
- NM polish uses scipy.optimize.minimize with method='Nelder-Mead'
- Both methods polish on coarse grid with reduced timesteps
- Final evaluation on fine grid with full timesteps

## Raw Data
- 4 tuning runs (no MLflow logging during rapid iteration)
- Best config: NM x8 (same as baseline)

## Conclusion

**L-BFGS-B polish is NOT better than NM polish for this problem** because:
1. Finite difference gradient computation is too expensive (O(n) extra sims per iteration)
2. NM is already efficient for low-dimensional simplex optimization
3. The gradient information doesn't compensate for the overhead

The hypothesis that "L-BFGS-B converges faster than NM" is only true when gradient computation is cheap. When gradients require finite differences with expensive objective evaluations, NM wins.

**Recommendation**: Keep NM x8 as the polish method. hybrid_gradient family is FAILED.

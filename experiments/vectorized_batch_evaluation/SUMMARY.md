# Experiment Summary: vectorized_batch_evaluation

## Metadata
- **Experiment ID**: EXP_VECTORIZED_BATCH_EVAL_001
- **Worker**: W1
- **Date**: 2026-01-25
- **Algorithm Family**: engineering

## Status: ABORTED (Technically Infeasible)

## Objective
Vectorize CMA-ES population evaluation by evaluating all candidates simultaneously in a batched call, exploiting NumPy SIMD/vectorization efficiency.

## Why Aborted

This experiment is **technically infeasible** due to the structure of the ADI (Alternating Direction Implicit) solver.

### The ADI Solver Structure

From `simulator.py` analysis:
```python
# LU decomposition (can be shared)
Ax_lu = splu(Ax)  # line 153
Ay_lu = splu(Ay)  # line 154

# Time-stepping loop (per timestep)
for n in range(nt):
    S_field = self._source_field(t_half, sources)  # ← DIFFERENT per candidate!

    # ADI Step 1: Implicit in X
    for j in range(self.ny):  # 50 columns
        U_star[:, j] = Ax_lu.solve(RHS_field_x[:, j])  # ← Can't batch

    # ADI Step 2: Implicit in Y
    for i in range(self.nx):  # 100 rows
        U_next[i, :] = Ay_lu.solve(RHS_field_y[i, :])  # ← Can't batch
```

### Key Technical Blockers

| Issue | Description |
|-------|-------------|
| **Source Fields Differ** | Each CMA-ES candidate has different (x, y, Q) → different S_field |
| **splu.solve() Not Batched** | scipy.sparse.linalg.splu.solve() takes single vector, not matrix |
| **Implicit Method Essential** | ADI uses dt=0.004; explicit methods need dt=0.001 (4x more timesteps) |
| **Current Parallelization** | Already using all CPUs across samples; within-sample would conflict |

### JAX Alternative Failed

From `jax_differentiable_solver` (EXP_JAX_AUTODIFF_001):
- JAX autodiff requires explicit time-stepping
- Explicit Euler stability: `dt_max = 0.002` (vs ADI's dt=0.004)
- Would need **4x more timesteps** → ~4x slower per simulation
- Quote: "Implicit methods are essential for time-domain PDEs at this scale"

### What Would Be Required

To truly vectorize batch evaluation:

1. **scipy Modification** (Impossible)
   - scipy.sparse.linalg.splu doesn't support batched RHS
   - Would need core library changes

2. **GPU Library** (cupy/JAX)
   - cupy has batched sparse solvers
   - But GPU memory overhead + transfer time may negate gains
   - JAX with implicit solvers is complex (manual adjoint needed)

3. **Explicit Methods** (4x Slower)
   - Could vectorize easily (matrix operations)
   - But 4x more timesteps due to stability constraint
   - Net result: slower, not faster

4. **Manual Batching** (No Gain)
   - Unroll batch into loop of individual solves
   - Same as current approach

## Current Parallelization Strategy

The baseline already maximizes parallelism:
```
joblib.Parallel(n_jobs=cpu_count):
    for sample in test_data:  # 80 samples, parallelized
        for init in [triangulation, hotspot]:  # sequential
            cmaes.optimize()  # 20-36 evals, sequential
```

- **Cross-sample**: Parallelized via joblib
- **Cross-init**: Sequential (only 2 inits)
- **Cross-eval**: Sequential (CMA-ES requires sequential fitness)

Adding within-sample parallelization would:
1. Conflict with cross-sample parallelization (CPU contention)
2. Add threading overhead for minimal gain

## Recommendations

1. **Do NOT pursue simulator vectorization** - technically infeasible with ADI
2. **engineering family is limited** - current approach is near-optimal
3. **Focus on algorithmic improvements** - not engineering optimizations
4. **GPU acceleration is out of scope** - would require major architecture change

## Raw Data
- MLflow run IDs: None (experiment not executed)
- Prior evidence: jax_differentiable_solver, simulator.py analysis

## Conclusion

The vectorized batch evaluation is **technically infeasible** due to the ADI solver's reliance on scipy's sparse LU decomposition, which doesn't support batched solving. Each CMA-ES candidate has different source configurations, requiring different forcing terms that cannot be vectorized. The JAX alternative failed because explicit methods need 4x more timesteps. The current cross-sample parallelization already uses all CPUs optimally.

The engineering optimization family has limited potential for this problem. Focus should remain on algorithmic improvements.

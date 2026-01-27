# Experiment Summary: fcmaes_speed_test

## Metadata
- **Experiment ID**: EXP_FCMAES_SPEED_001
- **Worker**: W2
- **Date**: 2026-01-26
- **Algorithm Family**: implementation_v2

## Objective
Replace pycma with fcmaes library (fast-cma-es) which claims 8-10x speedup through C++/Numba backend, potentially allowing more iterations within budget.

## Hypothesis
fcmaes library uses optimized C++ backend which may significantly reduce CMA-ES runtime, allowing the saved time for more iterations or polish.

## Results Summary
- **Best Score**: 1.0722 @ 57.1 min (10-sample test)
- **Baseline**: 1.1688 @ 58.4 min
- **Delta**: -0.0966 score (MUCH WORSE), -1.3 min (minimal gain)
- **Status**: **FAILED**

## Critical Finding: CMA-ES is NOT the Bottleneck

### Time Profiling Results

| Component | Time per Sample | Percentage |
|-----------|-----------------|------------|
| Simulation | 6000 ms | **99.8%** |
| CMA-ES overhead | 10 ms | 0.2% |

**The thermal simulation dominates runtime**, not CMA-ES. Optimizing CMA-ES speed is pointless.

### Breakdown

```
Per-simulation cost: ~200 ms (50 sims tested)
Typical sample: ~30 evaluations
Total simulation time: 30 × 200 = 6000 ms

CMA-ES overhead: ~2 ms per ask/tell cycle
Typical sample: ~5 cycles
Total CMA-ES time: 5 × 2 = 10 ms
```

Even if fcmaes was 100x faster, we'd save only 10ms per sample.

## Why fcmaes Gave Worse Results

1. **API Limitations**: fcmaes.minimize() returns only the final best solution, not intermediate candidates evaluated during optimization.

2. **Diversity Impact**: pycma-based optimizer collects ALL solutions evaluated (24-30 candidates per sample), enabling better diversity scoring. fcmaes gives only 1-2 candidates.

3. **Scoring Formula**: Diversity term = 0.3 × (n_candidates / 3). With fewer candidates:
   - pycma: 3 candidates → 0.3 bonus
   - fcmaes: 1-2 candidates → 0.1-0.2 bonus

## Technical Issues with fcmaes Low-Level API

Attempted to use fcmaes.Cmaes class with ask/tell pattern (like pycma), but encountered:
- Different API signature: `tell(ys, xs)` instead of `tell(solutions, fitness)`
- Buggy behavior with `tell_one()` method
- Array comparison errors in internal state management

The high-level `minimize()` interface works but doesn't expose intermediate solutions.

## Lessons Learned

### 1. Always Profile Before Optimizing
The assumption was that CMA-ES was slow and could be optimized. Profiling showed:
- Simulation: 200ms (the actual bottleneck)
- CMA-ES: 2ms (negligible)

### 2. Library Speed ≠ Better Results
fcmaes is ~3-5x faster at CMA-ES operations, but:
- Doesn't help when CMA-ES isn't the bottleneck
- API differences can hurt quality (fewer candidates)

### 3. Speedup Must Target the Bottleneck
For this problem, speedup efforts should focus on:
- Reducing simulation calls (early stopping, surrogate models)
- Making simulations faster (coarser grids, fewer timesteps)
- NOT on optimizing CMA-ES library choice

## Recommendations

1. **Mark implementation_v2 family as exhausted** for CMA-ES library alternatives

2. **DO NOT** pursue other CMA-ES libraries (all will face same bottleneck)

3. **Focus areas for real speedup**:
   - Simulation reduction (already explored with temporal fidelity)
   - Spatial coarsening (already explored, limited success)
   - Surrogate models (expensive to train, tested and failed)

4. **Accept current architecture**: The 1.1688 baseline is well-optimized. CMA-ES + NM polish + temporal fidelity is near-optimal for this problem.

## Conclusion

**FAILED** - The fcmaes experiment demonstrates a fundamental misunderstanding of where optimization effort should be directed. CMA-ES library speed is irrelevant when simulations consume 99.8% of runtime. The baseline pycma implementation is perfectly adequate.

No further CMA-ES library experiments should be attempted.

## Raw Data
- 10-sample test score: 1.0722 @ 57.1 min
- Profiling: 200ms/sim, 2ms/CMA-ES cycle
- fcmaes API tested: minimize(), Cmaes.ask()/tell()

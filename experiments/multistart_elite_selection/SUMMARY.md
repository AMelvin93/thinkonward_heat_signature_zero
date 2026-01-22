# Experiment: Multi-Start Elite Selection

**Experiment ID:** EXP_MULTISTART_ELITE_001
**Worker:** W2
**Status:** FAILED
**Family:** multi_start

## Hypothesis

Instead of running N full CMA-ES optimizations from different initializations, run N short evaluations and continue only the most promising 1-2 runs. This should allocate budget more efficiently by avoiding wasted computation on runs that start in poor regions.

## Approach

1. Generate diverse initializations (triangulation, smart, centroid, random)
2. Start N CMA-ES runs in parallel
3. After K generations, evaluate each run's best solution
4. Keep only top M runs (elite selection)
5. Continue elite runs to full convergence
6. Apply standard refinement and NM polish to best solution

## Results

| Run | Config | Score | Time (min) | In Budget | Notes |
|-----|--------|-------|------------|-----------|-------|
| 1 | 4 runs, keep 2, eval@3 gens | 1.1422 | 84.4 | NO | MUCH worse - overhead dominates |
| 2 | 2 runs, keep 1, eval@3 gens | 1.1529 | 67.0 | NO | Still worse - minimal multi-start fails |

**Baseline:** 1.1688 @ 58.4 min

## Analysis

### Why Multi-Start Elite Selection Failed

1. **Multi-start overhead is significant**: Even with just 2 parallel CMA-ES runs, the setup/evaluation overhead adds time without proportional benefit.

2. **Early generation fitness is not predictive**: After only 3 generations, CMA-ES hasn't converged enough to reliably identify which run will produce the best final solution. The "elite selection" criteria is essentially random.

3. **Single-start with good initialization is optimal**: The baseline uses smart initialization techniques (triangulation, thermal analysis) that already put the optimizer in a good starting region. Running multiple parallel searches from different starts doesn't improve on this.

4. **Budget spent on parallel evaluation is wasted**: Every simulation used to evaluate parallel runs early is a simulation not spent on deeper optimization of a single promising run.

### Detailed Results

**Run 1 (4 runs, keep 2):**
- 1-source RMSE: 0.1050 (similar to baseline)
- 2-source RMSE: 0.1469 (similar to baseline)
- Total simulations per sample: 111 (1-src), 280 (2-src)
- Time: 84.4 min (+26 min vs baseline)

**Run 2 (2 runs, keep 1):**
- 1-source RMSE: 0.1073 (worse than baseline)
- 2-source RMSE: 0.1587 (worse than baseline)
- Total simulations per sample: 76 (1-src), 184 (2-src)
- Time: 67.0 min (+8.6 min vs baseline)

## Conclusion

**FAILED** - Multi-start elite selection does NOT improve over single-start baseline.

The multi-start approach fundamentally doesn't work for this inverse heat problem because:
1. The problem landscape is complex enough that early-generation fitness doesn't predict final solution quality
2. The baseline single-start approach with intelligent initialization already finds good solutions efficiently
3. The overhead of running/evaluating parallel CMA-ES runs outweighs any early-selection benefit

## Recommendation

**ABANDON** the multi-start elite selection family of experiments.

Focus instead on:
- Improving single-start optimization quality (better initialization, parameter tuning)
- Reducing simulation cost per evaluation (temporal fidelity improvements)
- Smarter refinement strategies (adaptive NM, coordinate descent)

The single-start + smart initialization paradigm is the right approach for this problem.

## MLflow Run IDs
- Run 1: `8760276b84e447278c32a0437f08a57c`
- Run 2: `d42755b437a641d28d357c4f6efb8b91`

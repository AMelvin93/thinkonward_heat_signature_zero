# Two-Phase CMA-ES Restart Experiment - FAILED

## Experiment ID
EXP_CMAES_RESTART_BEST_001

## Hypothesis
A two-phase CMA-ES approach (Phase 1: broad exploration with large sigma, Phase 2: local refinement with small sigma starting from best solution) could improve accuracy by combining global search with local optimization.

## Configuration Tested
```
Phase 1 (Exploration):
  - fevals: 12/22 (1-src/2-src) - 60% of total budget
  - sigma: 0.25/0.30

Phase 2 (Refinement):
  - fevals: 8/14 (1-src/2-src) - 40% of total budget
  - sigma: 0.05/0.08 (restart from Phase 1 best)

Shared:
  - timestep_fraction: 0.40
  - NM polish: 8 iterations on top-3
  - candidate_pool_size: 10
```

## Results

| Metric | This Run | Baseline (W2) | Delta |
|--------|----------|---------------|-------|
| Score | **1.0986** | 1.1688 | **-0.0702** |
| Time (400 proj) | **175.7 min** | 58.4 min | **+117.3 min** |
| RMSE Mean | 0.1403 | ~0.12 | +0.02 |
| 1-src RMSE | 0.1034 | - | - |
| 2-src RMSE | 0.1650 | - | - |

**STATUS: FAILED** - Both score and timing are significantly worse than baseline.

## Root Cause Analysis

### Why This Failed

1. **Doubled Simulation Count**: The two-phase approach effectively doubles the number of CMA-ES evaluations per candidate initialization:
   - Phase 1: ~12-22 fevals per init
   - Phase 2: ~8-14 fevals per init (on top of Phase 1)
   - Result: ~100 sims for 1-src became 100+ sims, ~230 sims for 2-src

2. **Diversity Loss from Small Sigma**: Phase 2 with sigma=0.05-0.08 converges all candidate pool entries toward the same solution:
   - Many 1-src samples: only 1-2 candidates after diversity filtering (vs 3 expected)
   - Lost ~0.1 on diversity component of score

3. **No Accuracy Improvement**: Phase 2 refinement didn't improve RMSE compared to single-phase:
   - The CMA-ES in Phase 1 already converges to local optima
   - Phase 2 just wastes budget re-converging to same solutions

4. **Fallback Explosion**: Two-phase approach increases fallback frequency:
   - Sample 46: 440 sims, 520s (fallback triggered)
   - Sample 69: 462 sims, 573s (fallback triggered)
   - Fallbacks now take 2x longer due to two phases

### Mathematical Insight

For restart to help, Phase 2 would need to:
1. Find better optima than Phase 1 (not happening - same landscape)
2. Do so with fewer evaluations than Phase 1 (not happening - adds to total)

The fundamental issue: **CMA-ES already converges well in a single phase**. The problem isn't that we're not refining enough - it's that the fitness landscape has multiple local optima and CMA-ES finds one of them regardless of sigma.

## Candidate Diversity Analysis

From the run output:
- 1-source samples: Average ~1.7 candidates (many got only 1)
- 2-source samples: Average ~2.4 candidates (better but still low)

The small sigma in Phase 2 causes all initializations to converge to the same solution, destroying diversity.

## Key Learnings

1. **Restart strategies don't help when single-phase already works**: The baseline CMA-ES converges well enough that restarting from the best solution just wastes budget.

2. **Diversity is critical for scoring**: The scoring formula heavily weights candidate diversity (0.3 * N_valid/3). Approaches that reduce diversity hurt score significantly.

3. **Sigma controls exploration vs exploitation**: Very small sigma eliminates exploration, causing all candidates to collapse to one solution.

4. **Don't add phases - optimize the single phase**: The baseline approach is already well-tuned. Adding complexity doesn't help.

## Recommendation

**ABANDON** - Do not pursue further tuning of two-phase restart approaches.

The conclusion is clear: restart/refinement strategies don't improve upon single-phase CMA-ES for this problem. Future efforts should focus on:
- Better initialization strategies (not restart)
- Smarter sigma adaptation within a single phase
- Alternative optimization algorithms entirely (differential evolution, etc.)

## MLflow Run
- Run ID: `668b9e23840c4c5fb9fa11cb60b2a24e`
- Experiment: heat-signature-zero

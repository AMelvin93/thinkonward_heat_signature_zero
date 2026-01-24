# Experiment Summary: adaptive_population_size

## Metadata
- **Experiment ID**: EXP_ADAPTIVE_POPSIZE_001
- **Worker**: W1
- **Date**: 2026-01-24
- **Algorithm Family**: cmaes_enhancement

## Objective
Test whether starting CMA-ES with a larger population size (12) for exploration then reducing to default (6) for exploitation improves optimization quality. This is the inverse of IPOP which increases population size.

## Hypothesis
Different from IPOP (which increases popsize). Start large (popsize=12) for first 60% of feval budget to explore broadly, then reduce to default (~6) for fine-tuning in the remaining 40%. This should provide better initial exploration before focusing on local convergence.

## Results Summary
- **Best In-Budget Score**: N/A (no runs within budget)
- **Best Overall Score**: 1.0026 @ 124.6 min
- **Baseline Comparison**: -0.1662 vs 1.1688 baseline
- **Status**: **FAILED** - Massively over budget with much worse accuracy

## Tuning History

| Run | Config Changes | Score | Time (min) | In Budget | Notes |
|-----|---------------|-------|------------|-----------|-------|
| 1 | P1=12, P2=6, 60% phase1 | 1.0026 | 124.6 | **No** | 2.1x over budget, -0.17 score. Two-phase creates 8 CMA-ES runs per 2-source |

## Key Findings

### What Didn't Work
1. **Two-phase CMA-ES multiplies overhead**: Each initialization runs 2 sequential CMA-ES phases. With 4 inits for 2-source samples, that's 8 CMA-ES runs vs baseline's 1-2.

2. **Implementation bug**: Each init receives FULL feval budget (20/36) instead of dividing among inits. This alone doubles simulation count.

3. **CMA-ES covariance adaptation breaks between phases**: CMA-ES needs continuous updates to learn parameter correlations. Starting a new CMAEvolutionStrategy instance in phase 2 discards all learning from phase 1.

4. **Large popsize in phase 1 = fewer generations**: With popsize=12 and 60% of 36 fevals (~22), phase 1 only gets ~2 generations. Not enough to build good covariance.

5. **Phase 2 starts with suboptimal conditions**: Reduced sigma from phase 1 but poor covariance means phase 2 can't recover.

### Simulation Count Analysis
| Sample Type | This Experiment | Baseline | Ratio |
|-------------|-----------------|----------|-------|
| 1-source | ~53 sims | ~51 | 1.04x |
| 2-source | ~350 sims | ~190 | 1.84x |
| 2-source (fallback) | ~816 sims | ~380 | 2.15x |

### Critical Insights
1. **CMA-ES is designed for continuous operation**: Breaking into phases loses the accumulated covariance matrix adaptation. This is a fundamental limitation.

2. **Population size manipulation doesn't help**: Three experiments have now failed:
   - IPOP (increasing popsize): 75.7 min, score 1.1687
   - Larger fixed popsize (12): 73.0 min, score 1.1666
   - Adaptive popsize (this): 124.6 min, score 1.0026

3. **Default popsize is optimal**: CMA-ES's default popsize formula `4 + floor(3*ln(n))` is well-tuned for expensive black-box optimization. For 2D (n=2): popsize=7, for 4D (n=4): popsize=8.

## Parameter Sensitivity
- **Most impactful**: Number of CMA-ES instances per sample (phases * inits)
- **Time-sensitive**: Total fevals across all phases and inits

## Family Status
**cmaes_enhancement family should be marked EXHAUSTED**:
- Larger popsize: FAILED (no improvement, +15 min)
- IPOP (increasing): FAILED (restarts don't help)
- Adaptive (decreasing): FAILED (phases break covariance adaptation)

## Recommendations for Future Experiments

### Do NOT Try
1. Any multi-phase CMA-ES approaches - phases break covariance learning
2. Any popsize manipulation - default is already optimal
3. Any restart strategies - problem has no local optima to escape

### What Might Help
1. Focus on REDUCING simulations per sample, not changing CMA-ES internals
2. Better initialization (though this was already tried)
3. Smarter candidate filtering (40% timesteps is current best)
4. The baseline with 40% temporal fidelity (W2: 1.1688 @ 58.4 min) is likely near-optimal

## Raw Data
- MLflow run IDs: 70f8ae3c861e4699a38394355f90d940
- Best config: P1=12, P2=6, phase1_fraction=0.60, timestep_fraction=0.40, sigma=0.18/0.22, fevals=20/36

## Conclusion
**EXPERIMENT FAILED** - Two-phase adaptive population size is fundamentally incompatible with CMA-ES's covariance matrix adaptation mechanism. The baseline's single-phase CMA-ES with default popsize is optimal. No further tuning of this approach is warranted.

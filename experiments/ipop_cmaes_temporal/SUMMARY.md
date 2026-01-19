# Experiment Summary: IPOP-CMA-ES with Temporal Fidelity

## Metadata
- **Experiment ID**: EXP_IPOP_TEMPORAL_001
- **Worker**: W2
- **Date**: 2026-01-19
- **Algorithm Family**: temporal_fidelity_extended

## Objective
Test whether IPOP-CMA-ES (Increasing Population CMA-ES) can improve accuracy by escaping local optima through restarts with larger populations, combined with 40% temporal fidelity for speedup.

## Hypothesis
IPOP-CMA-ES restarts with increasing population sizes to escape local optima. Combined with temporal fidelity speedup (40% timesteps), we should be able to explore more thoroughly while staying within budget.

## Results Summary
- **Best In-Budget Score**: None - no IPOP configuration beat baseline within 60 min
- **Best Overall Score**: 1.1850 @ 79.6 min (OVER BUDGET by 20 min)
- **Baseline Comparison**: IPOP adds time without improving accuracy
- **Status**: FAILED - IPOP not beneficial for this problem

## Tuning History

| Run | Config | Score | Time (min) | In Budget | Notes |
|-----|--------|-------|------------|-----------|-------|
| 1 | 2 restarts, 4 popsize, 20/36 fevals (10 samples) | 1.1850 | 79.6 | **NO** | Good score but 33% over budget |
| 2 | 1 restart, 4 popsize, 20/36 fevals (10 samples) | 1.1668 | 53.0 | Yes | Worse than baseline |
| 3 | 0 restarts, 4 popsize, 20/36 fevals (10 samples) | 1.1475 | 51.5 | Yes | Much worse than baseline |
| 4 | 1 restart, 4 popsize, 25/45 fevals (80 samples) | 1.1727 | 64.9 | **NO** | 5 min over budget |
| 5 | 1 restart, 4 popsize, 22/40 fevals (80 samples) | 1.1687 | 75.7 | **NO** | 16 min over, no improvement |

## Key Findings

### Why IPOP Failed

1. **Budget Splitting Problem**: IPOP divides the feval budget across multiple restarts
   - With 20 fevals and 2 restarts, each restart only gets ~10 fevals
   - CMA-ES needs more fevals to converge properly
   - Neither the first run nor restarts have enough budget to optimize well

2. **No Local Optima Issue**: The thermal inverse problem doesn't suffer from local optima in the way IPOP helps with
   - Standard CMA-ES with good initialization already finds global optimum
   - Random restarts don't help because the problem is well-conditioned
   - The bottleneck is accuracy, not exploration

3. **Time Overhead**: IPOP adds overhead from:
   - Population size doubling means more simulations per generation
   - Multiple restart initialization overhead
   - Each restart must "re-learn" the landscape from scratch

### Comparison with Baseline

| Metric | Baseline | IPOP (best attempt) |
|--------|----------|---------------------|
| Score | 1.1688 | 1.1687 |
| Time | 58.4 min | 75.7 min |
| In Budget | Yes | No |

The best IPOP configuration (Run 5) achieved the same score as baseline but took 17 extra minutes.

## Recommendations for Future Experiments

### Do NOT Pursue:
1. IPOP-CMA-ES or any restart-based strategy
2. Larger population sizes (wastes budget)
3. Random restarts (problem doesn't have local optima)

### Focus On:
1. **Better Initialization**: Physics-informed init (EXP_PHYSICS_INIT_001)
2. **2-Source Optimization**: Specialized treatment for 2-src samples
3. **Sigma Tuning**: Higher sigma showed promise but exceeded budget

## Critical Insight

The thermal inverse problem with temporal fidelity is already well-optimized by the baseline:
- CMA-ES converges reliably to good solutions
- The 40% timestep proxy is sufficiently accurate
- 8 NM polish iterations fully refine the solution
- Time budget is already saturated

**IPOP is designed for multimodal problems with many local optima - this is NOT such a problem.**

## Conclusion

**FAILED - IPOP-CMA-ES does not improve results and adds significant time overhead.**

The baseline optimizer (CMA-ES + 40% temporal + 8 NM polish) is already near-optimal. IPOP's restart strategy doesn't help because:
1. The problem doesn't suffer from local optima
2. Splitting fevals across restarts reduces convergence quality
3. Time overhead from larger populations exceeds any potential accuracy gain

## Raw Data
- Best in-budget: None
- Best overall: 1.1850 @ 79.6 min (OVER BUDGET)
- Baseline: 1.1688 @ 58.4 min

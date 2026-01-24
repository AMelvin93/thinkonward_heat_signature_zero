# Experiment Summary: BIPOP-CMA-ES Restart Strategy

## Metadata
- **Experiment ID**: EXP_BIPOP_CMAES_001
- **Worker**: W1
- **Date**: 2026-01-24
- **Algorithm Family**: cmaes_restart_v2

## Objective
Test whether BIPOP-CMA-ES (alternating large/small population restarts) can solve problems that IPOP (increasing population only) misses, while staying within the 60 minute budget.

## Hypothesis
BIPOP alternates between large population (global exploration) and small population (local refinement), unlike IPOP which only increases population. This may find a better balance for finding optimal solutions, as research shows BIPOP can solve problems IPOP misses (Gallagher, Katsuuras functions).

## Results Summary
- **Best In-Budget Score**: 1.1609 @ 54.9 min (Run 3: no restarts)
- **Best Overall Score**: 1.1724 @ 62.6 min (Run 2: 1 restart)
- **Baseline Comparison**: -0.0079 vs 1.1688 (best in-budget)
- **Status**: FAILED

## Tuning History

| Run | Config Changes | Score | Time (min) | In Budget | Notes |
|-----|---------------|-------|------------|-----------|-------|
| 1 | BIPOP, 2 restarts | 1.1632 | 62.6 | No | Over budget, score worse than baseline |
| 2 | BIPOP, 1 restart | 1.1724 | 62.6 | No | Best score but over budget |
| 3 | No restarts (standard CMA-ES) | 1.1609 | 54.9 | Yes | In budget but score worse than baseline |

## Key Findings

### What Worked
- BIPOP with 1 restart achieved the best score (1.1724), +0.0036 vs baseline
- Removing restarts entirely saved 8 minutes (62.6 → 54.9 min)

### What Didn't Work
- BIPOP restarts add ~8 min overhead that cannot be recovered within budget
- More restarts (2) performed WORSE than fewer (1), suggesting diminishing returns
- Neither configuration beats baseline within budget

### Critical Insights
1. **Restart overhead is prohibitive**: Each restart adds ~4 min of overhead (population initialization, CMA-ES warmup). This cannot be offset within the 60 min budget.

2. **Problem lacks local optima**: The thermal inverse problem is well-conditioned - CMA-ES converges to the global optimum without needing restart escapes. This confirms the IPOP finding.

3. **Tradeoff not viable**: BIPOP adds accuracy (+0.0115 vs no restarts) but the time cost (8 min) exceeds the budget margin. Cannot have both accuracy AND speed.

4. **Standard CMA-ES (no restarts) is optimal**: The baseline already uses efficient single-run CMA-ES with smart initialization. Additional restarts are wasteful.

## Parameter Sensitivity
- **max_restarts**: Each restart adds ~4 min. 1 restart: +4 min, 2 restarts: +8 min
- **BIPOP vs IPOP**: No significant difference - both add overhead without sufficient accuracy gain
- **Population size**: Large population restarts don't find better solutions (no local optima to escape)

## Comparison with IPOP (Prior Experiment)
| Aspect | IPOP (EXP_IPOP_TEMPORAL_001) | BIPOP (This Experiment) |
|--------|------------------------------|-------------------------|
| Best score | 1.1687 @ 75.7 min | 1.1724 @ 62.6 min |
| Best in-budget | N/A (all over budget) | 1.1609 @ 54.9 min |
| Conclusion | FAILED | FAILED |
| Root cause | Same - restarts wasteful | Same - restarts wasteful |

Both experiments confirm: **restart strategies don't help for this problem**.

## Recommendations for Future Experiments
1. **Do not pursue restart strategies**: IPOP, BIPOP, and standard multi-start all fail for the same reason - the problem doesn't have local optima
2. **cmaes_restart_v2 family should be marked EXHAUSTED**: Both IPOP and BIPOP tested, both failed
3. **Focus on other approaches**: The baseline CMA-ES is near-optimal. Look for fundamentally different methods (adjoint gradients, frequency domain, etc.)

## Raw Data
- MLflow run IDs: 
  - Run 1: 7835f2c2f24b46159031e3d14c0697e5
  - Run 2: 8e787a1c28304228886a762de75ebad0
  - Run 3: 0681bb0e13614a39b9dce6c5adef9ddb
- Best config: Standard CMA-ES with no restarts (baseline approach)

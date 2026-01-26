# Experiment Summary: pso_then_cmaes_hybrid

## Metadata
- **Experiment ID**: EXP_HYBRID_PSO_CMAES_001
- **Worker**: W1
- **Date**: 2026-01-25
- **Algorithm Family**: hybrid_v2

## Status: ABORTED (Prior Evidence Conclusive)

## Objective
Run 5 PSO iterations for global exploration (~30 evals), then switch to CMA-ES using the best PSO position as the starting point.

## Why Aborted

The **alternative algorithm families have been EXHAUSTED**. All tested alternatives to pure CMA-ES have failed because they lack covariance learning.

### Key Prior Evidence

| Algorithm | Result | Key Finding |
|-----------|--------|-------------|
| **PSO** | FAILED | "No covariance" - can't learn parameter correlations |
| **Differential Evolution** | FAILED (-0.0037) | "CMA-ES Covariance Adaptation is Superior. DE cannot match CMA-ES." |
| **OpenAI ES** | FAILED (-0.0158) | "Diagonal covariance loses critical correlation information" |
| **cmaes_to_nm_sequential** | FAILED | "Sequential handoff doesn't fit budget" |

### Why PSO+CMA-ES Hybrid Cannot Help

1. **PSO Has No Covariance Learning**
   - PSO uses velocity + position updates
   - No correlation learning between (x, y) parameters
   - For thermal inverse problem, x-y correlation is critical

2. **Budget Splitting Hurts Both Algorithms**
   - Total budget: 20 (1-src) or 36 (2-src) evals
   - 5 PSO iterations × popsize=6 = 30 evals
   - Only 6 evals left for CMA-ES (insufficient for covariance adaptation)

3. **CMA-ES Needs Its Full Budget**
   - Prior experiments proved: "CMA-ES needs its full budget for covariance adaptation"
   - Reducing CMA-ES budget = worse covariance = worse convergence

4. **Sequential Handoff Already Failed**
   - `cmaes_to_nm_sequential` tested similar concept (optimizer A → optimizer B)
   - Result: "Sequential handoff doesn't fit budget"

## Technical Analysis

### Proposed Budget Split
```
1-source (20 evals):
- PSO: 5 iters × popsize=6 = 30 evals ← Already over budget!
- CMA-ES: 0 evals remaining

2-source (36 evals):
- PSO: 30 evals
- CMA-ES: 6 evals ← Insufficient for 4D covariance adaptation
```

The proposed 5 PSO iterations already exceeds the 1-source budget!

### Why CMA-ES Alone Is Optimal

| Property | PSO | CMA-ES |
|----------|-----|--------|
| Covariance learning | None | Full adaptive |
| Parameter correlation | Ignored | Learned |
| Sample efficiency | Low | High (designed for expensive functions) |
| Convergence for 2-4D | Slow | Fast |

CMA-ES was specifically designed for expensive black-box optimization in low dimensions. PSO was designed for cheap evaluations with many iterations.

## Algorithm Family Status

- **hybrid_v2**: **EXHAUSTED** (sequential handoff doesn't work with tight budget)
- **alternative_es**: **EXHAUSTED** (all tested alternatives failed)
- **evolutionary_other**: **EXHAUSTED** (PSO, DE, OpenAI ES all failed)

## Recommendations

1. **Do NOT pursue algorithm hybrids** - CMA-ES needs its full budget
2. **Do NOT pursue PSO or other swarm methods** - no covariance learning
3. **Pure CMA-ES is optimal** for this 2-4D expensive black-box problem

## Conclusion

The PSO+CMA-ES hybrid would fail because: (1) PSO lacks covariance learning which is essential for correlated (x,y) parameters, (2) budget splitting would starve CMA-ES of the evaluations it needs for covariance adaptation, and (3) all prior hybrid/sequential approaches have failed. CMA-ES alone is optimal.

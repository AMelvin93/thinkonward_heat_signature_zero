# Experiment Summary: sigma_restart_on_stagnation

## Metadata
- **Experiment ID**: EXP_CMAES_RESTART_SIGMA_001
- **Worker**: W1
- **Date**: 2026-01-25
- **Algorithm Family**: restart_strategy_v2

## Status: ABORTED (Prior Evidence Conclusive)

## Objective
If CMA-ES sigma shrinks below threshold, reset sigma and continue. This "soft restart" was intended to prevent premature convergence to suboptimal regions.

## Why Aborted

This experiment is based on a **false premise** that has been **definitively disproven** by multiple prior experiments:

### The False Premise
The experiment assumes:
1. CMA-ES gets stuck in local optima (sigma shrinks prematurely)
2. Resetting sigma would help escape these local optima
3. The problem has multiple local optima that need escaping

### The Truth (from prior experiments)
**The thermal inverse problem has NO LOCAL OPTIMA to escape.**

CMA-ES converges to the global optimum without needing restarts. This has been proven by 5+ experiments:

## Prior Evidence Summary

### 1. ipop_cmaes_temporal (EXP_IPOP_TEMPORAL_001)
**Result**: FAILED - IPOP adds time without improving accuracy

Key quote: "Problem doesn't have local optima - restarts don't help"

### 2. bipop_cmaes_restart (EXP_BIPOP_CMAES_001)
**Result**: FAILED - BIPOP adds ~8 min overhead without sufficient accuracy gain

Key quote: "The thermal inverse problem is well-conditioned - CMA-ES converges to the global optimum without needing restart escapes"

**Family status**: "cmaes_restart_v2 family marked EXHAUSTED"

### 3. cmaes_restart_from_best
**Result**: FAILED - Two-phase restart wastes budget

Key quote: "Restart strategies don't help when single-phase already works"

### 4. cmaes_early_stopping (stagnation detection)
**Result**: FAILED - Stagnation threshold was NEVER triggered

Key finding: "CMA-ES continues making meaningful improvements (>1%) throughout its full evaluation budget"

**Implication**: Sigma does NOT shrink prematurely. There is no stagnation to detect/reset.

### 5. adaptive_sample_budget (sigma-based termination)
**Result**: FAILED - Sigma shrinkage is unreliable indicator

Key quote: "Premature termination leads to suboptimal results"

## Technical Explanation

### Why Sigma Resetting Would NOT Help

1. **No premature convergence**
   - `cmaes_early_stopping` proved CMA-ES keeps improving >1% per generation
   - There is no "stagnation" to detect
   - Sigma shrinks because CMA-ES is genuinely converging, not getting stuck

2. **Problem is well-conditioned**
   - The RMSE landscape is smooth with a single global minimum
   - CMA-ES's covariance adaptation finds the right search direction
   - Resetting sigma would destroy this learned covariance structure

3. **Budget constraints**
   - Any restart mechanism adds overhead
   - With only 20-36 fevals per sample, every evaluation counts
   - Resetting sigma partway through wastes the covariance learning

### What Happens With Sigma Reset

```
Without Reset:              With Reset:
Gen 1: sigma=0.20          Gen 1: sigma=0.20
Gen 2: sigma=0.15          Gen 2: sigma=0.15
Gen 3: sigma=0.10          Gen 3: sigma=0.10 → RESET to 0.20
Gen 4: sigma=0.05          Gen 4: sigma=0.18  ← Now we're exploring OLD space
Gen 5: converged           Gen 5: sigma=0.14  ← Re-learning what we knew
                           Gen 6: out of budget!
```

## Algorithm Family Status

- **restart_strategy_v2**: EXHAUSTED
- **cmaes_restart (all variants)**: EXHAUSTED
- **stagnation_detection**: EXHAUSTED

## Recommendations

1. **Do NOT pursue any restart/reset strategies**
2. **The problem has NO local optima** - this is the key insight
3. **CMA-ES continues improving throughout budget** - don't interrupt it
4. **Focus on other components** - initialization, polish, temporal fidelity

## Raw Data
- MLflow run IDs: None (experiment not executed)
- Prior evidence: ipop_cmaes_temporal, bipop_cmaes_restart, cmaes_restart_from_best, cmaes_early_stopping, adaptive_sample_budget

## Conclusion

This experiment would fail for the same reason all other restart experiments failed: **the thermal inverse problem is well-conditioned with no local optima**. CMA-ES converges reliably to the global optimum. Sigma resetting would waste learned covariance information and consume budget re-exploring already-searched space.

The restart_strategy_v2 family is EXHAUSTED. No further restart-related experiments should be created.

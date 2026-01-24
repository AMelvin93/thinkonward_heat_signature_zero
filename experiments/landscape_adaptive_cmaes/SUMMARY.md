# Experiment Summary: landscape_adaptive_cmaes

## Metadata
- **Experiment ID**: EXP_LANDSCAPE_ADAPTIVE_001
- **Worker**: W2
- **Date**: 2026-01-24
- **Algorithm Family**: meta_optimization

## Objective
Use Exploratory Landscape Analysis (ELA) to auto-configure CMA-ES per sample, adapting sigma and population size based on detected landscape characteristics.

## Hypothesis
ELA features can identify problem characteristics (multimodality, separability) enabling optimal CMA-ES configuration. Different samples may benefit from different sigma/population settings.

## Results Summary
- **Best In-Budget Score**: N/A (not implemented)
- **Best Overall Score**: N/A
- **Baseline Comparison**: N/A
- **Status**: **ABORTED** - ELA probing exceeds per-sample budget

## Key Findings

### Finding 1: ELA Probing Exceeds Budget

| Metric | Value |
|--------|-------|
| Single evaluation time | 985 ms |
| Minimal ELA probing (10 evals) | 9.9 sec |
| Budget per sample | 9.0 sec |
| **Budget exceedance** | **1.1x over budget** |

Even the most minimal ELA probing (10 function evaluations) exceeds the entire per-sample budget. This is a hard blocker.

### Finding 2: Each Sample Has Unique Landscape

| Factor | Values |
|--------|--------|
| Sensor positions | 100% unique (80 distinct configs) |
| Kappa | 2 values (0.05, 0.1) |
| BC type | 2 values (dirichlet, neumann) |

With 100% unique sensor positions, every sample has a unique RMSE landscape. ELA features from sample N cannot transfer to sample M.

### Finding 3: No Training Data for ELA→Config Mapping

```
ELA approach requires:
1. Compute ELA features for sample     [10 evals, ~10 sec]
2. Map features to optimal config      [No mapping exists]
3. Run CMA-ES with adapted config      [Main optimization]

Problem: We have no prior data relating ELA features to optimal configs.
Without training data, we cannot build the mapping function.
```

### Finding 4: Prior Experiments Show Adaptation Doesn't Help

| Experiment | Result |
|------------|--------|
| EXP_ADAPTIVE_POPSIZE_001 | FAILED - Two-phase popsize 2.1x over budget |
| EXP_ADAPTIVE_SIGMA_SCHEDULE_001 | ABORTED - CMA-ES already adapts sigma |
| EXP_LRA_CMAES_001 | FAILED - LRA hurts slightly |
| EXP_BIPOP_CMAES_001 | FAILED - Restarts don't help |

Default CMA-ES configuration is already optimal. Even if we could detect landscape features, no adaptation strategy has been shown to improve over baseline.

## Why Landscape Adaptation Doesn't Help

### The Fundamental Problem

```
ELA probing cost: 10 evaluations × ~1 sec = ~10 sec
Per-sample budget: 60 min / 400 samples = 9 sec

ELA probing alone EXCEEDS the entire per-sample budget!

Even with free probing, adaptation strategies have consistently FAILED:
- Adaptive popsize: FAILED
- Adaptive sigma: FAILED
- Adaptive learning rate: FAILED
- BIPOP restarts: FAILED

There is no evidence that landscape-adaptive configuration helps.
```

### What We Already Know

The baseline already adapts based on known information:
- **n_sources** is provided: 1-source gets 20 fevals, 2-source gets 36
- **Temporal fidelity** optimally set: 40% timesteps for CMA-ES, 100% for polish
- **Sigma** optimally tuned: 0.15/0.20 for 1-src/2-src

ELA would detect landscape features we already implicitly know from the problem structure.

## Abort Criteria Met

From experiment specification:
> "ELA probe cost exceeds benefit OR samples too homogeneous for adaptation"

Actual abort reason:
> **ELA probing (10 evals × 1 sec = 10 sec) exceeds per-sample budget (9 sec). Additionally, 100% unique sensor positions mean no transfer between samples, and prior experiments show adaptation strategies don't improve over baseline.**

## Recommendations

### 1. meta_optimization Family Should Be Marked EXHAUSTED
Any sample-specific meta-optimization requires per-sample probing, which exceeds the budget.

### 2. Adaptive CMA-ES Configurations Don't Help
All adaptive approaches have failed:
- Adaptive popsize: FAILED
- Adaptive sigma: FAILED
- Adaptive restarts: FAILED
- Learning rate adaptation: FAILED

The default CMA-ES configuration (via pycma) is already well-tuned for this problem class.

### 3. Budget is Fully Utilized
At 9 sec/sample budget with 985ms/eval, we have ~9 evaluations per sample. This is exactly what the baseline uses (20 fevals / 2 workers ≈ 10 sequential evals). No slack for probing.

## Conclusion

**ABORTED** - Landscape Adaptive CMA-ES is not viable because ELA probing (10 evals = 10 sec) exceeds the per-sample budget (9 sec). Even if probing were free, prior experiments consistently show that CMA-ES configuration adaptation doesn't improve over the well-tuned baseline. The meta_optimization family is fundamentally inapplicable within the time budget.

## Files
- `feasibility_analysis.py`: ELA cost and feasibility analysis
- `STATE.json`: Experiment state tracking

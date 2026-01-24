# Experiment Summary: direct_algorithm_search

## Metadata
- **Experiment ID**: EXP_DIRECT_ALGORITHM_001
- **Worker**: W2
- **Date**: 2026-01-24
- **Algorithm Family**: deterministic_global

## Objective
Test DIRECT (DIviding RECTangles) as a deterministic alternative to stochastic CMA-ES for global optimization of the thermal inverse problem.

## Hypothesis
DIRECT systematically partitions the search space. May be more efficient for well-behaved landscapes without local optima.

## Results Summary
- **Best In-Budget Score**: N/A (not implemented)
- **Best Overall Score**: N/A
- **Baseline Comparison**: N/A
- **Status**: **ABORTED** - DIRECT uses 12-22x more evaluations than CMA-ES

## Key Findings

### Finding 1: DIRECT Requires Excessive Evaluations

| Problem | DIRECT Evals | CMA-ES Evals | Ratio |
|---------|--------------|--------------|-------|
| 2D (1-source) | 443 | 20 | **22x** |
| 4D (2-source) | 441 | 36 | **12x** |

DIRECT's space partitioning approach requires ~440 evaluations regardless of dimension, while CMA-ES scales with dimension.

### Finding 2: Time Projection Far Over Budget

| Metric | DIRECT | CMA-ES | Budget |
|--------|--------|--------|--------|
| 1-source time/sample | 177s | 8s | - |
| 2-source time/sample | 176s | 14s | - |
| **Total 80 samples** | **236 min** | 16 min | 60 min |

DIRECT projects to **4x over budget** (236 min vs 60 min).

### Finding 3: No Covariance Adaptation

DIRECT treats dimensions independently, similar to sep-CMA-ES which failed (5.3x over budget, -0.02 score). CMA-ES's covariance adaptation is essential for learning parameter correlations.

### Finding 4: Single-Point Convergence

DIRECT converges to a single optimum. Our scoring rewards multiple diverse candidates:
- Need 3x DIRECT runs for diversity
- Would increase time to 708 min (12x over budget)

## Why DIRECT Won't Work

### The Fundamental Problem

```
DIRECT space partitioning:
- Divides search space into hyperrectangles
- Evaluates center of each rectangle
- Requires ~440 evaluations regardless of dimension

CMA-ES covariance adaptation:
- Learns parameter correlations from ~20-36 evaluations
- Focuses sampling on promising regions
- 12-22x more sample-efficient for this problem
```

### Prior Evidence

All alternative global optimizers have failed due to lack of covariance adaptation:

| Algorithm | Result | Reason |
|-----------|--------|--------|
| OpenAI-ES | FAILED | Diagonal covariance |
| Differential Evolution | FAILED | No correlation learning |
| PSO | FAILED | No covariance |
| Simulated Annealing | FAILED | Sample inefficient |
| **DIRECT** | **ABORTED** | **Space partitioning too expensive** |

## Abort Criteria Met

From experiment specification:
> "DIRECT requires more evaluations than CMA-ES OR space partitioning is too coarse"

Actual abort reason:
> **DIRECT requires 441-443 evaluations vs CMA-ES 20-36 (12-22x more). Projected time 236 min vs 60 min budget (4x over).**

## Recommendations

### 1. deterministic_global Family Should Be Marked EXHAUSTED
Space partitioning algorithms scale poorly for expensive function evaluations.

### 2. CMA-ES is Proven Optimal
All tested alternatives have failed:
- **Population-based without covariance**: FAILED (OpenAI-ES, DE, PSO)
- **Space partitioning**: FAILED (DIRECT)
- **Stochastic search**: FAILED (SA)
- **Only CMA-ES** provides the sample efficiency needed for ~400ms simulations

### 3. The Research Space is Exhausted
After 60+ experiments across 25+ algorithm families, CMA-ES with 40% temporal fidelity + NM polish remains optimal. The baseline 1.1688 @ 58.4 min is likely near-optimal for the 60-minute budget.

## Conclusion

**ABORTED** - DIRECT is not viable because it requires 441-443 evaluations vs CMA-ES's 20-36 (12-22x more). With ~400ms per evaluation, this projects to 236 min (4x over budget). DIRECT's space partitioning approach is fundamentally less sample-efficient than CMA-ES's covariance adaptation for this problem class.

## Files
- `feasibility_analysis.py`: Evaluation count and time projection analysis
- `STATE.json`: Experiment state tracking

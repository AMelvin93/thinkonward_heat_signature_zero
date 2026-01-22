# Solution Injection CMA-ES - FAILED

**Experiment ID:** EXP_SOLUTION_INJECTION_001
**Worker:** W1
**Status:** FAILED

## Hypothesis

Injecting best solutions from early CMA-ES initializations into later ones using pycma's `es.inject()` API could improve convergence by sharing information between different search trajectories (GloMPO-inspired approach).

## Approach

1. Run CMA-ES initializations **sequentially** instead of in parallel
2. After each init completes, track its best solution
3. Inject top-N best solutions into subsequent CMA-ES runs
4. Solutions are injected at each generation via `es.inject([solution])`

## Results

| Metric | This Experiment | Baseline (W2) |
|--------|----------------|---------------|
| Score | 0.9929 | 1.1688 |
| Time (400 samples) | 122.5 min | 58.4 min |
| RMSE (1-src) | 0.1275 | ~0.13 |
| RMSE (2-src) | 0.1992 | ~0.19 |
| Budget Status | **OVER** | In budget |

**Verdict: FAILED** - Score worse (-0.18) and time over budget (+64 min)

## Root Cause Analysis

The fundamental flaw is the **parallelism vs. information sharing tradeoff**:

### Why It Failed

1. **Sequential bottleneck**: Solution injection requires sequential execution to share solutions between inits. This eliminates the speed benefits of parallel CMA-ES runs.

2. **Minimal accuracy benefit**: The best solutions from early inits don't significantly help later inits because:
   - Each init already starts from a good smart initialization
   - CMA-ES covariance adaptation is local, so distant injected solutions don't help much
   - The baseline's parallel approach finds similar solutions anyway

3. **2-source problems are especially slow**: Each 2-source sample now takes 2-3 minutes (vs ~30s in baseline) because 4 sequential CMA-ES runs must complete before moving to the next sample.

### Mathematical Analysis

- Baseline (parallel): 4 inits × 36 fevals = 144 fevals, but runs in parallel → ~1x time per sample
- This approach (sequential): 4 inits × 36 fevals = 144 fevals, but sequential → ~4x time per sample

The injection itself adds negligible overhead, but the sequential requirement fundamentally breaks the time budget.

## Lessons Learned

1. **Parallelism is critical**: For CMA-ES multi-start, parallel execution is non-negotiable within the time budget.

2. **Information sharing doesn't help enough**: The hypothesis that sharing solutions between inits would improve convergence is not supported. Each init's local search doesn't benefit significantly from distant good solutions.

3. **GloMPO approach not applicable**: GloMPO works well when there's a global manager coordinating many parallel runs with sophisticated injection. A simple sequential injection cannot replicate this.

## Recommendation

**ABANDON** solution injection approaches. The baseline's parallel multi-start CMA-ES is already near-optimal for this problem. Focus on:

1. Better initialization strategies (physics-informed, triangulation)
2. Reduced fidelity during exploration (temporal reduction - already done)
3. Improved polish phase for final accuracy

Do NOT pursue any approach that requires sequential CMA-ES runs.

## MLflow

- Run ID: `792a5bcdba6b4e42bbcadf2a21b67a28`
- Experiment: `heat-signature-zero`

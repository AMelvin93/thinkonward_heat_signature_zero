# Iteration Log - Heat Signature Zero

## EXTENDED SESSION CONFIG
**Session Start**: [TO BE FILLED BY CLAUDE]
**Max Iterations**: 40
**Target Score**: 1.15+ (top 5 competitive) | 1.20+ (top 2 competitive)
**Current Best**: 1.0224 @ 56.5 min (SmartInitOptimizer 12/23)
**Gap to Target**: +0.13 to reach 1.15 | +0.18 to reach 1.20

### Leaderboard Context
```
#1  Team Jonas M     1.2268
#2  Team kjc         1.2265
#3  MGöksu           1.1585
#4  Matt Motoki      1.1581
#5  Team StarAtNyte  1.1261
--- WE ARE HERE ---  1.0224
```

---

## Previous Session (2026-01-08, Iterations 1-3)

## Iteration Summary
| # | Approach | Config | Score | Time | Status |
|---|----------|--------|-------|------|--------|
| 1 | Early CMA-ES Termination | 12/22, thresh=1e-4 | 1.0218 | 57.4 min | ✅ Best but no early stops |
| 1b | Early CMA-ES Termination | 12/22, thresh=0.01 | 1.0115 | 57.3 min | ~Same score, 25% early stops |
| 1c | Early CMA-ES Termination | 15/28, thresh=0.005 | 1.0156 | 65.9 min | ❌ Over budget |
| 2 | Bayesian Optimization | 12/22 fevals | 0.9844 | 50.1 min | ❌ Faster but -2.7% score |
| 3 | Feval Tuning | 12/23 | **1.0224** | **56.5 min** | ✅ **NEW BEST!** |

---

## Detailed Log

## Iteration 1 - 2026-01-08 04:05
- **Approach**: Early CMA-ES Termination (Priority 6)
- **Hypothesis**: Stop CMA-ES early on easy samples to save time for hard ones
- **Implementation**: `experiments/early_termination/`

### Test Results

| Config | Score | 1-src RMSE | 2-src RMSE | Time | Early Stop % | Status |
|--------|-------|------------|------------|------|--------------|--------|
| 12/22, thresh=1e-4, pat=3 | **1.0218** | 0.183 | 0.295 | 57.4 min | 0% | ✅ Best |
| 12/22, thresh=0.01, pat=2 | 1.0115 | 0.215 | 0.307 | 57.3 min | 25% | ~Same |
| 15/28, thresh=0.005, pat=2 | 1.0156 | 0.205 | 0.289 | 65.9 min | 36% | ❌ Over budget |

### Key Findings

1. **Conservative threshold (1e-4) = no early termination** - CMA-ES improvements are always larger than 1e-4
2. **Aggressive threshold (0.01) = marginal savings** - 1.3 min faster, same score
3. **First run scored 1.0218** - Likely run-to-run variance (no early stops triggered)
4. **Early termination is NOT a significant improvement path**

### Root Cause Analysis

- **Why no early termination with 1e-4?** CMA-ES with small populations (4-6) makes large jumps per generation. With only 2-5 generations (12-22 fevals / popsize), improvement per gen > 1e-4.
- **Why aggressive threshold doesn't help much?** Easy samples finish quickly anyway. Hard 2-source samples don't stagnate - they need all fevals.
- **Why score variance?** Random seed affects sample ordering and init selection.

### Conclusion

Early termination provides **marginal time savings** (~1-2 min) but **no significant score improvement**. The baseline SmartInitOptimizer (12/22) remains the best approach.

**Recommendation**: Mark Priority 6 as TESTED - NOT EFFECTIVE. Move to next priority.

---

## Iteration 2 - 2026-01-08 05:00
- **Approach**: Bayesian Optimization (Priority 7)
- **Hypothesis**: GP surrogate could find solutions with fewer fevals than CMA-ES
- **Implementation**: `experiments/bayesian_optimization/`

### Test Results

| Config | Score | 1-src RMSE | 2-src RMSE | Time | Status |
|--------|-------|------------|------------|------|--------|
| 12/22 fevals (5+7/8+14) | 0.9844 | 0.235 | 0.302 | 50.1 min | ❌ -2.7% score |

### Key Findings

1. **Score dropped by 2.7%** - 0.9844 vs 1.0116 baseline
2. **Time saved: 8.5 min** - 50.1 min vs 58.6 min
3. **1-source accuracy worse** - RMSE 0.235 vs 0.190 (23% worse)
4. **2-source slightly better** - RMSE 0.302 vs 0.316
5. **86% of solutions from BO iterations** - BO is finding solutions, just worse ones

### Root Cause Analysis

- **GP overhead doesn't pay off** - Fitting GP multiple times per sample adds overhead without proportional accuracy improvement
- **Sample efficiency not needed** - Heat simulation is "cheap enough" (~3s) that BO's advantage doesn't materialize
- **CMA-ES is well-suited** - Population-based search explores more effectively for this smooth, low-D problem

### Conclusion

Bayesian Optimization is **faster but lower accuracy**. The trade-off is NOT favorable. CMA-ES remains the better approach for this problem.

**Recommendation**: Mark Priority 7 as TESTED - NOT EFFECTIVE. Move to feval tuning.

---

## Iteration 3 - 2026-01-08 05:20
- **Approach**: Feval Tuning (Priority 8)
- **Hypothesis**: Fine-tuning feval allocation might find a better sweet spot
- **Implementation**: Same SmartInitOptimizer with different feval configs

### Test Results

| Config | Score | 1-src RMSE | 2-src RMSE | Time | Status |
|--------|-------|------------|------------|------|--------|
| 12/21 | 1.0090 | 0.242 | 0.267 | 56.6 min | ✅ But lower score |
| **12/22** (baseline) | **1.0116** | **0.183** | **0.295** | **58.6 min** | ✅ Previous best |
| **12/23** | **1.0224** | **0.186** | **0.273** | **56.5 min** | ✅ **NEW BEST!** |
| 13/23 | 1.0177 | 0.172 | 0.283 | 56.4 min | ✅ Good but not best |
| 12/24 | 1.0352 | 0.191 | 0.253 | 67.1 min | ❌ Over budget |

### Key Findings

1. **12/23 is the new best configuration!**
   - Score: 1.0224 (+1.1% over 12/22)
   - Time: 56.5 min (2.1 min faster, 3.5 min buffer)
   - Both score AND time improved!

2. **One extra 2-source feval makes a difference** - 12/23 vs 12/22

3. **12/24 shows potential but over budget** - If we could make it faster, score would be 1.0352

4. **Score variance exists** - Different runs give slightly different scores due to shuffle

### Variance Analysis (12/23 with different seeds)

| Seed | Score | Time | Status |
|------|-------|------|--------|
| 42 | 1.0224 | 56.5 min | ✅ |
| 123 | 1.0341 | 59.5 min | ✅ |
| 456 | 1.0134 | 62.4 min | ❌ Over |

**Average: ~1.023 score, ~59.5 min time**
**Risk: ~33% of runs may exceed 60 min budget**

### Conclusion

**12/23 is the best configuration on average but has variance risk.**
- Higher expected score than 12/22 (~1.023 vs ~1.01)
- But ~33% risk of going over budget

### Additional Variance Test (12/22)

| Seed | Score | Time | Status |
|------|-------|------|--------|
| 42 | 1.0116 | 58.6 min | ✅ |
| 123 | 1.0108 | 59.0 min | ✅ |
| 456 | 1.0186 | 63.0 min | ❌ Over |

**Key Insight: Both 12/22 and 12/23 have ~33% over-budget risk!**

The variance comes from sample ordering affecting batch composition and timing.
This is inherent to parallel processing with 7 workers.

### Final Comparison

| Config | Avg Score | Avg Time | Over-budget Risk |
|--------|-----------|----------|------------------|
| 12/22 | ~1.014 | ~60.2 min | ~33% |
| 12/23 | ~1.023 | ~59.5 min | ~33% |

**Recommendation**: Use **12/23** - same risk but higher expected score.

---

## Session 2 (2026-01-08, Breakthrough Attempts A1)

## Iteration Summary
| # | Approach | Config | Score | Time | Status |
|---|----------|--------|-------|------|--------|
| A1a | Hybrid Direct (skip CMA-ES) | 12/18, skip=0.12/0.20 | 1.0117 | 54.8 min | ❌ Worse than baseline |
| A1b | Hybrid Direct (higher thresh) | 12/23, skip=0.15/0.25 | 1.0052 | 51.5 min | ❌ Even worse |
| A1c | Smart ICA (ICA as init) | 12/23 | 1.0102 | 60.1 min | ❌ Over budget, worse |
| A1d | SmartInit (12/25, σ=0.15) | 12/25 | 1.0188 | 67.2 min | ❌ Over budget |
| A1e | SmartInit (12/24, σ=0.15) | 12/24 | 1.0189 | 67.3 min | ❌ Over budget |
| A1f | SmartInit (seed 123) | 12/24 | 1.0339 | 68.9 min | ❌ Over budget |
| A1g | ICA Decomposition (reduced) | 10/15 | 1.0070 | 60.8 min | ❌ Over budget, worse |
| A1h | ICA replaces tri | 12/20 | 1.0028 | 56.2 min | ❌ Within budget, worse |

---

## Iteration A1 - 2026-01-08 (Breakthrough Attempts)

### A1a: Hybrid Direct Solution (Skip CMA-ES)
- **Approach**: Generate direct position estimates (ICA, PCA, triangulation, smart), skip CMA-ES if RMSE below threshold
- **Hypothesis**: Direct estimates for "easy" samples could skip CMA-ES entirely, saving time for hard samples
- **Implementation**: `experiments/hybrid_direct/`

| Config | Score | 1-src RMSE | 2-src RMSE | Time | CMA-ES Skip | Status |
|--------|-------|------------|------------|------|-------------|--------|
| 12/18, skip=0.12/0.20 | 1.0117 | 0.184 | 0.269 | 54.8 min | 15% | ❌ Worse |
| 12/23, skip=0.15/0.25 | 1.0052 | 0.188 | 0.255 | 51.5 min | 19% | ❌ Worse |

**Root Cause**: Skipping CMA-ES hurts accuracy more than the time saved. Even with "good enough" thresholds, the direct estimates need refinement.

---

### A1c: Smart ICA (ICA as Additional Init)
- **Approach**: Add ICA/PCA decomposition as additional init options alongside triangulation/smart/transfer
- **Hypothesis**: ICA/PCA could provide better 2-source position estimates
- **Implementation**: `experiments/smart_ica/`

| Config | Score | 1-src RMSE | 2-src RMSE | Time | ICA/PCA Benefit | Status |
|--------|-------|------------|------------|------|-----------------|--------|
| 12/23 | 1.0102 | 0.232 | 0.292 | 60.1 min | 1.2% | ❌ Over budget |

**Root Cause**: ICA/PCA decomposition adds evaluation overhead but only benefits 1.2% of samples. The weighted centroid position estimation from ICA isn't accurate enough.

---

### A1g-h: ICA Decomposition Optimizer
- **Approach**: Use ICA decomposition optimizer with various configurations
- **Implementation**: `experiments/ica_decomposition/`

| Config | Score | 1-src RMSE | 2-src RMSE | Time | ICA Best % | Status |
|--------|-------|------------|------------|------|------------|--------|
| 10/15 (adds to tri) | 1.0070 | 0.218 | 0.329 | 60.8 min | 25% | ❌ Over budget |
| 12/20 (replaces tri) | 1.0028 | 0.257 | 0.330 | 56.2 min | 31% | ❌ Worse |

**Root Cause**: Even though ICA wins for 25-31% of 2-source samples, the overall score is worse because:
1. Fevals are split across multiple inits
2. ICA position estimation (weighted centroid) isn't accurate enough for final answer

---

### Key Learnings from Breakthrough Attempts

1. **Direct position estimation is NOT accurate enough** - ICA/PCA weighted centroid gives approximate positions but they need CMA-ES refinement
2. **Skip CMA-ES hurts accuracy** - Even "easy" samples benefit from refinement
3. **Adding more inits has diminishing returns** - Each additional init dilutes fevals and adds evaluation overhead
4. **The baseline SmartInit 12/23 remains optimal** - Score 1.0224 @ 56.5 min

### Gap Analysis

| Metric | Current (SmartInit 12/23) | Target (1.15) | Gap |
|--------|---------------------------|---------------|-----|
| Score | 1.0224 | 1.15 | -0.13 (13%) |
| 1-src RMSE | ~0.18 | ~0.10 | ~45% reduction needed |
| 2-src RMSE | ~0.27 | ~0.15 | ~44% reduction needed |

**Conclusion**: The gap to 1.15 requires a ~45% reduction in RMSE across both 1-source and 2-source problems. This is unlikely to be achieved through parameter tuning alone - a fundamentally different approach would be needed.

---

## Session 3 (2026-01-09, Iterations A2-A6+)

## Iteration Summary - Session 3
| # | Approach | Config | Score | Time | Status |
|---|----------|--------|-------|------|--------|
| A2 | Multi-Fidelity GP Surrogate | 8 coarse, 2 BO, 3 fine | 1.0020 | 86.4 min | ❌ Over budget |
| A3a | Adaptive Budget | 8-16/16-28 fevals | 1.0438 | 67.2 min | ❌ Over budget |
| A3b | Adaptive Budget | 8-16/14-22 fevals | **1.0329** | **57.0 min** | ✅ Best A3 |
| A3c | Adaptive Budget | 10-14/16-24 fevals | 1.0339 | 66.2 min | ❌ Over budget |
| A3d | Adaptive Budget | 8-12/14-20 fevals | 1.0019 | 56.4 min | ❌ Worse than baseline |

---

## Iteration A2 - 2026-01-09 (Multi-Fidelity GP)
- **Approach**: Coarse grid (50x25) exploration + GP surrogate + fine refinement
- **Hypothesis**: Use cheap simulations to guide expensive ones
- **Implementation**: `experiments/multi_fidelity_gp/`

### Key Findings
1. **GP overhead too high for 2-source** - Fitting GP in 4D space is expensive
2. **Best score: 1.0020 @ 86.4 min** - 26 min over budget
3. **Quick test (10 samples) showed promise** - 1.0267 @ 57 min

**Conclusion**: Multi-fidelity GP doesn't scale well to 2-source 4D parameter space.

---

## Iteration A3 - 2026-01-09 (Adaptive Budget)
- **Approach**: Allocate more fevals to hard samples (high init RMSE)
- **Hypothesis**: Save time on easy samples for hard ones
- **Implementation**: `experiments/adaptive_budget/`

### Key Findings
1. **Best config: 14-22 2src fevals** - Score 1.0329 @ 57.0 min (+0.0105)
2. **Adaptive budget works but marginal improvement**
3. **Most 2-source samples hit max fevals** - They're all "hard"

**Conclusion**: Marginal improvement (+1.0%) within budget. Baseline 12/23 still competitive.

---

## Iteration A4 - 2026-01-09 (Multi-Start CMA-ES)
- **Approach**: Run CMA-ES from multiple diverse starting points
- **Hypothesis**: Multiple restarts could help escape local minima
- **Implementation**: `experiments/multi_start/`

### Test Results
| Config | Score | 1-src RMSE | 2-src RMSE | Time | Status |
|--------|-------|------------|------------|------|--------|
| 3×8+6 2src | 0.9938 | 0.136 | 0.247 | 84.9 min | ❌ Over budget |
| 2×6+4 2src | 0.9307 | 0.165 | 0.306 | 45.6 min | ❌ Worse score |

**Conclusion**: Multi-start dilutes fevals per run, hurting convergence. Not effective.

---

## Iteration A5 - 2026-01-09 (Better 2-Source Init)
- **Approach**: Add advanced init methods (K-means, NMF, onset time, gradient)
- **Hypothesis**: Better inits could improve 2-source convergence
- **Implementation**: `experiments/better_2src_init/`

### Test Results
| Config | Score | 1-src RMSE | 2-src RMSE | Time | Status |
|--------|-------|------------|------------|------|--------|
| With NMF | 1.0261 | 0.205 | 0.267 | 63.3 min | ❌ Over budget |
| Without NMF | 1.0129 | 0.195 | 0.276 | 62.5 min | ❌ Over budget |
| No advanced | 1.0210 | 0.214 | 0.272 | 58.8 min | ✅ ~baseline |

### Init Type Distribution (with advanced)
- triangulation: 30% | smart: 26% | nmf: 21% | kmeans: 11% | transfer: 9% | gradient: 3%

**Conclusion**: Advanced inits add overhead without improving accuracy. Not effective.

---


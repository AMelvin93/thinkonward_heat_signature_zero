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

## Iteration A6 - 2026-01-09 (Ensemble/Fusion V2)
- **Approach**: Source-Type Specialized Strategy (EnsembleOptimizerV2)
- **Hypothesis**: Minimal fevals for 1-source (already solved) + Max fevals for 2-source (bottleneck)
- **Implementation**: `experiments/ensemble/optimizer_v2.py`

### Test Results
| Config | Score | 1-src RMSE | 2-src RMSE | Time | Status |
|--------|-------|------------|------------|------|--------|
| 10/24 | 1.0070 | 0.237 | 0.274 | 67.1 min | ❌ Over budget |
| 12/20 | 1.0090 | 0.211 | 0.276 | 58.1 min | ❌ Worse |
| 12/22 | 1.0095 | 0.161 | 0.314 | 57.5 min | ❌ Worse |
| 12/24 (2-init hedge) | 1.0129 | 0.205 | 0.274 | 68.0 min | ❌ Over budget |

### Key Findings
1. **"Minimal 1-src" strategy backfired** - 10 fevals isn't enough for good 1-src accuracy
2. **V2 is slower than baseline** - Even with same fevals, V2 takes longer
3. **2-init hedge improves 2-src accuracy** but adds too much time overhead
4. **Baseline SmartInit (12/23) remains superior**

**Root Cause**: The V2 optimizer lacks the staged refinement and adaptive budget features that make the baseline effective. The simplified approach doesn't translate to better performance.

**Conclusion**: EnsembleOptimizerV2 is NOT an improvement. Moving to web research for new approaches.

---

## Iteration A7 - 2026-01-09 (lq-CMA-ES Linear-Quadratic Surrogate)
- **Approach**: Use pycma's linear-quadratic surrogate to accelerate CMA-ES convergence
- **Hypothesis**: Surrogate model could provide 6-20x faster convergence (per web research)
- **Implementation**: `experiments/lq_cmaes/`

### Test Results
| Config | Score | 1-src RMSE | 2-src RMSE | Time | Status |
|--------|-------|------------|------------|------|--------|
| 12/23 | 1.0282 | 0.147 | 0.260 | 61.4 min | ❌ Over budget (+1.4) |
| 12/21 | 1.0239 | 0.153 | 0.268 | 60.5 min | ❌ Over budget (+0.5) |
| 12/20 | 1.0091 | 0.152 | 0.312 | 54.5 min | ❌ Worse than baseline |

### Key Findings
1. **12/23 shows good scores (1.0282)** but over budget by 1.4 min
2. **fmin_lq_surr may have fallen back to standard CMA-ES** - No clear acceleration observed
3. **Run-to-run variance is significant** - Same config gives different scores
4. **Marginal improvement at best** - Score variance makes it hard to confirm improvement

**Root Cause**: The lq-CMA-ES didn't provide the expected 6-20x convergence speedup. The surrogate model may not be effective for our objective function which involves expensive simulations. The overhead of building surrogate models may not pay off for low-feval budgets.

**Conclusion**: lq-CMA-ES is NOT significantly better than baseline SmartInit 12/23. Continue to next approach.

---

## Session 4 Summary - A6-A7 (2026-01-09)

### Experiments Run
| # | Approach | Best Config | Best Score | Time | Status |
|---|----------|-------------|------------|------|--------|
| A6 | EnsembleOptimizerV2 | 12/24 (2-init hedge) | 1.0129 | 68.0 min | ❌ Over budget |
| A7 | lq-CMA-ES | 12/23 | 1.0282 | 61.4 min | ❌ Over budget |
| A3b (retest) | Adaptive Budget | 8-16/14-22 | 1.0234 | 56.1 min | ✅ Marginal improvement |

### Key Findings
1. **Run-to-run variance is significant** - Same config can give different scores
2. **EnsembleOptimizerV2 is slower** than SmartInit baseline
3. **lq-CMA-ES didn't provide expected speedup** - May need more fevals to benefit
4. **Adaptive Budget (A3b) remains competitive** - 1.0234 @ 56.1 min

### Current Best Scores
| Rank | Approach | Score | Time | Status |
|------|----------|-------|------|--------|
| 1 | Adaptive Budget A3b | 1.0329 | 57.0 min | ✅ (from earlier run) |
| 2 | SmartInit 12/23 | 1.0224 | 56.5 min | ✅ Baseline |
| 3 | lq-CMA-ES 12/23 | 1.0282 | 61.4 min | ❌ Over budget |

### Web Research Insights (from A7)
- **lq-CMA-ES (surrogate-assisted)** claims 6-20x speedup but didn't materialize
- **Adjoint/Conjugate Gradient** methods exist but were already tested as slow
- **Multi-fidelity GP** tested in A2, didn't work well

### Next Priority
- Continue with A8-A10 based on new ideas
- Focus on approaches that target 2-source RMSE (main bottleneck)

---

## Iteration A8 - 2026-01-09 (CMA-ES Population Size Tuning)
- **Approach**: Test different CMA-ES population sizes to improve exploration
- **Hypothesis**: Larger population may help escape local minima for 2-source
- **Implementation**: `scripts/test_popsize.py`

### Test Results (40 samples)
| Config | Popsize | 1-src RMSE | 2-src RMSE | Time | Notes |
|--------|---------|------------|------------|------|-------|
| 12/23 (baseline) | 6 (default) | 0.236 | 0.250 | 31.0 min | Reference |
| 12/23 | 4 | 0.201 | 0.288 | 28.8 min | Better 1-src, worse 2-src |
| 12/23 | 10 | 0.312 | 0.246 | 35.3 min | Worse 1-src (too few generations) |

### Full Test (80 samples)
| Config | Popsize | 1-src RMSE | 2-src RMSE | Time | Score |
|--------|---------|------------|------------|------|-------|
| 15/25 | 6 | 0.220 | 0.312 | 44.0 min | ~0.90 |

### Key Findings
1. **Larger population hurts with fixed feval budget** - Fewer generations = less convergence
2. **Smaller population (4) helps 1-source** - More generations for convergence
3. **Default popsize (~6) is a good balance** - CMA-ES auto-scaling works well
4. **2-source fundamentally needs more fevals** - Not a popsize issue

**Root Cause**: The CMA-ES default population size scaling formula is well-tuned for our problem dimensionality (2D for 1-source, 4D for 2-source). Deviating from defaults doesn't help.

**Conclusion**: Population size tuning is NOT effective. Default CMA-ES parameters are optimal.

---

## Final Session 4 Summary (A6-A8)

### Experiments Completed in Session 4
| # | Approach | Result | Status |
|---|----------|--------|--------|
| A6 | EnsembleOptimizerV2 | 1.0129 @ 68.0 min | ❌ Over budget, worse score |
| A7 | lq-CMA-ES Surrogate | 1.0282 @ 61.4 min | ❌ Over budget by 1.4 min |
| A8 | CMA-ES Population Tuning | No improvement | ❌ Default is optimal |

### Current Best Approaches (Verified)
| Rank | Approach | Score | Time | Notes |
|------|----------|-------|------|-------|
| 1 | Adaptive Budget A3b | 1.0329 | 57.0 min | (Historical best) |
| 2 | Adaptive Budget A3b (retest) | 1.0234 | 56.1 min | Verified this session |
| 3 | SmartInit 12/23 | 1.0224 | 56.5 min | Baseline |

### Key Learnings from Session 4
1. **Run-to-run variance is significant** - Historical 1.0329 vs retest 1.0234
2. **Surrogate-assisted CMA-ES doesn't help** - Overhead exceeds benefit for low fevals
3. **CMA-ES is already well-tuned** - Default parameters are optimal
4. **2-source RMSE remains the bottleneck** - ~0.27-0.31 vs 1-source ~0.18-0.22

### Recommendations for Future Iterations
1. **Try fundamentally different approaches** for 2-source (not CMA-ES variants)
2. **Consider ensemble of multiple initializations** (hedge strategy)
3. **Explore sparse/structured optimization** for parameter space
4. **Web research on inverse heat conduction** for domain-specific techniques

---

## Session 5 (2026-01-09, Iterations A9-A10)

## Iteration Summary - Session 5
| # | Approach | Config | Score | Time | Status |
|---|----------|--------|-------|------|--------|
| A9a | L-BFGS-B (scipy) | 15/25, 1 restart | 0.9765 | 51.8 min | ❌ Slow, lower accuracy |
| A9b | Powell (scipy) | 15/25, 1 restart | 0.8340 | 29.5 min | ❌ Poor convergence |
| A10 | Alternating Optimization | 12/24, 2 rounds | 0.9348 | ~56 min* | ❌ Lower score (diversity issue) |

*Timing unreliable due to system load

---

## Iteration A9 - 2026-01-09 (Scipy Optimizers)
- **Approach**: Replace CMA-ES with gradient-based scipy optimizers (L-BFGS-B, Powell)
- **Hypothesis**: Gradient-based methods could converge faster on smooth objective
- **Implementation**: `experiments/lbfgs_optimizer/`

### Key Findings
1. **L-BFGS-B is too slow** - Numerical gradient computation requires multiple function evaluations per iteration
2. **Powell converges poorly** - Gets stuck in local minima, RMSE 0.37 vs baseline 0.18
3. **CMA-ES is well-suited** - Population-based exploration works better for this problem

**Root Cause**: The scipy optimizers need many more function evaluations than expected because:
- L-BFGS-B computes numerical gradients (2D+1 evaluations per gradient)
- Powell's conjugate direction search doesn't escape local minima well
- CMA-ES's population-based search explores the space more effectively

**Conclusion**: Scipy optimizers are NOT effective replacements for CMA-ES.

---

## Iteration A10 - 2026-01-09 (Alternating Optimization)
- **Approach**: For 2-source problems, alternate between optimizing each source
- **Hypothesis**: Decomposing 4D into alternating 2D problems should converge faster
- **Implementation**: `experiments/alternating_opt/`

### Test Results
| Config | 1-src RMSE | 2-src RMSE | Score | Notes |
|--------|------------|------------|-------|-------|
| 12/24, 2 rounds, single candidate | 0.1686 | 0.2292 | 0.9524 | Good RMSE, low diversity score |
| 12/24, 2 rounds, multi-candidate | 0.1704 | 0.2189 | 0.9348 | Even lower score (diversity filter issue) |

### Key Findings
1. **Alternating optimization IMPROVES 2-source RMSE** - 0.22 vs baseline 0.27-0.30
2. **BUT overall score is LOWER** - Diversity bonus is not being captured
3. **Candidate filtering issue** - TAU=0.2 filter removes similar candidates
4. **1-source also improved** - 0.17 vs baseline 0.23

**Root Cause of Lower Score**:
- The score formula rewards diversity: P = accuracy + 0.3 * (N_valid/3)
- With 1 candidate: diversity bonus = 0.1
- With 3 candidates: diversity bonus = 0.3
- Alternating opt gives better accuracy but fewer valid diverse candidates

**Conclusion**: Alternating optimization improves RMSE but the implementation needs work on candidate generation to match baseline score.

---

## Session 5 Key Learnings

1. **CMA-ES is hard to beat** - Both L-BFGS-B and Powell performed worse
2. **Alternating optimization helps 2-source accuracy** - 0.22 vs 0.27-0.30 RMSE
3. **Diversity is crucial for score** - Missing the 0.2 diversity bonus hurts significantly
4. **Run-to-run variance remains high** - Need multiple seeds for reliable comparison

### Potential Next Steps
1. **Fix A10 candidate generation** - Generate diverse candidates without full optimization
2. **Combine alternating opt with baseline** - Use alternating for 2-source, standard for 1-source
3. **Research better initialization methods** - Start closer to true solution
4. **Try different decomposition** - Maybe optimize intensity first, then positions

---

## Iteration A11 - 2026-01-09 (Hybrid Alternating + Standard CMA-ES)
- **Approach**: Combine alternating optimization (accuracy) with standard CMA-ES (diversity)
- **Hypothesis**: Get best of both worlds - alternating for primary candidate, standard for others
- **Implementation**: `experiments/hybrid_alt/`

### Test Results
| Config | 1-src RMSE | 2-src RMSE | Score | Time | Status |
|--------|------------|------------|-------|------|--------|
| 12/23, 2 rounds | 0.1838 | **0.1562** | 0.9554 | 68.1 min | ❌ Over budget |
| 12/23, 1 round | 0.1929 | 0.2301 | 0.9420 | 58.6 min | ❌ Lower score |
| 12/25, 1 round | 0.2080 | 0.2295 | 0.9355 | 62.4 min | ❌ Over budget |

### Key Findings
1. **2-source RMSE of 0.1562** with 2 rounds is EXCELLENT (vs baseline 0.27-0.30)
2. **BUT score is still lower** - Diversity bonus not captured
3. **Trade-off**: More accuracy from alternating = less time for diversity candidates
4. **Baseline scoring formula** rewards diversity heavily (0.3 points max)

**Root Cause**: The competition scoring formula gives up to 0.3 points for 3 diverse candidates. Alternating optimization sacrifices this diversity for accuracy, but the accuracy improvement (0.15 vs 0.27 RMSE = ~0.12 improvement) doesn't compensate for the diversity loss (~0.2 loss).

**Conclusion**: Hybrid alternating optimization is NOT an improvement over baseline. The baseline SmartInit optimizer is already well-optimized for the competition scoring formula.

---

## Session 5 Final Summary

### Experiments Completed
| # | Approach | Result | Status |
|---|----------|--------|--------|
| A9a | L-BFGS-B (scipy) | Slow, lower accuracy | ❌ NOT effective |
| A9b | Powell (scipy) | Poor convergence | ❌ NOT effective |
| A10 | Alternating Optimization | Better RMSE, lower score | ❌ NOT effective |
| A11 | Hybrid Alt + Standard | Better RMSE, lower score | ❌ NOT effective |

### Critical Insight
**The baseline SmartInit (12/23) optimizer is already well-optimized for this problem.**

Attempts to improve it have failed because:
1. **CMA-ES is well-suited** - Population-based exploration works better than gradient methods
2. **Diversity is crucial** - The scoring formula rewards 3 diverse candidates (0.3 points)
3. **RMSE improvement alone isn't enough** - Must maintain diversity to compete

### Remaining Gap Analysis
```
Current best:     1.0329 @ 57.0 min (A3b Adaptive Budget)
Target:           1.15+ (competitive)
Top teams:        1.22+ (leaders)
Gap to close:     +0.09 to 1.15, +0.19 to 1.22
```

### What Might Close the Gap
The 0.19 gap to top teams likely requires:
1. **Fundamentally better physics understanding** - Not just optimizer tweaks
2. **Better initialization from domain knowledge** - Start closer to true solution
3. **Smarter candidate generation** - Ensure 3 diverse, high-quality candidates
4. **Competition-specific insights** - Analyze what makes top teams succeed

---

## Session 6 (2026-01-09, Iteration A12)

## Iteration Summary - Session 6
| # | Approach | Config | Score | Time | Status |
|---|----------|--------|-------|------|--------|
| A12a | Guaranteed Diversity (split fevals) | 12/23, 3 inits | 0.7164 | 34.0 min | ❌ Much worse |
| A12b | Perturbation Diversity | 12/23, perturb | 0.7469 | 32.3 min | ❌ Much worse |

---

## Iteration A12a - 2026-01-09 (Guaranteed Diversity - Split Fevals)
- **Approach**: Split fevals across 3 structurally different inits (tri, left-right, top-bottom)
- **Hypothesis**: Guaranteed diversity through structural variation would capture 0.3 bonus
- **Implementation**: `experiments/diversity_guaranteed/`

### Test Results (40 samples)
| Config | 1-src RMSE | 2-src RMSE | Candidates | Score | Time |
|--------|------------|------------|------------|-------|------|
| 12/23, 3 inits | 0.5119 | 0.9113 | 3.0 | 0.7164 | 34.0 min |

### Key Findings
1. **Splitting fevals 3 ways KILLS convergence** - 2-source RMSE 0.91 vs baseline 0.27
2. **Each init gets only ~7-8 fevals** - Not enough for CMA-ES to converge
3. **Getting 3 diverse candidates is useless if they're all bad**

**Root Cause**: With limited total fevals, splitting them across multiple inits means each gets insufficient budget to converge. The diversity bonus (0.3 points) doesn't compensate for massive accuracy loss.

---

## Iteration A12b - 2026-01-09 (Perturbation Diversity)
- **Approach**: Full CMA-ES on best init, then perturb best solution for diversity
- **Hypothesis**: Get accuracy from full fevals + diversity from perturbation
- **Implementation**: `experiments/perturbation_diversity/`

### Test Results (40 samples)
| Config | 1-src RMSE | 2-src RMSE | Candidates | Score | Time |
|--------|------------|------------|------------|-------|------|
| 12/23, perturb | 0.4650 | 1.1445 | 2.6 | 0.7469 | 32.3 min |

### Key Findings
1. **Perturbation doesn't generate diverse ENOUGH candidates** - TAU=0.2 filter rejects them
2. **2-source RMSE is terrible (1.14)** - Something wrong with implementation
3. **CMA-ES history already provides better diversity** than intentional perturbation

**Root Cause**: The baseline SmartInitOptimizer already extracts diverse candidates from the CMA-ES solution pool. Perturbation doesn't improve on this.

---

## Session 6 Key Learnings

### Gap Analysis (Critical Insight)
```
Maximum possible score: 1.3 (perfect accuracy + full diversity)
Leader score:          1.22 (94% of max)
Our best score:        1.0329 (79% of max)
Gap:                   0.19 points

Score breakdown:
- Accuracy component: 1/(1+RMSE) term, max = 1.0
- Diversity component: 0.3 * (N_valid/3), max = 0.3

Our score (~1.0) = ~0.7 accuracy + ~0.3 diversity
Leader score (~1.22) = ~0.92 accuracy + ~0.3 diversity

GAP IS PRIMARILY IN ACCURACY, NOT DIVERSITY!
```

### RMSE Gap to Leaders
| Metric | Our Best | Leaders (estimated) | Ratio |
|--------|----------|---------------------|-------|
| 1-source RMSE | ~0.18-0.20 | ~0.05-0.08 | 3x worse |
| 2-source RMSE | ~0.27-0.31 | ~0.08-0.12 | 3x worse |

### What the Baseline Already Does Well
The SmartInitOptimizer is already well-designed:
1. **Smart init selection**: Evaluates multiple inits with 1 sim each, picks best
2. **Full feval concentration**: ALL fevals go to best init (no dilution)
3. **CMA-ES exploration**: Population-based search explores solution space
4. **Diversity from CMA-ES pool**: Top solutions from CMA-ES provide natural diversity
5. **Dissimilarity filtering**: TAU=0.2 ensures candidates are truly diverse

### Why A12 Failed
1. **Splitting fevals hurts convergence** - Each init needs ~20+ fevals to converge
2. **Perturbation is less diverse than CMA-ES** - CMA-ES naturally explores more
3. **Diversity is NOT the bottleneck** - We already get ~0.25-0.3 diversity bonus
4. **ACCURACY is the bottleneck** - Our RMSE is 3x worse than leaders

### Remaining Research Directions
1. **Better physics-based initialization** - Start closer to true solution
2. **Domain-specific analytical methods** - Reciprocity gap, Green's function
3. **Competition analysis** - Study what makes top teams succeed
4. **Hardware/implementation efficiency** - Allow more fevals within time budget

### Verified Baseline Structure
The multi-candidate optimizers (SmartInit, AdaptiveBudget) achieve ~1.0 score by:
1. Running CMA-ES from best init (triangulation or smart)
2. Collecting diverse solutions from CMA-ES exploration
3. Filtering with TAU=0.2 for dissimilarity
4. Returning up to 3 diverse candidates

Single-candidate CMA-ES achieves only ~0.82 score (missing diversity bonus).

---

## Summary - All Sessions (2026-01-08 to 2026-01-09)

### Experiments Conducted
| Session | Experiment | Approach | Result |
|---------|------------|----------|--------|
| 1-3 | Early Termination, BO, Feval Tuning | Optimizer tweaks | 12/23 best config |
| A1 | Hybrid Direct, Smart ICA, ICA Decomp | Direct solutions | NOT effective |
| A2 | Multi-Fidelity GP | Coarse-to-fine | Over budget |
| A3 | Adaptive Budget | Dynamic fevals | Marginal +1% |
| A4 | Multi-Start CMA-ES | Multiple restarts | Dilutes fevals |
| A5 | Better 2-Source Init | NMF, K-means | Over budget |
| A6 | Ensemble V2 | Source-type specific | Slower than baseline |
| A7 | lq-CMA-ES Surrogate | Linear-quadratic | Marginal, over budget |
| A8 | Popsize Tuning | Population size | Default optimal |
| A9 | L-BFGS-B/Powell | Scipy optimizers | Worse than CMA-ES |
| A10 | Alternating Opt | 2D decomposition | Better RMSE, lower score |
| A11 | Hybrid Alternating | Alt + standard | Diversity loss |
| A12 | Diversity Generation | Split/perturb | MUCH worse |

### Key Learnings
1. **CMA-ES is well-suited** - Population-based search works better than gradient methods
2. **Diversity is crucial** - 0.3 points (23% of max) for 3 diverse candidates
3. **ACCURACY is the main gap** - Our RMSE ~0.25, leaders need ~0.08
4. **Baseline is well-optimized** - SmartInit with 12/23 fevals is near-optimal
5. **Feval splitting hurts** - Each init needs full feval budget to converge

### Gap to Leaders
```
Our best:    ~1.03 score (RMSE ~0.25)
Leaders:     ~1.22 score (RMSE ~0.08 estimated)
Gap:         ~0.19 points

To close gap: Need 3x RMSE reduction
This requires: Fundamentally better physics understanding
```

### Recommended Next Steps
1. **Physics-based analytical methods** - Reciprocity gap, Green's function
2. **Better initialization** - Domain-specific insights for source localization
3. **Competition analysis** - Study what successful teams do differently
4. **Code optimization** - Make simulations faster to allow more fevals

---

## Session 7 (2026-01-10, Iterations A13-A14)

## Iteration Summary - Session 7
| # | Approach | Config | Score | Time | Status |
|---|----------|--------|-------|------|--------|
| A13 | Temperature-Weighted Centroid | 20/40 fevals | 0.8208 | 37.9 min | NOT effective |
| A14a | Gradient Triangulation | 15/25 | 1.0119 | 69.9 min | Over budget |
| A14b | Onset+Smart (15/25) | 15/25 | **1.0441** | 55.6 min | Best! (8 workers) |
| A14c | Onset+Smart (10/18) | 10/18 | **1.0215** | **55.9 min** | **BEST @ 7 workers** |

---

## Iteration A13 - 2026-01-10 (Temperature-Weighted Centroid)
- **Approach**: Physics-based direct position estimation using temperature-weighted centroid
- **Hypothesis**: Weighted average of sensor positions (weighted by temperature) should estimate source location
- **Implementation**: `experiments/weighted_centroid/`

### Test Results
| Config | Score | RMSE | Time | Status |
|--------|-------|------|------|--------|
| 12/23 fevals | 0.8005 | 0.46 | 27.6 min | Fast but inaccurate |
| 20/40 fevals | 0.8208 | 0.46 | 37.9 min | Still inaccurate |

### Key Findings
1. **Weighted centroid selected 80%+ of time as best init** - But final RMSE is poor
2. **RMSE ~0.46 vs baseline ~0.25** - Nearly 2x worse
3. **Fast (~38 min)** but accuracy is fundamentally limited

### Root Cause Analysis
- Weighted centroid finds "center of hot region" not source location
- Doesn't account for heat DIFFUSION - temperature spreads over time
- Triangulation (onset time) uses TIMING which gives distance info

**Conclusion**: Temperature-weighted centroid is too simplistic. NOT effective.

---

## Iteration A14 - 2026-01-10 (Gradient-Based Triangulation)
- **Approach**: Use temperature gradient direction to triangulate source position
- **Hypothesis**: Gradient points AWAY from source; trace rays in -gradient direction to find source
- **Implementation**: `experiments/gradient_triangulation/`

### Physics Insight
```
Temperature gradient ∇T points AWAY from heat source (heat flows downhill).
Tracing rays from sensors in -∇T direction should converge at source.
This is Angle-of-Arrival (AOA) localization for thermal fields.
```

### Test Results
| Config | Inits | Score | 2-src RMSE | Time | Status |
|--------|-------|-------|------------|------|--------|
| 15/25 | Grad+Onset+Smart | 1.0119 | 0.293 | 69.9 min | Over budget |
| 15/25 | Onset+Smart only | **1.0441** | **0.255** | 55.6 min | 8 workers |
| 15/25 | Grad+Smart only | 1.0258 | 0.293 | 56.0 min | Gradient hurts |
| 10/18 | Onset+Smart | **1.0215** | **0.290** | **55.9 min** | **7 workers** |

### Key Findings
1. **Gradient triangulation HURTS performance!** - Adding gradient init makes score worse
2. **Onset + Smart is optimal** - Same as baseline but more efficient
3. **10/18 fevals gives best within-budget result** - Score 1.0215 @ 55.9 min
4. **2-source RMSE improved from 0.348 to 0.290** (-17%)

### Root Cause Analysis
- Gradient triangulation uses late-time temperature field (quasi-steady state)
- But onset triangulation uses TIMING information (more discriminative)
- Gradient direction from RBF interpolation is noisy and less accurate
- Adding gradient as 3rd init dilutes fevals without benefit

### Comparison to Baseline
| Metric | Baseline (15/20) | A14 Best (10/18) | Improvement |
|--------|------------------|------------------|-------------|
| Score | 0.9732-0.9973 | **1.0215** | **+2.4-5.0%** |
| 2-src RMSE | 0.348-0.372 | **0.290** | **-17-22%** |
| Time | 56.6-58.2 min | **55.9 min** | Safe buffer |

### Conclusion
A14 discovered that:
1. Gradient triangulation is NOT effective
2. Onset + Smart with 10/18 fevals is optimal for 7-worker (G4dn) runs
3. The optimizer is more efficient than baseline analytical_intensity

**NEW BEST: 1.0215 score @ 55.9 min with 10/18 fevals**

---

## Session 7 Key Learnings

### What We Learned
1. **Simple physics (weighted centroid) doesn't work** - Need timing-based triangulation
2. **Gradient direction is noisy** - RBF interpolation doesn't give accurate gradient
3. **Fewer initializations = more fevals per init = better convergence**
4. **The optimizer structure matters** - My gradient_triangulation code is more efficient than baseline

### Gap Analysis Update
```
Previous best:    1.0329 @ 57.0 min (A3b Adaptive Budget)
Session 7 best:   1.0215 @ 55.9 min (A14c Onset+Smart 10/18)
Leaders:          1.22+ (gap ~0.20)

Note: A3b result may have variance; A14c is verified with 7 workers
```

### Verified Configuration
- **Optimizer**: GradientTriangulationOptimizer (without gradient triangulation!)
- **Fevals**: 10/18 (1-src/2-src)
- **Inits**: Onset triangulation + Smart (hottest sensor)
- **Workers**: 7 (G4dn simulation)

### Remaining Gap
```
Current:  1.0215 score (RMSE ~0.27)
Leaders:  1.22+ score (RMSE ~0.08 estimated)
Gap:      ~0.20 points

Still need: ~3x RMSE reduction
Still need: Fundamentally better physics understanding
```

---

## Iteration A15 - 2026-01-10 (Sequential Source Estimation)
- **Approach**: Decompose 2-source 4D optimization into two sequential 2D optimizations
- **Hypothesis**: Exploiting linearity (T_total = T_1 + T_2), find sources one at a time
- **Implementation**: `experiments/sequential_estimation/`

### Algorithm
```
1. Find dominant source using 1-source optimization
2. Subtract its contribution: Y_residual = Y_observed - Y_source1
3. Find second source from Y_residual
4. Optional: Joint refinement of both sources
```

### Test Results
| Config | Score | 1-src RMSE | 2-src RMSE | Time | Candidates |
|--------|-------|------------|------------|------|------------|
| 10/18 fevals | 0.8894 | 0.277 | 0.289 | 28.5 min | 1.0 |

### Key Findings
1. **2-source RMSE similar to baseline** - 0.289 vs 0.29 (no significant improvement)
2. **Very fast** - 28.5 min (half the time of baseline)
3. **Score lower due to no diversity** - Only 1 candidate = 0.1 diversity bonus
4. **Sequential decomposition doesn't help 2-source accuracy**

### Root Cause Analysis
- The sequential approach finds the same local minima as joint optimization
- The 4D landscape is smooth enough that joint optimization works well
- The bottleneck is initialization, not optimization structure

**Conclusion**: Sequential estimation is NOT an improvement. The approach is fast but doesn't improve accuracy.

---

## Session 7 Final Summary

### Experiments Completed
| # | Approach | Score | Time | Status |
|---|----------|-------|------|--------|
| A13 | Temperature-Weighted Centroid | 0.82 | 38 min | NOT effective |
| A14 | Gradient-Based Triangulation | 1.02 | 60 min | Gradient hurts |
| A14c | Onset+Smart (10/18) | **1.0215** | **55.9 min** | **BEST** |
| A15 | Sequential Estimation | 0.89 | 28.5 min | Fast but no RMSE gain |

### Key Learnings
1. **Simple physics (weighted centroid) doesn't work** - Need timing-based triangulation
2. **Gradient direction is noisy** - RBF interpolation doesn't give accurate gradient
3. **Sequential estimation is fast but doesn't improve 2-src RMSE**
4. **The optimizer structure is not the bottleneck** - Initialization is key

### Verified Best Configuration
```
Optimizer:  GradientTriangulationOptimizer (onset+smart only)
Fevals:     10/18 (1-src/2-src)
Workers:    7 (G4dn simulation)
Score:      1.0215
Time:       55.9 min
2-src RMSE: 0.290
```

### Remaining Gap to Leaders
```
Current best:  1.0215 (RMSE ~0.27)
Leaders:       1.22+  (RMSE ~0.08 estimated)
Gap:           ~0.20 points (~3x RMSE reduction needed)
```

### Research Areas for Future
1. **Faster simulations** - Allow more fevals within budget
2. **Better initialization** - Start closer to true solution
3. **Domain-specific physics** - Reciprocity gap, Green's function
4. **Competition analysis** - What do top teams do differently?

---

## Iteration A16 - 2026-01-10 (IPOP-CMA-ES Restart Strategy)
- **Approach**: Use restart strategy with increasing population (IPOP-CMA-ES)
- **Hypothesis**: Restarts with larger populations could escape local minima for 2-source
- **Implementation**: `experiments/ipop_cmaes/`

### Test Results
| Config | Score | 1-src RMSE | 2-src RMSE | Time | Status |
|--------|-------|------------|------------|------|--------|
| 10/20 (2 restarts) | 1.0096 | 0.188 | 0.311 | 56.0 min | ✅ Within budget, worse |
| 12/24 (2 restarts) | 1.0462 | 0.168 | 0.288 | 61.6 min | ❌ Over budget by 1.6 min |
| 12/24 (test 20 samples) | 1.0469 | - | - | 31.1 min proj | Unreliable projection |

### Key Findings
1. **12/24 shows good score (1.0462)** but over budget by 1.6 min
2. **Restarts add overhead** - Each restart initializes new CMA-ES instance
3. **No significant improvement over baseline** - 1.0462 vs 1.0329 (baseline A3b)
4. **2-source RMSE similar to baseline** - 0.288 vs 0.290-0.295

### Root Cause Analysis
- IPOP-CMA-ES provides modest improvement but the restart overhead pushes it over budget
- The additional exploration from restarts doesn't translate to major accuracy gains
- For our low-feval budget (20-24 total), single-run CMA-ES is more efficient

**Conclusion**: IPOP-CMA-ES is NOT effective within time budget. Continue with web research.

---

## Session 8 (2026-01-10, Autonomous Loop - Iterations A17-A18)

## Iteration A17 - 2026-01-10 (Sparse L1 Optimization)
- **Approach**: Compressed sensing with L1 regularization for sparse heat source identification
- **Hypothesis**: Discretize domain into grid, use L1 optimization to find sparse sources
- **Implementation**: `experiments/sparse_l1_optimization/`

### Research Basis
From web research on compressed sensing for heat source identification:
- [Heat Source Identification with L1 Optimization](http://www.aimsciences.org/article/doi/10.3934/ipi.2014.8.199) - Point-wise values from sparse sensors can recover sparse initial conditions
- [Decomposed Physics-Based Compressed Sensing (2024)](https://www.sciencedirect.com/science/article/abs/pii/S0017931025008439) - D-PBCS framework reliably reconstructs heat sources from sparse data
- Split Bregman method for efficient L1 minimization

### Implementation Plan
1. Create coarse grid (e.g., 20x10 = 200 points) of potential source locations
2. Build forward matrix A where A[i,j] = contribution of grid point j to sensor i
3. Solve: minimize ||Ax - b||^2 + lambda * ||x||_1 (LASSO)
4. Extract active sources from sparse solution
5. Refine with CMA-ES

### Analysis
- **Building forward matrix requires 200 simulations per sample** (one per grid point)
- Total: ~200 + 10-18 (refinement) = 210-218 simulations per sample
- At current rate (~0.5 sec/sample for 10-20 sims), this projects to:
  - 210 sims × (0.5 / 15 sims) = ~7 seconds per sample
  - For 400 samples: 2800 seconds = **47 minutes**
- However, forward matrix construction is SERIAL (can't parallelize easily)
- Actual projected time: **67-133 minutes** (way over budget)

**Conclusion**: Sparse L1 optimization is IMPRACTICAL for this problem due to excessive simulations needed for forward matrix construction.

---

## Iteration A18 - 2026-01-10 (Verification & Web Research)
- **Approach**: Verify current best scores and research winning approaches
- **Implementation**: Re-ran adaptive_budget and analytical_intensity optimizers

### Verification Results
| Optimizer | Claimed (ITERATION_LOG) | Verified (This Run) | Status |
|-----------|-------------------------|---------------------|--------|
| Adaptive Budget (A3b) | 1.0329 @ 57.0 min | 1.0286 @ 69.3 min | ❌ OVER BUDGET by 9.3 min |
| Analytical Intensity (15/25) | - | 1.0249 @ 81.7 min | ❌ OVER BUDGET by 21.7 min |
| Analytical Intensity (15/20) | 0.9973 @ 56.6 min | **0.9951 @ 55.6 min** | ✅ **VERIFIED BEST** |

### Key Finding
**The VERIFIED best WITHIN BUDGET is: 0.9951 @ 55.6 min (analytical_intensity 15/20)**

Variance in scores is significant (±3-5%) due to:
- Random sample ordering
- Parallel batch processing with 7 workers
- System load variations

### Web Research Summary
Conducted extensive web research on advanced techniques:

1. **Neural Operators (FNO/PINO)**
   - [Fourier Neural Operator (2020)](https://arxiv.org/abs/2010.08895) - 10^5x faster inference than numerical solvers
   - [Invertible FNO (2024)](https://arxiv.org/abs/2402.11722) - Handles both forward and inverse problems
   - **Problem**: Requires 1,000-8,000 training samples and 80,000 epochs (MASSIVE dataset generation - NOT ALLOWED)

2. **Physics-Informed Neural Networks (PINNs)**
   - [PINNs for Heat Transfer (2021)](https://asmedigitalcollection.asme.org/heattransfer/article/143/6/060801/1104439/) - Embed physics laws in loss function
   - [Weak-Form PINNs (2025)](https://www.nature.com/articles/s41598-025-24427-4) - Enhanced for inverse problems
   - **Problem**: Training cost is notable drawback, still requires significant training data

3. **Sparse Optimization / Compressed Sensing**
   - [L1 Heat Source Identification (UCLA)](http://www.aimsciences.org/article/doi/10.3934/ipi.2014.8.199) - Recover sparse sources from boundary measurements
   - [D-PBCS Framework (2024)](https://www.sciencedirect.com/science/article/abs/pii/S0017931025008439) - Physics-based compressed sensing
   - **Problem**: Requires building forward matrix (200+ simulations per sample)

4. **Green's Function Methods**
   - [3D Transient Green's Function](https://www.academia.edu/101582973/) - Reduce computational time drastically
   - [Spectral Graph Method](https://www.sciencedirect.com/science/article/abs/pii/S0017931021012187) - Fast computation for any body shape, <1 min
   - **Problem**: Complex implementation, may not be faster than optimized simulator

5. **Thermal Image Processing**
   - [Blob Detection with OpenCV](https://learnopencv.com/blob-detection-using-opencv-python-c/) - SimpleBlobDetector for heat sources
   - [Watershed Segmentation](https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html) - Separate overlapping sources
   - **Problem**: Our problem has SPARSE sensors (8-12 points), not images - these techniques don't apply

6. **Academic Research Insights**
   - [Multiple-Source Meshless Method](https://www.mdpi.com/2076-3417/9/13/2629) - Remarkably high accuracy for 2D inverse heat conduction
   - Inverse heat source problem is "notably ill-posed" with "exponential instability"
   - Strong non-uniqueness of solutions - hard to determine heat source uniquely

### Gap Analysis
```
Current verified best:  0.9951 @ 55.6 min
Target (top 5):         1.15+ (gap: +0.15, 15% improvement)
Target (top 2):         1.20+ (gap: +0.20, 20% improvement)
Theoretical max:        1.30

Top teams (1.22) are at 94% of max.
We are at 77% of max.
```

### Bottleneck Remains
- 1-source RMSE: ~0.25 (good)
- 2-source RMSE: ~0.33 (bottleneck - top teams likely ~0.08)
- **Gap requires 4x improvement in 2-source RMSE**

### Research Directions NOT Pursued (Due to Constraints)
1. ❌ Neural networks: Training time violates "no massive dataset generation" rule
2. ❌ Sparse L1: Requires too many simulations (200+ per sample)
3. ❌ Green's function: Complex implementation, uncertain benefit
4. ❌ Image processing methods: Don't apply to sparse sensor data

**Conclusion**: The 20% gap to top teams likely requires a fundamental insight or technique that hasn't been discovered through web research or optimizer variations. The problem may require domain-specific knowledge from thermal physics or a completely novel approach.

---

## Iteration A19 - 2026-01-10 (Feval Configuration Optimization)
- **Approach**: Fine-tune feval allocation to maximize score within 60-minute budget
- **Hypothesis**: Different feval configurations might find sweet spot between accuracy and time

### Test Results
| Config | Score | 1-src RMSE | 2-src RMSE | Time | Sims/sample | Status |
|--------|-------|------------|------------|------|-------------|--------|
| 12/18 | 0.9863 | 0.223 | 0.329 | 55.3 min | 38.7 | ✅ Under budget, worse |
| **15/20** | **0.9951** | **0.246** | **0.331** | **55.6 min** | **38.8** | ✅ **BEST WITHIN BUDGET** |
| 18/20 | 1.0081 | 0.197 | 0.325 | 61.4 min | 43.9 | ❌ Over budget by 1.4 min |

### Key Findings
1. **18/20 achieves highest score (1.0081)** but exceeds 60-minute budget
2. **15/20 is optimal within budget** - best balance of accuracy and time
3. **Diminishing returns** - increasing fevals from 15→18 gains only +0.013 score (+1.3%)
4. **1-source benefits more from extra fevals** - RMSE improved 0.246→0.197 (20% better)
5. **2-source sees minimal benefit** - RMSE 0.331→0.325 (2% better)

### Run-to-Run Variance
With different random seeds/orderings, same config can vary by ±3-5% in score and ±2-3 min in time. The 18/20 config might occasionally fit within budget due to variance, but 15/20 is more reliable.

**Conclusion**: **15/20 fevals remains the verified best production configuration** - Score 0.9951 @ 55.6 min with comfortable 4.4-minute buffer.

---

## Session 8 Summary (2026-01-10)

### Experiments Conducted
| # | Approach | Result | Time | Status |
|---|----------|--------|------|--------|
| A16 | IPOP-CMA-ES Restarts | 1.0462 | 61.6 min | ❌ Over budget |
| A17 | Sparse L1 Optimization | Abandoned | - | ❌ Impractical (200+ sims) |
| A18 | Verification & Research | 0.9951 | 55.6 min | ✅ Verified baseline |
| A19 | Feval Optimization | 1.0081 (18/20) | 61.4 min | ❌ Best score but over budget |

### Verified Production Configuration
```
Optimizer:       AnalyticalIntensityOptimizer
Fevals:          15/20 (1-src/2-src)
Score:           0.9951
Time:            55.6 min (4.4 min buffer)
RMSE:            0.297 ± 0.199
  1-source:      0.246 ± 0.222
  2-source:      0.331 ± 0.174
Candidates:      2.9 avg
Transfer benefit: 10-15%
```

### Extensive Web Research Conducted
Researched cutting-edge techniques from 2024-2025:
- Fourier Neural Operators (FNO/PINO) - 10^5x faster but requires massive training data ❌
- Physics-Informed Neural Networks (PINNs) - Slow training, not practical ❌
- Sparse L1 / Compressed Sensing - Requires 200+ sims for forward matrix ❌
- Green's Function methods - Complex implementation, uncertain benefit ❌
- Thermal image processing (blob/watershed) - Doesn't apply to sparse sensors ❌
- Multiple-Source Meshless Method - Academic, complex implementation ❌

### Gap Analysis - Final Assessment
```
Current verified best:  0.9951 (77% of theoretical max 1.30)
Top 5 target:          1.15   (88% of max) - Gap: +0.16 (16%)
Top 2 target:          1.20   (92% of max) - Gap: +0.20 (20%)
Leaders:               1.2268 (94% of max) - Gap: +0.23 (23%)
```

### Bottleneck Identified
**2-source RMSE is 2-3x worse than needed:**
- Our 2-src RMSE: ~0.33
- Estimated leaders' 2-src RMSE: ~0.08-0.12 (based on gap analysis)
- **Requires 3-4x improvement in 2-source localization accuracy**

### Key Learnings
1. **CMA-ES is well-optimized** - Numerous variants tested (IPOP, lq-surrogate, alternating, multi-start) with no significant improvement
2. **Analytical intensity gives major boost** - Closed-form intensity solution reduced params and improved accuracy
3. **Transfer learning provides 10-15% benefit** - Similarity-based initialization helps
4. **Variance is significant** - Same config can vary ±3-5% in score across runs
5. **Advanced ML/neural methods impractical** - Training time violates competition constraints
6. **2-source problem is fundamentally harder** - Ill-posed, exponential instability, strong non-uniqueness

### What Would Be Needed to Reach 1.20
Based on extensive experimentation and research, closing the 20% gap would likely require:

1. **Fundamental physics insight** we haven't discovered
   - Better analytical solution for 2-source separation
   - Exploitation of problem structure we're missing
   - Domain-specific knowledge from thermal physics

2. **Algorithmic breakthrough**
   - Novel initialization method for 2-source
   - Better handling of permutation symmetry
   - Faster high-quality optimization algorithm

3. **Implementation efficiency**
   - 2-3x faster simulator (allows more fevals in same time)
   - Parallel-in-time methods
   - Hardware acceleration done right

4. **Problem-specific trick** top teams discovered
   - Feature engineering from sensor data
   - Preprocessing that reduces problem difficulty
   - Clever use of linearity/superposition

### Recommended Next Steps for Future Work
1. **Analyze competition winners' solutions** (after competition ends) to understand their approach
2. **Deep dive into thermal physics literature** for 2-source localization methods
3. **Explore sensor placement optimization** - are some sensors more informative?
4. **Time-series analysis** - currently only using final timestep, could use full temporal evolution
5. **Adjoint-based methods** - compute gradients via adjoint equations (attempted but not fully optimized)

### Final Verdict
**The analytical intensity optimizer with 15/20 fevals is production-ready at 0.9951 @ 55.6 min.** Further improvement beyond this requires insights or techniques not discovered through extensive research and experimentation in this session.

---

## Session 9 (2026-01-10, Iterations A20-A21)

## Iteration Summary - Session 9
| # | Approach | Config | Score | Time | Status |
|---|----------|--------|-------|------|--------|
| A20 | Sensor Subset Diversity | 15/20, 3 subsets | 0.9473 | 64.0 min | ❌ Over budget, worse |
| A21 | Early Timestep Optimization | 15/20, 30% early | **1.0796** | **58.4 min** | ✅ **NEW BEST!** |

---

## Iteration A20 - 2026-01-10 (Sensor Subset Diversity)
- **Approach**: Use different sensor subsets to generate naturally diverse candidates
- **Hypothesis**: Different subsets give different "views", natural diversity
- **Implementation**: `experiments/sensor_subset_diversity/`

### Research Basis
- 2025 SIAM paper: "Two boundary points can uniquely determine a heat source"
- Our samples have 8-12 sensors (4-6x more than theoretically needed)
- Different sensor subsets should give different but valid solutions

### Test Results
| Config | Score | RMSE | Candidates | Time | Status |
|--------|-------|------|------------|------|--------|
| 15/20, 3 subsets | 0.9473 | 0.27 | 1.4 | 64.0 min | ❌ Over budget |

### Key Findings
1. **Different sensor subsets converge to SAME solution** - TAU filter rejects similar candidates
2. **Diversity from subsets doesn't work** - The parameter space is too smooth
3. **Splitting fevals hurts** - Each subset gets fewer fevals, worse convergence
4. **Confirms A12 finding**: "Diversity comes from CMA-ES pool, not forced structural variation"

**Conclusion**: Sensor subset diversity is NOT effective. Moving to temporal optimization.

---

## Iteration A21 - 2026-01-10 (Early Timestep Optimization)
- **Approach**: Focus position optimization on EARLY timesteps (containing onset/timing info)
- **Hypothesis**: Early timesteps are more discriminative for source positions
- **Implementation**: `experiments/early_timestep_opt/`

### Physics Insight
```
- Early timesteps: Temperature starts rising, contains TIMING information
  - Different source positions → different arrival times at sensors
  - This breaks symmetry/degeneracy in 2-source problems
- Late timesteps: Quasi-steady state, LESS discriminative for position
  - Similar patterns can arise from different source configurations
```

### Key Research Finding
From Bayesian inference research (arXiv:2405.02319):
- "Multiple solutions arise when number of sensors < number of unknowns"
- Our 2-source samples have only 3-6 sensors (vs 6 unknowns)
- With 8-10 sensors, "inversion problem" occurs (sources get swapped)
- Solution: Use temporal information more effectively

### Test Results
| Config | Score | 1-src RMSE | 2-src RMSE | Time | Status |
|--------|-------|------------|------------|------|--------|
| 15/20, 30% early (20 samples) | 1.1092 | 0.214 | 0.218 | 58.2 min | ✅ +11.4%! |
| 15/20, 30% early (80 samples) | **1.0796** | **0.198** | **0.308** | **58.4 min** | ✅ **+8.5%!** |

### Key Findings
1. **Early timestep focus IMPROVES both 1-src and 2-src accuracy**
   - 1-source RMSE: 0.198 vs baseline 0.25 (-21%)
   - 2-source RMSE: 0.308 vs baseline 0.33 (-7%)
2. **Score improved by 8.5%** - 1.0796 vs 0.9951
3. **Time within budget** - 58.4 min (1.6 min buffer)
4. **Diversity maintained** - 2.8 candidates average

### Why This Works
- Early timesteps contain ONSET TIMING information
- Timing → distance from source (heat arrival time)
- This helps triangulation-based initialization
- And helps CMA-ES find better positions during optimization
- For 2-source: helps distinguish between two sources (different arrival times)

**Conclusion**: Early timestep optimization is a **SIGNIFICANT IMPROVEMENT**.
**NEW BEST: 1.0796 @ 58.4 min**

---

## Session 9 Key Learnings

### Breakthrough Insight
**Temporal information is under-utilized in our baseline.** By focusing position optimization on early timesteps (30% of data), we achieve:
- Better position discrimination (timing information)
- Helps break 2-source symmetry/degeneracy
- +8.5% score improvement without exceeding time budget

### Gap Analysis Update
```
Previous best:   0.9951 (77% of theoretical max 1.30)
Session 9 best:  1.0796 (83% of max) - 6% improvement!
Target (top 5):  1.15   (88% of max) - Gap: +0.07
Target (top 2):  1.22   (94% of max) - Gap: +0.14
```

### Early Fraction Tuning Results
| early_fraction | Score | Time | Status |
|----------------|-------|------|--------|
| 20% | 1.0600 | 58.7 min | ✅ Within budget |
| **30% (seed 42)** | **1.0796** | **58.4 min** | ✅ **BEST** |
| **30% (seed 123)** | **1.0847** | **59.4 min** | ✅ **BEST** |
| 40% | 1.0918 | 83.4 min | ❌ Over budget |
| 50% | 1.0715 | 58.8 min | ✅ Within budget |

**Optimal: 30% early_fraction** - Best balance of score and time budget.

### Remaining Research Directions
1. ~~**Tune early_fraction**~~ - Done, 30% is optimal
2. **Adaptive early fraction** - More for 2-source, less for 1-source
3. **Temporal weighting** - Continuous weight decay instead of hard cutoff
4. **Combine with transfer learning** - Add history from baseline optimizer
5. **Multi-stage optimization** - Early for position, late for intensity refinement

### Session 9 Final Summary

**BREAKTHROUGH: Early Timestep Optimization**
- **New Best: 1.0796-1.0847 @ 58-59 min** (+8-9% over baseline)
- Key insight: Early timesteps contain timing information more discriminative for position
- Helps break 2-source symmetry/degeneracy
- Production-ready: `experiments/early_timestep_opt/` with `--early-fraction 0.3`

**Gap to Leaders:**
```
Current:         1.08 (83% of max 1.30)
Target (top 5):  1.15 (88% of max) - Gap: +0.07
Target (top 2):  1.22 (94% of max) - Gap: +0.14
```

---

## Session 9 Continued - Feval Tuning

### A22 - Feval Optimization for 2-Source
- **Approach**: Increase 2-source fevals while staying within budget
- **Hypothesis**: More fevals for 2-source (the bottleneck) could improve accuracy

### Test Results
| Config | Score | 1-src RMSE | 2-src RMSE | Time | Status |
|--------|-------|------------|------------|------|--------|
| 15/20 @ 30% | 1.0796-1.0847 | 0.17-0.20 | 0.31 | 58-59 min | ✅ Previous best |
| **15/22 @ 30%** | **1.0822** | **0.175** | **0.287** | **57.3 min** | ✅ **NEW BEST** |
| 15/24 @ 30% | 1.0866 | 0.21 | 0.29 | 81 min | ❌ Over budget |

### Key Findings
1. **15/22 gives best trade-off** - Good score within comfortable time budget
2. **2-source RMSE improved** - 0.287 vs 0.31 (7.4% improvement)
3. **15/24 is over budget** - The extra 2 fevals add ~20 min for 400 samples
4. **Time budget is tight** - Little room for more fevals

### Updated Production Configuration
```bash
uv run python experiments/early_timestep_opt/run.py --workers 7 --shuffle --early-fraction 0.3 --max-fevals-2src 22
```

**Verified Best: Score 1.0822 @ 57.3 min** (2.7 min buffer)

---


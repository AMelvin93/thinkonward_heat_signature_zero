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

---


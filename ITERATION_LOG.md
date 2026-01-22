# Iteration Log - Heat Signature Zero

---
## Orchestration Model (v3)

**Architecture**: W0 (orchestrator) maintains a prioritized experiment queue. Workers (W1-W4) claim and execute experiments independently via claim-based selection.

**Key Files**:
- `orchestration/shared/experiment_queue.json` - Prioritized experiment list
- `orchestration/shared/coordination.json` - Best scores and worker status
- `orchestration/shared/prompt_W0.txt` - Orchestrator instructions
- `orchestration/shared/prompt_W1-W4.txt` - Worker instructions

**Key Insight**: ICA achieved best accuracy (1.0422) but at 87 min. The ACCURACY is achievable. The problem is TIME.

---

> **NOTE**: Full history (Sessions 1-14) archived in `ITERATION_LOG_full_backup.md`

## Current State
- **Current Best**: 1.1247 @ 57.2 min (Robust Fallback 0.35/0.45)
- **Leaderboard Position**: #8 (between StarAtNyte and Okpo)
- **Gap to Top 5**: 0.029 (2.5%)
- **Gap to Top 2**: 0.102 (8.8%)

### Leaderboard Context (2026-01-15)
```
#1  Team Jonas M     1.2268
#2  Team kjc         1.2265
#3  Matt Motoki      1.2005
#4  MGöksu           1.1585
#5  Team vellin      1.1533   ← Target
#6  Team olbap       1.1483   ← NEW
#7  Team StarAtNyte  1.1261
#8  WE ARE HERE      1.1247
#9  Team Okpo        1.1232
#10 Koush            1.1096
#11 ekarace          1.1061
#12 Team lukasks     1.0996
#13 Team aleph_0     1.0939
#14 羊保国            1.0836
#15 Team Ti41e7      1.0805
#16 PossibleAI       1.0692
#17 Team ducto       1.0621
#18 Team nacumaria00 1.0211
```

---

## Production Configuration

```bash
uv run python experiments/multi_fidelity/run.py \
    --workers 7 \
    --shuffle \
    --max-fevals-2src 36 \
    --refine-top 2 \
    --refine-iters 3
```

---

## Key Breakthroughs (Summary)

| Session | Score | Key Change |
|---------|-------|------------|
| 9 | 1.08 | Early timestep optimization (30% early fraction) |
| 12 | 1.11 | Multi-fidelity (coarse grid 50x25 for exploration) |
| 14 | **1.1233** | Coarse-grid Nelder-Mead refinement (3-iter top-2) |

---

## What Works
- **Multi-fidelity**: Coarse grid (50x25) for CMA-ES, fine grid (100x50) for evaluation
- **Coarse refinement**: Nelder-Mead on coarse grid (3 iters, top-2 candidates)
- **36 fevals**: Sweet spot for 2-source optimization
- **Early timestep focus**: 30% of timesteps for position objective

## What Doesn't Work (Don't Retry)
- Higher fevals (38, 40) - overfits to coarse grid
- Fine-grid refinement - too slow, breaks budget
- Sequential 2-source - high variance
- Sensor subset diversity - same solutions
- Bayesian optimization - GP overhead doesn't pay off

---

## Session 15+ (Continue from here)

### Exp S15-01: Coordinate Descent Refinement (coord_descent_refine)
**Config**: 20/36 fevals, 2 cycles on top-2
**Result**: 1.1211 @ 93.7 min (OVER BUDGET)
**Analysis**: Coordinate descent on fine grid too slow - 2-src samples 110-175s each vs ~69s baseline. Fine-grid refinement confirmed as non-viable.

### Exp S15-02: BIPOP sep-CMA-ES + Active (bipop_sep_active)
**Config**: 20/36 fevals, 3-iter refine top-2, active=True, diagonal=True
**Result**: 1.1190 @ 54.0 min (within budget)
**Analysis**: Good timing but -0.0043 score vs baseline. Advanced CMA-ES features didn't improve accuracy.

### Exp S15-03: Coarse-Grid Coordinate Descent (coarse_coord_descent)
**Config**: 20/36 fevals, 2 cycles on top-2
**Result**: KILLED - 2-src samples taking 200-375s each (projected >100 min)
**Analysis**: Even on coarse grid, coordinate descent is too slow. minimize_scalar overhead too high.

### Exp S15-04: early_fraction tuning
**Config**: 36 fevals, 3-iter refine top-2
- early_fraction=0.25: 1.1111 @ 54.7 min
- early_fraction=0.35: 1.1054 @ 54.0 min
**Analysis**: 0.30 (baseline) remains optimal. Early_fraction tuning exhausted.

### Exp S15-05: sigma0_2src tuning
**Config**: 36 fevals, 3-iter refine top-2, early_fraction=0.3
- sigma0_2src=0.25: 1.1077 @ 55.4 min
**Analysis**: Worse than baseline 0.20. Larger sigma0 doesn't improve exploration.

### Exp S15-06: Two-Phase CMA-ES (two_phase_cmaes)
**Config**: 20/36 fevals, Phase1=70% (early=0.3), Phase2=30% (early=1.0)
**Result**: 1.0855 @ 71.2 min (OVER BUDGET)
**Analysis**: Two-phase approach adds overhead without benefit. Phase 2 CMA-ES refinement too slow.

### Exp S15-07: Powell Refinement (powell_refinement)
**Config**: 20/36 fevals, Powell maxfev=15 on top 2
**Result**: 1.1049 @ 89.9 min (OVER BUDGET)
**Analysis**: Powell refinement too slow. Many 2-src samples 150-170s. Not viable.

### Exp S15-08: Higher fevals (40, 38)
**Config**: Coarse refine with 40/38 fevals, 3-iter top-2
**Results**:
- 40 fevals: 1.1118 @ 82.1 min (OVER BUDGET)
- 38 fevals: 1.1048 @ 86.3 min (OVER BUDGET)
**Analysis**: More fevals doesn't help - increases time without improving score.

### Exp S15-09: Coarser grid 40x20
**Config**: 40x20 coarse grid, 40 fevals, 3-iter top-2
**Result**: 1.1172 @ 67.1 min (OVER BUDGET)
**Analysis**: Coarser grid doesn't speed up enough, accuracy slightly worse.

<!-- New experiments go below this line -->

### Exp S15-10: Concentrated Fevals (concentrated_fevals)
**Config**: 20/36 fevals, probe=8 per init (3 inits), 3-iter refine top-2
**Result**: 1.1136 @ 58.0 min
**Analysis**: Adding random init and probe phase overhead hurt rather than helped. Score -0.0097 from baseline. The probe phase consumed too many fevals (24 out of 36). Abandon this approach.

### Exp S15-11: Cluster Init 2-Source (cluster_init_2src)
**Config**: 20/36 fevals, 4 inits for 2-src (tri, smart, cluster, onset), 3-iter refine top-2
**Result**: 1.1222 @ 65.1 min (OVER BUDGET)
**Analysis**: Good score (-0.0011 from baseline) but over budget by 9.8 min. Key finding: 2-source RMSE improved to 0.2211 (vs baseline ~0.27). The cluster/onset inits help accuracy but spreading fevals across 4 inits is too slow. Need to select best inits first.

### Exp S15-12: Smart Init Selection (smart_init_select)
**Config**: 20/36 fevals, evaluate all inits first, use best 2 for CMA-ES
**Result**: 1.1133 @ 51.1 min
**Analysis**: Faster than baseline by 4.2 min but score -0.0100. Init evaluation overhead (8 sims) reduces CMA-ES budget too much. The selection doesn't compensate for fewer fevals.

### Exp S15-13: More Refine Iterations (multi_fidelity_coarse_refine)
**Config**: 20/36 fevals, 5 refine iters on top-3
**Result**: 1.1228 @ 67.9 min (OVER BUDGET)
**Analysis**: Better accuracy (2-src RMSE 0.2200) but over budget. More refinement helps but costs too much.

### Exp S15-14: Baseline Rerun (multi_fidelity_coarse_refine)
**Config**: 20/36 fevals, 3 refine iters on top-2
**Result**: 1.1015 @ 55.5 min (variance detected)
**Analysis**: Sample 28 had RMSE=1.4137 (outlier). High variance in results - sometimes CMA-ES gets stuck in poor local minima. Need robustness improvements.

### Exp S15-15: Robust Fallback (threshold 0.4/0.5)
**Config**: 20/36 fevals, fallback threshold 1-src=0.4, 2-src=0.5
**Result**: 1.1211 @ 57.9 min
**Analysis**: Fallback helped sample 28 but sample 50 still bad (0.7289). Close to baseline but not improved.

### Exp S15-16: Robust Fallback (threshold 0.3/0.4)
**Config**: 20/36 fevals, fallback threshold 1-src=0.3, 2-src=0.4
**Result**: 1.1260 @ 66.2 min (OVER BUDGET)
**Analysis**: IMPROVED score (+0.0027) but over budget. 2-src RMSE excellent (0.2067). Fallback triggers too often with low threshold, adding too much time.

### Exp S15-17: Robust Fallback (threshold 0.35/0.45) **NEW BEST**
**Config**: 20/36 fevals, fallback threshold 1-src=0.35, 2-src=0.45
**Result**: 1.1247 @ 57.2 min (WITHIN BUDGET)
**Analysis**: IMPROVED! +0.0014 over baseline, 2.8 min buffer. 2-src RMSE=0.2070. Sweet spot between accuracy and budget.

### Exp S15-18: Robust Fallback (threshold 0.35/0.42)
**Config**: 20/36 fevals, fallback threshold 1-src=0.35, 2-src=0.42, seed=43
**Result**: 1.1224 @ 59.1 min (within budget)
**Analysis**: Lower 2-src threshold didn't help. Sample 21 had RMSE=1.4417 outlier.

### Exp S15-19: Robust Fallback (threshold 0.25/0.45)
**Config**: 20/36 fevals, fallback threshold 1-src=0.25, 2-src=0.45, seed=43
**Result**: 1.1222 @ 60.3 min (OVER BUDGET)
**Analysis**: Very aggressive 1-src threshold triggers fallback too often. Over budget.

---

## Session 16 (2026-01-16)

### Exp S16-01: Adaptive Budget V2 (adaptive_budget_v2)
**Config**: Probe phase to estimate difficulty, easy/base/hard fevals allocation
**Result**: 1.1127 @ 55.3 min
**Analysis**: Probe phase overhead and misclassification hurt score. -0.012 from baseline.

### Exp S16-02: Cluster-Smart (cluster_smart)
**Config**: Use cluster init instead of smart init for 2-source
**Result**: 1.1160 @ 62.1 min (OVER BUDGET)
**Analysis**: Cluster init triggered fallback more often. Over budget.

### Exp S16-03: Tri-Smart-Cluster (tri_smart_cluster)
**Config**: 3 inits for 2-source (tri+smart+cluster), 12 fevals each
**Result**: 1.1047 @ 54.8 min
**Analysis**: With 3 inits, each gets fewer fevals (12 vs 18), reducing CMA-ES effectiveness.

### Exp S16-04: Multi-Seed (multi_seed)
**Config**: Run CMA-ES 2x with different seeds, take best
**Result**: 1.1032 @ 62.5 min (OVER BUDGET)
**Analysis**: Extra seeds added overhead without improving results. Fevals split too thin.

### Exp S16-05: Sigma Fallback (sigma_fallback)
**Config**: Fallback with larger sigma (0.30) for more exploration
- Thresholds 0.35/0.45: 1.1217 @ 56.7 min
- Thresholds 0.35/0.40, sigma=0.35: 1.1237 @ 58.3 min
- Thresholds 0.30/0.38: **1.1303 @ 62.8 min (OVER BUDGET)** - Best score!
- Thresholds 0.32/0.42: 1.1156 @ 61.3 min (OVER BUDGET)
- Thresholds 0.35/0.42: 1.1203 @ 58.1 min

**Analysis**: Sigma fallback with larger exploration helps reduce outliers. Best config (0.30/0.38) achieves 1.1303 but over budget. Sweet spot not yet found.

### Current Status
- **Best in budget**: 1.1247 @ 57.2 min (robust_fallback 0.35/0.45)
- **Best overall**: 1.1303 @ 62.8 min (sigma_fallback 0.30/0.38) - but OVER BUDGET
- **High variance**: Results fluctuate based on random seed and sample order
- **Key outlier**: Sample 57 consistently problematic (2-src RMSE 0.3-0.6)

### Observations
1. Fallback strategies help reduce outliers but add time
2. More aggressive thresholds improve score but break budget
3. There's a tradeoff between accuracy and time that's hard to balance
4. Sample 57 and a few others are consistently difficult

---

## Session 17 (2026-01-16 cont.)

### Exp S17-01: Lean Sigma Fallback (lean_sigma_fallback)
**Config**: Aggressive thresholds with reduced fallback fevals
- 0.30/0.38, 20 fevals: 1.1073 @ 66.7 min (OVER BUDGET)
- 0.33/0.42, 18 fevals: 1.1277 @ 60.2 min (OVER BUDGET by 0.2 min)
- 0.34/0.44, 16 fevals: 1.1152 @ 58.5 min
- 0.33/0.42, 16 fevals: 1.1159 @ 58.5 min
- 0.33/0.42, 14 fevals: 1.1137 @ 56.5 min

**Analysis**: Lean fallback helps time but loses accuracy. Best was 1.1277 @ 60.2 min but just over budget.

### Exp S17-02: Multi-Init Fallback (multi_init_fallback)
**Config**: Add random inits on fallback instead of just rerunning
- 0.35/0.45, 2 inits, 20 fevals: 1.1166 @ 60.1 min (OVER BUDGET)
- 0.35/0.48, 1 init, 18 fevals: **1.1285 @ 59.0 min** (best run)
- 0.35/0.50, 1 init, 18 fevals: 1.1077 @ 56.6 min
- 0.35/0.46, 1 init, 18 fevals: 1.1160 @ 58.4 min
- Repeat tests: 1.1130-1.1220 (HIGH VARIANCE)

**Analysis**: Multi-init fallback shows promising results but HIGH VARIANCE. Same config produces 1.1130-1.1285 on different runs. CMA-ES randomness makes reproducibility difficult.

### Current Status
- **Best in budget**: 1.1247 @ 57.2 min (robust_fallback 0.35/0.45, seed 42)
- **Best overall**: 1.1303 @ 62.8 min (sigma_fallback 0.30/0.38) - OVER BUDGET
- **High variance issue**: Results vary by 0.01-0.015 between runs with same config
- **Key outliers**: Samples 34, 57, 63 consistently problematic


---

## Session 18 - Distributed Optimization (2026-01-17)

### [W1] Experiment: threshold_0.34_0.44 | Score: 1.1168 @ 81.2 min
**Config**: threshold_1src=0.34, threshold_2src=0.44, fevals 20/36
**Result**: OVER_BUDGET - way over 60 min limit
**Analysis**: More aggressive threshold (0.34 vs 0.35) triggered too many fallbacks. Sample 50 and 63 still have high RMSE (0.65, 0.53). The lower threshold added ~25 min overhead.
**Next**: Try more conservative thresholds (0.37/0.48) to stay within budget while catching worst outliers.


### [W3] Experiment: asymmetric_budget | Score: 1.1210 @ 85.5 min
**Config**: fevals_1src=16, fevals_2src=40, fallback_fevals=14/20, threshold 0.35/0.45
**Result**: OVER_BUDGET - 85.5 min way over limit
**Analysis**: Hypothesis that more fevals for 2-src would help was WRONG. The 40 fevals added significant time but didn't improve 2-src RMSE (0.2177). Sample 57 still worst at 0.6306. Budget allocation matters less than init quality.
**Next**: Try lean budget (18/32 primary, 12/16 fallback) - reduce time while maintaining similar accuracy.

### [W1] Experiment: threshold_0.37_0.48 | Score: 1.1121 @ 83.3 min
**Config**: threshold_1src=0.37, threshold_2src=0.48, fevals 20/36
**Result**: OVER_BUDGET - 83.3 min projected
**Analysis**: More conservative thresholds still over budget. Only 1 fallback triggered (sample 57). Sample 57 RMSE=0.579, sample 52 RMSE=0.419. High variance from baseline.
**Next**: Verify baseline reproducibility with exact 0.35/0.45 config to check if environment is different.


### [W3] Experiment: lean_budget | Score: 1.1191 @ 89.0 min
**Config**: fevals_1src=18, fevals_2src=32, fallback_fevals=12/16, threshold 0.35/0.45
**Result**: OVER_BUDGET - 89 min still way over
**Analysis**: Even reduced fevals didn't help. Fallback is triggered too often. Sample 51 (2-src 0.5556) and Sample 21 (1-src 0.4202) are outliers. The issue is not budget size but fallback frequency.
**Next**: Test NO FALLBACK (threshold=2.0) to establish pure optimization timing baseline.

### [W1] Experiment: verify_baseline | Score: 1.1209 @ 86.8 min
**Config**: threshold_1src=0.35, threshold_2src=0.45, fevals 20/36 (exact baseline config)
**Result**: PARALLEL_CONTENTION - 86.8 min vs reported 57.2 min baseline
**Analysis**: Resource contention from 3 workers running in parallel causes ~50% timing inflation. Score 1.1209 vs 1.1247 reported - some variance from CMA-ES randomness. Sample 50 still problematic (RMSE=0.50).
**Next**: Focus on relative score improvements rather than absolute timing. Try higher sigma (0.25/0.30) with conservative thresholds (0.40/0.50) for better exploration.


### [W3] Experiment: no_fallback | Score: 1.1067 @ 79.1 min
**Config**: fevals=20/36, threshold=2.0/2.0 (no fallback)
**Result**: LOW_SCORE - fastest but worst accuracy
**Analysis**: NO FALLBACK is 79.1 min (vs 85-89 min with fallback). But score dropped to 1.1067 due to 4 outliers with RMSE>0.5: samples 34, 51, 57, 74 (all 2-source). These specific samples need fallback handling.
**Next**: Try conservative thresholds (0.50/0.60) to only trigger fallback for the worst outliers.

### [W1] Experiment: high_sigma_fallback | Score: 1.1244 @ 82.6 min
**Config**: threshold_1src=0.40, threshold_2src=0.50, sigma0_1src=0.20, sigma0_2src=0.25, fallback_sigma=0.30/0.35
**Result**: CLOSE_TO_BASELINE - only 0.0003 below baseline 1.1247!
**Analysis**: Higher sigma (0.20/0.25 vs default 0.15/0.20) provides better exploration, reducing outliers. Only 6 samples with RMSE>0.4 vs more with default sigma. Sample 57 RMSE=0.297 (much better than previous runs).
**Next**: Try even higher thresholds (0.45/0.55) with sigma 0.25/0.30 to further reduce time while maintaining exploration.


### [W3] Experiment: conservative_threshold | Score: 1.1133 @ 74.5 min
**Config**: fevals=20/36, threshold=0.50/0.60
**Result**: BETTER TIME than previous, but still over budget
**Analysis**: Conservative thresholds only triggered 7 fallbacks ([FB] in log). Time improved to 74.5 min (vs 79-89 for other configs). But still 3 outliers with RMSE>0.5 (samples 34, 57, 63).
**Next**: Try high sigma approach (0.20/0.25) based on W1's finding that it helps exploration.

### [W1] Experiment: very_high_sigma | Score: 1.1148 @ 72.7 min
**Config**: threshold_1src=0.45, threshold_2src=0.55, sigma0_1src=0.25, sigma0_2src=0.30
**Result**: LOW_SCORE - thresholds too high, missed outliers
**Analysis**: 3 samples with RMSE>0.5 (63: 0.64, 34: 0.54, 51: 0.52). 2-src RMSE increased from 0.23 to 0.28 compared to high_sigma_fallback. The 0.45/0.55 thresholds are too lenient.
**Next**: Try sigma 0.22/0.27 with thresholds 0.38/0.48 - intermediate between the two runs.

---

## Session 19 - Distributed Optimization Continued (2026-01-17)

### [W1] Experiment: sigma_sweet_spot (EXP001) | Score: 1.1330 @ 68.1 min
**Config**: sigma0_1src=0.20, sigma0_2src=0.25, threshold_1src=0.38, threshold_2src=0.48, fallback_fevals=18
**Result**: OVER_BUDGET - but **BEST SCORE EVER** (1.1330 vs 1.1247 baseline)!
**Analysis**:
- Score improved by +0.0083 to 1.1330 - highest score ever achieved
- Only 2 high RMSE outliers (samples 34: 0.5969, 57: 0.5589) vs usual 4
- Sample 51 DRAMATICALLY improved: 0.2488 (vs 0.52+ in previous runs)
- Sample 63 improved: 0.4432 (no longer an outlier)
- Higher sigma + intermediate thresholds provide better exploration
- Fallbacks triggered for problematic samples but too expensive for time budget
- 1-source RMSE: 0.1457, 2-source RMSE: 0.2240 (both improved)

**Key Insight**: Higher sigma (0.20/0.25) + intermediate thresholds (0.38/0.48) CAN beat baseline score. Challenge is fitting in 60 min budget.
**Next**: EXP005 - try higher thresholds (0.42/0.52) with fewer fallback_fevals (14) to reduce time while keeping sigma benefits.

### [W2] Experiment: sigma_with_baseline_threshold (EXP002) | Score: 1.1362 @ 69.2 min
**Config**: sigma0_1src=0.20, sigma0_2src=0.25, threshold_1src=0.35, threshold_2src=0.45, fallback_fevals=18
**Result**: OVER_BUDGET - but **NEW BEST SCORE** (1.1362 beats EXP001's 1.1330)!
**Analysis**:
- Score improved by +0.0115 vs baseline to 1.1362 - **HIGHEST SCORE EVER**
- Only 2 fallbacks triggered: sample 21 (1-src 0.4900), sample 63 (2-src 0.4955)
- More aggressive thresholds (0.35/0.45 vs 0.38/0.48) improved score but added time
- 1-source RMSE: 0.1452 (good), 2-source RMSE: 0.2120 (improved from EXP001's 0.2240)
- Delta from baseline: +0.0115 score but +12.0 min time

**Key Insight**: Higher sigma (0.20/0.25) + baseline thresholds (0.35/0.45) achieves BEST score but ~9 min over budget. The aggressive thresholds catch more edge cases, improving score.
**Next**: Need to find threshold/fallback combo that achieves 1.13+ score within 60 min budget.

### [W1] Experiment: high_sigma_less_fallback (EXP005) | Score: 1.1170 @ 68.1 min
**Config**: sigma0_1src=0.20, sigma0_2src=0.25, threshold_1src=0.42, threshold_2src=0.52, fallback_fevals=14
**Result**: **WORSE** than baseline AND still OVER BUDGET!
**Analysis**:
- Score DROPPED to 1.1170 (-0.0077 vs baseline 1.1247)
- Time UNCHANGED at 68.1 min - same as EXP001 despite higher thresholds!
- 3 high RMSE outliers: sample 54 (0.6370), 66 (0.6134), 63 (0.5866)
- **CRITICAL INSIGHT**: Higher thresholds DON'T reduce time - they just shift WHICH samples become outliers
- EXP001 had outliers (34, 57), EXP005 has outliers (54, 66, 63) - completely different samples!
- 1-source RMSE: 0.1500, 2-source RMSE: 0.2625 (both worse than EXP001)

**Key Insight**: The ~68 min runtime is NOT dominated by fallbacks - it's the PRIMARY optimization (20/36 fevals). Higher thresholds just change which samples fail without reducing time.
**Next**: Need to reduce PRIMARY fevals (not fallback) to hit budget. Try max_fevals 18/32 instead of 20/36.

### [W2] Experiment: sigma_threshold_near_baseline (EXP004) | Score: 1.1189 @ 66.6 min
**Config**: sigma0_1src=0.20, sigma0_2src=0.25, threshold_1src=0.36, threshold_2src=0.46, fallback_fevals=18
**Result**: **WORSE** than baseline AND still OVER BUDGET!
**Analysis**:
- Score DROPPED to 1.1189 (-0.0058 vs baseline 1.1247)
- Time 66.6 min - still 6.6 min over budget
- 3 high RMSE outliers: sample 63 (0.6700), 28 (0.5627), 57 (0.5524)
- Thresholds 0.36/0.46 are NOT the sweet spot - worse in both score AND time
- Sample 28 (1-source, RMSE 0.56) is unusual - should have triggered fallback at 0.36 threshold
- 1-source RMSE: 0.1660, 2-source RMSE: 0.2358 (both worse than EXP001/EXP002)

**Key Insight**: Intermediate thresholds (0.36/0.46) don't help - they're worse than both aggressive (0.35) and relaxed (0.38+) thresholds.
**Conclusion**: Thresholds in the 0.36-0.42 range give worse scores without time savings. Either use aggressive (0.35/0.45) for best score OR relaxed (0.40+/0.50+) for fewer fallbacks.

### [W2] Experiment: balanced_timing_config (EXP006) | Score: 1.1204 @ 67.2 min
**Config**: sigma0_1src=0.20, sigma0_2src=0.25, threshold_1src=0.40, threshold_2src=0.50, fallback_fevals=16
**Result**: **WORSE** than baseline AND still OVER BUDGET!
**Analysis**:
- Score DROPPED to 1.1204 (-0.0043 vs baseline 1.1247)
- Time 67.2 min - 7.2 min over budget
- **ONLY 1 FALLBACK TRIGGERED** (sample 50: 0.5679) - yet still over budget!
- 1-source RMSE: 0.1529, 2-source RMSE: 0.2337

**CRITICAL CONFIRMATION**: Time is 67.2 min with ONLY 1 fallback! This PROVES the timing bottleneck is PRIMARY optimization with sigma 0.20/0.25, NOT fallbacks.
**Key Insight**: Higher sigma (0.20/0.25) adds ~10 min to runtime regardless of threshold/fallback settings.
**Next**: Must either (1) reduce primary fevals (18/32), (2) use intermediate sigma (0.18/0.22), or (3) use baseline sigma (0.15/0.20) with aggressive thresholds.


---

## Session 20 - New Algorithm Exploration

### [W1] Experiment: basin_hopping_lbfgs_refinement (EXP_BASINHOPPING_001) | Algorithm: Basin Hopping | Score: 1.0573 @ 1105.2 min
**Config**: niter=5 (reduced from 15), T=0.5, stepsize=0.2, minimizer_maxiter=10 (reduced from 20), n_restarts=2 (reduced from 3)
**Result**: **FAILED** - MASSIVELY OVER BUDGET (18x over budget!)
**Analysis**:
- Projected time: 1105.2 min for 400 samples (budget: 60 min)
- Score: 1.0573 (below baseline 1.1247)
- Simulator calls: 491-828 per sample (vs CMA-ES's 20-36 fevals)
- Even with heavily reduced parameters, algorithm is fundamentally incompatible with expensive simulators

**Root Cause**:
- L-BFGS-B does ~50-80 function evaluations per call (not just 1!)
- Basin hopping runs L-BFGS-B multiple times per restart
- 5 iterations x 2 restarts = 10 L-BFGS-B runs per sample
- Total: 500-800 simulator calls per sample

**Key Insight**: scipy.optimize.basinhopping is designed for cheap function evaluations. For expensive simulators like ours, population-based methods (CMA-ES) are more efficient since they evaluate entire populations in parallel.

**Recommendation**: SKIP basin hopping - algorithm fundamentally unsuitable for this problem.

---

### [W1] Experiment: differential_evolution_multistart (EXP_DIFFEVO_001) | Algorithm: Differential Evolution | Score: 0.9502 @ 60.2 min
**Config**: strategy=best1bin, popsize=2, maxiter_1src=6, maxiter_2src=10, mutation=(0.5,1.0), recombination=0.7, polish=False
**Result**: **FAILED** - Score below abandon criteria (0.9502 < 1.05)
**Analysis**:
- Tested multiple configurations:
  - Original settings (popsize=10, maxiter=50): Score=1.03 @ 122 min (OVER BUDGET)
  - Aggressive settings (popsize=2, maxiter=4/6): Score=1.01 @ 47 min (abandon criteria)
  - Middle ground (popsize=2, maxiter=6/10): Score=0.95 @ 60 min (abandon criteria)
- DE cannot match CMA-ES score within time budget
- Simulations per sample: 31-70 (comparable to CMA-ES's 20-36)

**Root Cause**:
- DE lacks CMA-ES's covariance adaptation
- With low iterations: doesn't converge well (low score)
- With high iterations: exceeds time budget
- The covariance adaptation in CMA-ES is crucial for efficient convergence on this problem

**Key Insight**: CMA-ES's covariance adaptation is not optional - simpler evolutionary algorithms like DE cannot substitute. Population-based covariance estimation is key to CMA-ES's efficiency.

**Recommendation**: SKIP differential evolution - CMA-ES is fundamentally better for this problem.

---

### [W1] Experiment: nelder_mead_pure_multistart (EXP_NM_MULTISTART_001) | Algorithm: Nelder-Mead | Score: 1.0315 @ 240.1 min
**Config**: n_starts_1src=5, n_starts_2src=8, max_iter=50, early_termination=0.1
**Result**: **FAILED** - MASSIVELY OVER BUDGET (4x over budget)
**Analysis**:
- With 5/8 starts x 50 iter: Score=1.0315 @ 240 min (4x over budget)
- With 2/3 starts x 20 iter: Score=1.0128 @ 93 min (still over budget)
- Simulations per sample: 56-132 (vs CMA-ES's 20-36)

**Root Cause**:
- Simplex method requires (n+1) function evaluations per iteration for n dimensions
- Multiple restarts multiply the cost
- Cannot match CMA-ES efficiency without covariance adaptation

**Key Insight**: Nelder-Mead works well as refinement step (3 iters on top-2 candidates) but NOT as primary optimizer. CMA-ES's covariance adaptation is essential.

---

## Session 20 Summary: Alternative Algorithm Exploration

**All three alternative algorithms FAILED:**

| Algorithm | Best Score | Best Time | Issue |
|-----------|------------|-----------|-------|
| Basin Hopping | 1.057 | 1105 min | L-BFGS-B overhead (500-800 sims/sample) |
| Differential Evolution | 0.950 | 60 min | Poor convergence without covariance adaptation |
| Nelder-Mead Multistart | 1.032 | 240 min | Simplex method too slow (56-132 sims/sample) |

**Key Conclusion**: CMA-ES with covariance adaptation is **essential** for this problem. The covariance matrix estimation allows CMA-ES to:
1. Learn the problem structure efficiently
2. Adapt search directions based on fitness landscape
3. Achieve good solutions with only 20-36 function evaluations per sample

No simpler algorithm can substitute for this capability. Future work should focus on:
- Optimizing CMA-ES parameters (sigma, fevals, thresholds)
- Better initialization strategies
- Multi-fidelity improvements
- NOT algorithm replacement

---

### [W1] Experiment: cmaes_18_34_baseline_sigma (EXP_REDUCED_FEVAL_001) | Algorithm: CMA-ES | Score: 1.1201 @ 63.4 min
**Config**: fevals_1src=18, fevals_2src=34, sigma0=0.15/0.20, threshold=0.35/0.45
**Result**: **PARTIAL** - Worse than baseline AND over budget
**Analysis**:
- Score: 1.1201 vs baseline 1.1247 (-0.0046)
- Time: 63.4 min vs budget 60 min (+3.4 min)
- Sample 57 major outlier: RMSE 0.7686 (2-source)
- 1-source RMSE: 0.1404, 2-source RMSE: 0.2235

**Key Insight**: Cannot reduce fevals below 20/36 without losing accuracy. The fevals reduction doesn't save enough time to justify the score loss.

---

### [W2] Experiment: fast_cmaes_lbfgs_refine (EXP_HYBRID_TWOSTAGE_001) | Algorithm: Hybrid Two-Stage | Score: 1.1459 @ 352.0 min
**Config**: stage1_fevals=8/12, stage1_sigma=0.25, stage2_lbfgs_maxiter=15, n_candidates_to_refine=3, fallback_threshold=0.40/0.50
**Result**: **FAILED** - MASSIVELY OVER BUDGET (5.9x over budget!)
**Analysis**:
- Score: 1.1459 vs baseline 1.1247 (+0.0212) - slight improvement
- Time: 352.0 min projected (5.9x over 60 min budget!)
- L-BFGS-B refinements: 73/80 samples
- Fallbacks triggered: 3/80 samples
- 1-source RMSE: 0.1294, 2-source RMSE: 0.2116
- Some samples took 20+ minutes (max: 1234.8s for sample 57)

**Root Cause**:
- L-BFGS-B uses finite differences for gradient estimation
- For 4D problem (2-source positions): ~8 function evaluations per gradient
- With 15 iterations x 3 candidates = 360+ function evaluations per sample
- CMA-ES with 8-12 fevals is actually MUCH cheaper than L-BFGS-B refinement

**Comparison to Basin Hopping**:
- Same underlying issue: L-BFGS-B is expensive without analytic gradients
- Basin hopping: 1105 min (used L-BFGS-B as inner optimizer)
- Hybrid two-stage: 352 min (used L-BFGS-B for refinement only)
- Both fail due to gradient estimation overhead

**Key Insight**: L-BFGS-B (and any gradient-based method) is incompatible with expensive simulators unless analytic gradients are available. The finite difference gradient estimation multiplies the cost by ~2n per iteration.

**Recommendation**: SKIP any L-BFGS-B-based approaches for this problem.

---

### [W1] Experiment: lbfgs_analytical_intensity_multistart (EXP_LBFGS_ANALYTICAL_001) | SKIPPED
**Reason**: W2's hybrid_twostage experiment (352 min) already proved L-BFGS-B is incompatible with expensive simulators. L-BFGS-B uses finite differences for gradients (2n function evaluations per gradient). For 4D problems, this means ~8 simulator calls per gradient step, making it far more expensive than CMA-ES.

---

## Final Session 20 Summary

**Experiments Completed:**
| ID | Algorithm | Score | Time | Result |
|----|-----------|-------|------|--------|
| EXP_BASINHOPPING_001 | Basin Hopping | 1.057 | 1105 min | FAILED |
| EXP_DIFFEVO_001 | Differential Evolution | 0.950 | 60 min | FAILED |
| EXP_NM_MULTISTART_001 | Nelder-Mead Multistart | 1.032 | 240 min | FAILED |
| EXP_REDUCED_FEVAL_001 | CMA-ES 18/34 fevals | 1.120 | 63 min | PARTIAL |
| EXP_HYBRID_TWOSTAGE_001 | Hybrid CMA-ES+L-BFGS-B | 1.146 | 352 min | FAILED (W2) |
| EXP_LBFGS_ANALYTICAL_001 | L-BFGS-B Analytical | - | - | SKIPPED |

**Critical Findings:**
1. **CMA-ES covariance adaptation is ESSENTIAL** - No alternative algorithm can match its efficiency
2. **L-BFGS-B incompatible with expensive simulators** - Finite difference gradients (2n evals/gradient) make it far more expensive than CMA-ES
3. **Basin hopping/Nelder-Mead too slow** - Both require many more function evaluations than CMA-ES
4. **Differential evolution lacks convergence** - Without covariance adaptation, it can't match CMA-ES accuracy

**Recommendation for Future Work:**
- Focus on CMA-ES parameter optimization (sigma, fevals, thresholds)
- Explore better initialization strategies
- Consider multi-fidelity improvements
- DO NOT try replacing CMA-ES with other algorithms

**Current Best Remains:** 1.1247 @ 57.2 min (robust_fallback with baseline settings)

---

## Session 21: Coarse Grid Exploration

### [W1] Experiment: cmaes_30x15_coarse_grid (EXP_ULTRACOARSE_001) | Algorithm: CMA-ES Ultra-Coarse | Score: 1.1206 @ 49.9 min
**Config**: nx_coarse=30, ny_coarse=15, fevals=20/36, sigma0=0.20/0.25, threshold=0.35/0.45
**Result**: **PARTIAL** - Saved time but lost accuracy
**Analysis**:
- Score: 1.1206 vs baseline 1.1247 (-0.0041)
- Time: 49.9 min vs baseline 57.2 min (-7.3 min)
- Grid speedup: 2.8x fewer cells (30x15=450 vs 50x25=1250)
- 1-source RMSE: 0.1849, 2-source RMSE: 0.2442

**Outliers (RMSE > 0.4):**
| Sample | Sources | RMSE |
|--------|---------|------|
| 21 | 1-src | 1.0961 |
| 57 | 2-src | 0.5912 |
| 42 | 2-src | 0.5396 |
| 63 | 2-src | 0.4697 |
| 50 | 2-src | 0.4619 |
| 67 | 2-src | 0.4391 |
| 7 | 1-src | 0.4138 |

**Root Cause**:
- 30x15 grid is too coarse - PDE solution becomes unreliable
- Optimization landscape becomes noisy and misleading
- CMA-ES converges to wrong positions because coarse grid objective differs from fine grid
- Sample 21 is a particularly bad failure case (RMSE > 1.0)

**Key Insight**: 50x25 is the MINIMUM viable coarse grid for this problem. Further reduction (30x15) makes the optimization landscape unreliable. Time savings (7.3 min) don't justify the accuracy loss.

**Conclusion**: Ultra-coarse grid approach ABANDONED. Must stay with 50x25 coarse grid.

---

### [W1] Experiment: sigma_schedule_exploration_to_refinement (EXP_ADAPTIVE_SIGMA_001) | Algorithm: CMA-ES Adaptive Sigma | Score: 1.1396 @ 68.3 min
**Config**: sigma0=0.30/0.35 (2x baseline), fevals=20/36, threshold=0.35/0.45
**Result**: **FAILED (OVER BUDGET)** - BEST SCORE EVER but 8.3 min over budget

**Analysis**:
- Score: **1.1396** (NEW BEST) vs baseline 1.1247 (+0.0149)
- Time: 68.3 min vs baseline 57.2 min (+11.1 min)
- 1-source RMSE: 0.1481 (vs baseline 0.1404)
- 2-source RMSE: 0.2243 (vs baseline 0.2235)

**Outliers (RMSE > 0.4):**
| Sample | Sources | RMSE |
|--------|---------|------|
| 57 | 2-src | 0.8126 |
| 40 | 2-src | 0.4410 |
| 63 | 2-src | 0.4068 |
| 65 | 2-src | 0.4025 |

**Sigma-Time Relationship:**
| Sigma | Score | Time |
|-------|-------|------|
| 0.15/0.20 (baseline) | 1.1247 | 57.2 min |
| 0.20/0.25 | 1.1362 | 69.2 min |
| 0.30/0.35 | 1.1396 | 68.3 min |

**Key Finding**:
- Higher sigma ALWAYS improves accuracy but proportionally increases time
- CMA-ES internal sigma adaptation does NOT give "free" exploration
- The hypothesis that CMA-ES would adapt down quickly was WRONG
- Time cost is fixed: ~10-11 min per sigma increment regardless of starting point

**Key Insight**: The sigma-time trade-off is fundamental to CMA-ES. There is no way to get higher sigma exploration benefits without paying the time cost. We cannot beat baseline 1.1247 @ 57.2 min within budget.

**Conclusion**: Adaptive sigma approach ABANDONED. CMA-ES parameter tuning is EXHAUSTED.

---

### [W2] Experiment: sample_57_specific_handling (EXP_OUTLIER_FOCUS_001) | Algorithm: CMA-ES Outlier Handling | Score: 1.1167 @ 68.2 min
**Config**: base_fevals=20/36, hard_sample_extra_fevals=10, hard_threshold=0.4 (initial RMSE), sigma=0.15/0.20, fallback_threshold=0.35/0.45
**Result**: **FAILED** - OVER BUDGET (+8.2 min) AND worse score (-0.008)

**Analysis**:
- Score: 1.1167 vs baseline 1.1247 (-0.0080) - worse than baseline
- Time: 68.2 min projected (13.7% over 60 min budget)
- Hard samples detected: 51/80 (63.8%) - threshold too aggressive
- Hard samples improved (RMSE < 0.4): 44/51
- Hard samples still bad (RMSE >= 0.4): 7/51
- High RMSE outliers: Sample 63 (0.6066), Sample 67 (0.5814), Sample 57 (0.5732), Sample 50 (0.5123)

**Root Cause**:
1. Initial RMSE threshold of 0.4 flagged 64% of samples as "hard"
2. Quick initial evaluation (2-4 sims) adds overhead without benefit
3. Extra 10 fevals per hard sample adds massive time overhead
4. Hard samples often remain hard even with extra fevals
5. Sample 57 still has RMSE 0.5732 despite extra fevals + fallback triggered

**Key Finding**:
- Initial RMSE is a POOR predictor of optimization difficulty
- The problem with hard samples is not feval budget, but initialization quality
- Extra fevals don't help if CMA-ES starts in a bad basin

**Why Sample 57 Keeps Failing**:
- This 2-source sample has sources that are close together or aligned in a way that confuses triangulation
- Neither triangulation nor smart init provides a good starting point
- CMA-ES gets stuck in local minima regardless of feval budget

**Key Insight**: Hard samples need better initialization strategies (e.g., multiple random restarts, physics-informed init), not more function evaluations. The initial guess quality matters more than the search budget.

**Recommendation**: Focus on initialization quality, not adaptive feval budgets.

---

### [W2] Experiment: targeted_2src_sigma_v1 | Algorithm: CMA-ES Targeted 2-Source | Score: 1.1236 @ 68.2 min
**Config**: 1-src (sigma=0.15, fevals=20), 2-src (sigma=0.25, fevals=36)
**Result**: **PARTIAL** - Improved 2-src RMSE but OVER BUDGET

**Analysis**:
- Score: 1.1236 vs baseline 1.1247 (-0.0011)
- Time: 68.2 min projected (OVER BUDGET by 8.2 min)
- 1-source RMSE: 0.1434 (similar to baseline 0.1397)
- 2-source RMSE: 0.2183 (IMPROVED vs baseline 0.2337, delta -0.0154!)
- Sample 57: RMSE 0.4050 (IMPROVED from 0.5732!)
- Sample 50: RMSE 0.6926 (new major outlier)

**Key Finding**: Higher sigma for 2-source (0.25 vs 0.20) DOES improve 2-source accuracy by ~7%, but adds ~11 min overhead due to CMA-ES exploration.

---

### [W2] Experiment: targeted_2src_sigma_v2 | Algorithm: CMA-ES Targeted 2-Source Reduced | Score: 1.1177 @ 57.5 min
**Config**: 1-src (sigma=0.15, fevals=18), 2-src (sigma=0.25, fevals=32)
**Result**: **FAILED** - IN BUDGET but worse score

**Analysis**:
- Score: 1.1177 vs baseline 1.1247 (-0.0070)
- Time: 57.5 min projected (IN BUDGET)
- 1-source RMSE: 0.1469 (slightly worse)
- 2-source RMSE: 0.2438 (WORSE than baseline 0.2337)
- Major outliers: Sample 63 (0.7344), Sample 50 (0.5558)

**Key Finding**: Cannot compensate for higher sigma with reduced fevals - accuracy loss exceeds exploration benefit.

**Conclusion from Targeted 2-Source Experiments**:
- Higher sigma (0.25) for 2-source improves 2-source accuracy (~7%)
- But time cost (~11 min) cannot be offset by reducing fevals
- Trade-off is fundamental: exploration vs time
- Baseline sigma (0.15/0.20) with fevals (20/36) remains optimal in-budget config

---

### [W2] Experiment: pso_optimizer (EXP_PSO_001) | Algorithm: Particle Swarm Optimization | Score: 1.0978 @ 54.5 min
**Config**: n_particles=8, pso_iterations=8, fevals=20/36, w=0.7, c1=1.5, c2=1.5
**Result**: **FAILED** - FASTER but significantly WORSE accuracy

**Analysis**:
- Score: 1.0978 vs baseline 1.1247 (-0.0269)
- Time: 54.5 min projected (IN BUDGET - 2.7 min faster than CMA-ES!)
- 1-source RMSE: 0.2771 (MUCH WORSE than CMA-ES 0.1397!)
- 2-source RMSE: 0.2391 (similar to CMA-ES 0.2337)
- Major outliers: Sample 21 (RMSE 2.13), Sample 28 (RMSE 1.74), Sample 7 (RMSE 0.71)

**Key Finding**: PSO fails badly on 1-source problems (2D optimization) where CMA-ES succeeds easily. PSO velocity-based dynamics lack the adaptive covariance matrix that makes CMA-ES effective at learning the local landscape geometry.

**Why PSO Fails Here**:
1. PSO uses fixed cognitive/social coefficients - doesn't adapt to landscape
2. CMA-ES learns covariance structure of the fitness landscape
3. For heat source identification, landscape has elongated/curved valleys
4. CMA-ES can follow these valleys; PSO bounces around inefficiently

**Conclusion**: PSO is NOT a viable replacement for CMA-ES. While faster, the accuracy loss is unacceptable. CMA-ES's covariance adaptation is essential for this problem.

---

### [W1] Experiment: neural_network_surrogate_prefilter (EXP_SURROGATE_NN_001) | Algorithm: Surrogate NN Pre-filter | Score: 1.1203 @ 75.6 min
**Config**: MLP surrogate (64-32 hidden), filter_ratio=50%, online learning per worker
**Result**: **FAILED** - SLOWER and no filtering occurred

**Analysis**:
- Score: 1.1203 vs baseline 1.1247 (-0.0044)
- Time: 75.6 min projected (OVER BUDGET by 15.6 min!)
- Total sims: 9242, Filtered: 0 (0.0%)
- 1-source RMSE: 0.1539 (similar to baseline)
- 2-source RMSE: 0.2092 (similar to baseline)

**Why Filtering Never Activated**:
1. Each worker process has its own optimizer instance
2. Training data collected per sample is too sparse (~20-50 samples/run)
3. Need 20+ samples per source type (1-src vs 2-src) to train
4. Parallel processing means each worker only sees fraction of total samples
5. Surrogate never accumulated enough data to become useful

**Root Cause**:
- Online learning during parallel processing doesn't work
- Each worker gets ~11 samples (80/7 workers)
- Need minimum ~20 training samples per source type
- Workers don't share training data

**What Would Work Better**:
1. PRE-TRAIN surrogate on existing CMA-ES run data before deployment
2. Use CENTRALIZED data collection (not per-worker)
3. Train on coarse grid RMSE values (cheap to compute)
4. Pre-compute surrogate from previous experiments' data

**Key Insight**: Surrogate-assisted CMA-ES requires PRE-TRAINING or centralized data sharing. Online learning per-worker is insufficient for parallel execution.

---

### [W2] Experiment: cobyla_refine (EXP_COBYLA_REFINE_001) | Algorithm: COBYLA Trust Region Refinement | Score: 1.1235 @ 88.8 min
**Config**: COBYLA maxiter=20, rhobeg=0.1, replacing Nelder-Mead refinement
**Result**: **FAILED** - Better accuracy but MASSIVELY over budget

**Analysis**:
- Score: 1.1235 vs baseline 1.1247 (-0.0012)
- Time: 88.8 min projected (31.6 min SLOWER than baseline!)
- 1-source RMSE: 0.1335 (IMPROVED from 0.1397)
- 2-source RMSE: 0.2050 (IMPROVED from 0.2337)
- COBYLA improved: 66/80 samples (82.5%)

**Why COBYLA Fails Here**:
1. COBYLA builds a linear model at each iteration, requiring n+1 function evaluations per step
2. With maxiter=20 and 2 candidates to refine, that's ~40+ extra function evaluations
3. Nelder-Mead with maxiter=3 only adds ~6-12 function evaluations
4. The accuracy improvement doesn't justify the 55% time increase

**Key Insight**: For this problem, a quick 3-iteration Nelder-Mead refinement is more efficient than COBYLA. The marginal accuracy gain from COBYLA is overwhelmed by its function evaluation cost.

**Recommendation**: Keep Nelder-Mead for refinement. COBYLA would need analytic gradients to be competitive.

---

### [W2] Experiment: fast_ica (EXP_FAST_ICA_001) | Algorithm: Accelerated ICA Decomposition | Score: 1.1046 @ 151.1 min
**Config**: FastICA max_iter=50, coarse search (50x25), Nelder-Mead refinement, fevals 10/15
**Result**: **FAILED** - Massively over budget and worse accuracy

**Analysis**:
- Score: 1.1046 vs baseline 1.1247 (-0.0201)
- Time: 151.1 min projected (91 min OVER BUDGET!)
- 1-source RMSE: 0.1202 (good)
- 2-source RMSE: 0.2188 (okay)
- ICA best init: 23/48 2-source samples (47.9%)
- 2 samples had RMSE=inf (errors in refinement)

**Why Fast ICA Failed**:
1. The coarse-to-fine strategy DOUBLED simulation count instead of reducing it
2. Each sample runs: CMA-ES on coarse grid + Nelder-Mead refinement on fine grid
3. Refinement step adds 100+ simulations per sample
4. Total sims: ~190-200 for 2-source (vs ~70-100 in baseline)
5. ICA decomposition itself is fast (milliseconds) - NOT the bottleneck

**Critical Misunderstanding**:
The original ICA experiment's 87 min runtime was NOT from ICA computation. It was from:
1. Batched transfer learning with history accumulation
2. Multiple initialization sources being fully optimized
3. Complex feature extraction and matching

**Key Insight**: The ICA signal decomposition is lightning fast (<10ms). The bottleneck was never the ICA algorithm - it was the optimizer workflow around it. Adding coarse-to-fine approach only made things worse.

**What the Original ICA Got Right**:
- ICA provides excellent 2-source initialization (47.9% win rate)
- Position estimates from mixing matrix are physically meaningful
- Linear superposition principle enables signal separation

**Recommendation**: ICA is valuable as an initialization method but CANNOT be accelerated by changing ICA parameters. The time cost comes from simulation evaluations, not ICA computation. To use ICA within budget:
1. Use ICA init WITHOUT additional refinement stages
2. Replace other inits with ICA (not add to them)
3. Keep existing CMA-ES workflow, just swap triangulation for ICA init


---

### [W2] Experiment: cluster_transfer (EXP_TRANSFER_LEARN_001) | Score: 1.0804 @ 59.7 min
**Algorithm**: Cluster-based transfer learning with K-means clustering on sensor features
**Tuning Runs**: 3 runs (see experiments/cluster_transfer/STATE.json for details)
**Result**: **FAILED** vs baseline (1.1247 @ 57 min)
**Key Finding**: Sensor feature clustering does NOT predict solution similarity - transfer learning fundamentally doesn't work for inverse heat problems.

**Hypothesis Tested**: Similar sensor patterns (temperature profiles, onset times, spatial centroids) have similar optimal heat source configurations. Clustering samples and transferring solutions from cluster representatives should reduce optimization time.

**Tuning History**:
| Run | Transfer Fevals | Transfer Sigma | Score | Time | Status |
|-----|-----------------|----------------|-------|------|--------|
| 1 | 10 | 0.08 | 1.0334 | 61.2 min | 5 failures, over budget |
| 2 | 15 | 0.12 | 1.0804 | 59.7 min | Best - still 4% below baseline |
| 3 | 18 | 0.15 | 1.0600 | 61.4 min | MORE budget = WORSE results |

**Why Cluster Transfer Failed**:
1. **Sensor features don't predict solutions**: The relationship between observed temperatures and source locations is highly non-linear. Samples with similar temperature patterns can have very different heat source positions.
2. **Inverse problem degeneracy**: Multiple source configurations can produce similar temperature patterns, so similar features don't guarantee similar solutions.
3. **Transfer hurt exploration**: Starting CMA-ES from transferred positions reduced diversity, causing the optimizer to get stuck in suboptimal regions.
4. **Run 3 paradox**: Giving transfer mode MORE budget made results WORSE (1.0600 vs 1.0804), confirming transfer init was directing search to wrong regions.

**Critical Insight**: For inverse heat problems, the mapping from observations to sources is too non-linear for feature-based clustering to provide useful warm-starts. The approach fundamentally doesn't work.

**Recommendation**: 
- Do NOT pursue cluster-based or feature-based transfer learning for this problem
- WS-CMA-ES (covariance transfer, not solution transfer) may still be viable - transferring the learned search landscape shape rather than solutions
- Focus on surrogate methods (lq-CMA-ES) or multi-fidelity instead of transfer learning


### [W1] Experiment: lq_cma_es_builtin | RMSE: 0.253 @ 64.1 min | FAILED
**Algorithm**: pycma's built-in lq-CMA-ES (linear-quadratic surrogate)
**Tuning Runs**: 3 runs
**Result**: **FAILED** - API mismatch, 29-71% worse RMSE, 7-17% slower than baseline
**Key Finding**: fmin_lq_surr returns ONE solution but we need MULTIPLE candidates for diversity scoring. High-level API doesn't expose intermediate population solutions. See experiments/lq_cma_es_builtin/SUMMARY.md for details.


---

### [W1] Experiment: multi_fidelity_pyramid (EXP_MULTIFID_OPT_001) | RMSE: 0.259 @ 44.9 min | FAILED
**Algorithm**: Multi-fidelity pyramid with 3-level grid (25x12 -> 50x25 -> 100x50)
**Tuning Runs**: 3 runs
**Result**: **FAILED** - Coarse grid RMSE landscape differs from fine grid. 45% worse accuracy than baseline.
**Key Finding**: Multi-fidelity via grid coarsening doesn't work for inverse problems. Solutions don't transfer between resolutions. See experiments/multi_fidelity_pyramid/SUMMARY.md for details.

**Tuning History**:
| Run | Config | RMSE Mean | Time (min) | Status |
|-----|--------|-----------|------------|--------|
| 1 | 25x12 coarse, 10/16 fevals | 0.2589 | 44.9 | In budget, 45% worse |
| 2 | 30x15 coarse, 15/24 fevals | 0.2320 | 92.2 | Over budget, 21% worse |
| 3 | Same as Run 2 | 0.3039 | 73.3 | High variance, 79% worse |

**Why Multi-Fidelity Pyramid Failed**:
1. The RMSE landscape is fundamentally different at different grid resolutions
2. Optimal source positions on coarse grid (25x12) do NOT correspond to optimal on fine grid (100x50)
3. Heat diffusion physics change with cell size - coarse grids approximate poorly
4. High variance between runs with same config (31% RMSE difference)

**Recommendation**: ABANDON multi-fidelity via spatial grid coarsening. Alternative: try temporal fidelity (fewer timesteps) instead.


---

### [W2] Experiment: cmaes_to_nm_sequential (EXP_SEQUENTIAL_HANDOFF_001) | Score: 1.1132 @ 56.6 min | FAILED
**Algorithm**: Sequential CMA-ES to Nelder-Mead handoff
**Tuning Runs**: 4 runs
**Result**: **FAILED** - Best in-budget score 1.1132 is WORSE than baseline 1.1247. See experiments/cmaes_to_nm_sequential/SUMMARY.md for details.

**Hypothesis Tested**: Sequential handoff captures CMA-ES exploration + NM local refinement without doubling simulation count (unlike the failed ensemble approach).

**Tuning History**:
| Run | Config | Score | Time (min) | Status |
|-----|--------|-------|------------|--------|
| 1 | NM maxiter=40/50, fine grid | ABORTED | - | 500+ sec/sample |
| 2 | NM maxiter=8/12, fine grid | 1.1439 | 126.5 | Best score, 2x over budget |
| 3 | NM maxiter=12/18, coarse grid | 1.1331 | 108.3 | 48 min over budget |
| 4 | NM maxiter=3/5, coarse grid | 1.1132 | 56.6 | IN BUDGET but BELOW baseline |

**Why Sequential Handoff Failed**:
1. **CMA-ES uses full budget**: Baseline CMA-ES already takes ~57 min, leaving no room for NM
2. **NM's value requires time**: NM improves score (+0.0192 in Run 2) only when given adequate iterations
3. **Accuracy-time tradeoff unfavorable**: Reducing NM to fit budget eliminates the accuracy benefit
4. **Coarse NM loses precision**: Moving NM to coarse grid helps time but hurts accuracy

**Critical Insight**: This approach fundamentally doesn't work because ANY post-processing after CMA-ES pushes us over budget. When reduced enough to fit, the benefit disappears.

**Recommendation**: ABANDON hybrid/sequential handoff approaches. Focus on improving CMA-ES itself (initialization, surrogate) rather than adding post-processing.


---

### [W2] Experiment: fast_source_count_detection (EXP_FAST_SOURCE_DETECT_001) | ABORTED
**Algorithm**: Peak detection / feature-based classification for n_sources detection
**Status**: **ABORTED** - Invalid premise
**Key Finding**: Baseline already uses `sample['n_sources']` from data. Detection not needed. See experiments/fast_source_count_detection/SUMMARY.md for details.

**Why This Experiment Was Aborted**:
1. **Invalid premise**: The baseline optimizer already has direct access to `n_sources` from the sample data. Detection is unnecessary.
2. **Not feasible anyway**: Even if needed, sensor-based features only achieve 67% accuracy (target was 95%). Sparse sensor data doesn't carry enough information to distinguish 1-source from 2-source.

**Analysis**:
- Simple heuristic accuracy: 57.5% (barely better than random)
- Random Forest CV accuracy: 67.5% (far below 95% target)
- No features showed clear separation between 1-src and 2-src samples

**Lesson Learned**: Always verify the problem exists before trying to solve it. Saved significant implementation time by catching the flawed premise early.

---

### [W1] Experiment: warm_start_cmaes (EXP_WS_CMAES_001) | RMSE: 0.276 @ 65.8 min | FAILED
**Algorithm**: Warm Start CMA-ES (WS-CMA-ES) from CyberAgentAILab's cmaes library
**Tuning Runs**: 2 runs
**Result**: **FAILED** - WS-CMA-ES causes divergence, 62-170% worse accuracy than baseline
**Key Finding**: WS-CMA-ES doesn't work for thermal inverse problems. Each sample is unique with no shared landscape structure. Probing phase wastes budget. meta_learning family EXHAUSTED.

**Tuning History**:
| Run | Config | RMSE | Time (min) | Status |
|-----|--------|------|------------|--------|
| 1 | 3 probing starts, 5 fevals | 0.2759 | 65.8 | Over budget, 62% worse |
| 2 | 2 probing starts, 3 fevals | 0.4599 | 23.3 | In budget, 170% worse |

**Why WS-CMA-ES Failed**:
1. Each sample has unique heat source positions - no transfer between samples possible
2. Probing phase wastes budget without providing useful warm start information
3. CyberAgentAILab's cmaes library not as well-tuned as pycma for this problem
4. get_warm_start_mgd() combines information from random probes that don't help convergence

**Recommendation**: ABANDON meta_learning approaches (cluster_transfer + WS-CMA-ES both failed). Focus on within-sample optimizations.

---

### [W1] Experiment: adaptive_sample_budget (EXP_ADAPTIVE_BUDGET_001) | Score: 1.1143 @ 56.2 min | FAILED
**Algorithm**: Adaptive budget allocation via early termination + bonus budget for hard samples
**Tuning Runs**: 3 runs
**Result**: **FAILED** - Early termination hurts CMA-ES accuracy. Fixed-budget baseline is already optimal.
**Key Finding**: CMA-ES needs its full budget to adapt the covariance matrix properly. Early termination based on sigma/stagnation hurts accuracy. budget_allocation family FAILED.

**Tuning History**:
| Run | Config | Score | Time (min) | Early Term % | Status |
|-----|--------|-------|------------|--------------|--------|
| 1 | fevals 20/36, bonus 10/18, sigma<0.01 | 1.1277 | 75.4 | 0% | Over budget, bonus adds overhead |
| 2 | fevals 20/36, no bonus, sigma<0.05 | 1.1070 | 70.2 | 23.8% | Over budget, -1.8% accuracy |
| 3 | fevals 15/28, no early term | 1.1143 | 56.2 | 0% | In budget, -1% vs baseline |

**Why Adaptive Budget Failed**:
1. **Early termination hurts accuracy**: When 23.8% of samples terminate early (Run 2), overall score drops -0.018 vs baseline
2. **No reliable convergence signal**: Sigma < threshold doesn't mean optimum is found
3. **Bonus budget adds overhead**: Extra fevals for hard samples increases time without corresponding savings
4. **Fixed-budget is optimal**: The baseline's 15/28 fevals was already tuned to be near-optimal
5. **Parallel processing limits redistribution**: Can't easily share saved budget between parallel workers

**Critical Insight**: CMA-ES is designed to use its full budget effectively. The algorithm adapts its strategy based on progress. Premature termination disrupts this adaptation.

**Recommendation**: ABANDON budget reallocation approaches. Focus on reducing per-eval cost (temporal fidelity, surrogate models) rather than reducing number of evals.

---

### [W2] Experiment: niching_cmaes_diversity (EXP_NICHING_CMAES_001) | Score: 1.0622 @ 46.9 min | FAILED
**Algorithm**: Niching CMA-ES with taboo regions for diversity
**Tuning Runs**: 2 runs
**Result**: **FAILED** - Niching hurts accuracy more than it helps diversity. Baseline already near-optimal.
**Key Finding**: Scoring formula AVERAGES accuracy over candidates. Adding diverse but worse candidates hurts score. diversity family EXHAUSTED.

**Critical Discovery - The Scoring Formula**:
```
score = (1/N) * sum(1/(1+L_i)) + 0.3 * (N/3)
```
- First term AVERAGES accuracy over all N candidates
- Adding worse diverse candidates decreases average accuracy
- Diversity gain (0.1 per candidate) doesn't compensate for accuracy loss

**Tuning History**:
| Run | Config | Score | Time (min) | N_valid | Status |
|-----|--------|-------|------------|---------|--------|
| 1 | Taboo niching (radius=0.15) | 1.0622 | 46.9 | 1.70 | FAILED (-0.11 vs baseline) |
| 2 | Baseline analysis | 1.1741 | 47.2 | 2.75 | 80% of samples already have 3 candidates |

**Why Niching Failed**:
1. **Baseline already achieves near-maximum diversity**: 80% of samples get 3 candidates, avg N_valid = 2.75
2. **Taboo regions push to suboptimal solutions**: Penalizing proximity to found optima causes worse RMSE
3. **Scoring punishes worse candidates**: Even if diverse, adding high-RMSE candidates hurts average
4. **One global optimum per sample**: Thermal inverse problems don't have multiple distinct optima

**Example Calculation**:
- 1 candidate @ RMSE=0: score = 1.0 + 0.1 = 1.1
- 3 candidates @ RMSE=[0, 0.5, 0.5]: score = (1/3)*(1 + 0.67 + 0.67) + 0.3 = 0.78 + 0.3 = 1.08

Adding diverse but worse candidates DECREASED score from 1.1 to 1.08!

**Recommendation**: ABANDON diversity optimization. Focus on accuracy improvement only. Baseline is already near-optimal for diversity.

---

### [W2] Experiment: extended_nm_polish (EXP_EXTENDED_POLISH_001) | Score: 1.1703 @ 82.3 min | FAILED
**Algorithm**: Extended NM polish iterations (8 -> 12)
**Tuning Runs**: 3 runs (10 iter, 12 iter on 20 samples, 12 iter on 80 samples)
**Result**: **FAILED** - 12 iterations exceeds budget by 37% (82.3 min vs 60 min limit)
**Key Finding**: 8 NM iterations is already optimal. Each additional iteration adds ~6 min. +0.0015 score not worth +24 min.

**Why Extended Polish Failed**:
1. **Time cost is prohibitive**: 12 iterations = 82.3 min (37% over budget)
2. **Marginal improvement**: Only +0.0015 score gain
3. **2-source samples expensive**: Each NM iteration requires 2 simulations per 2-src sample
4. **Diminishing returns**: 8 iterations already captures most of the benefit

**Recommendation**: 8 NM polish iterations is optimal. Time budget is fully utilized - cannot add more refinement.

---

### [W2] Experiment: adaptive_timestep_fraction (EXP_ADAPTIVE_TIMESTEP_001) | Score: 1.1635 @ 69.9 min | FAILED
**Algorithm**: Adaptive timestep fidelity during CMA-ES (start 25%, switch to 40% mid-run)
**Tuning Runs**: 2 runs
**Result**: **FAILED** - Variable timestep fidelity during CMA-ES is counterproductive
**Key Finding**: CMA-ES covariance adaptation requires consistent fitness landscape. Switching fidelity mid-run disrupts learning and leads to WORSE performance.

**Tuning History**:
| Run | Config | Score | Time (min) | Status |
|-----|--------|-------|------------|--------|
| 1 | 25%->40% at 50% switch | 1.1473 | 66.0 | Over budget, -0.0215 vs baseline |
| 2 | 35%->40% at 50% switch | 1.1635 | 69.9 | Over budget, -0.0053 vs baseline |

**Baseline**: 40% timesteps FIXED = 1.1688 @ 58.4 min

**Why Adaptive Timestep Failed**:
1. **CMA-ES covariance adaptation disrupted**: Algorithm learns correlation structure based on fitness. Changing fitness mid-run invalidates learned covariance.
2. **Lower early fidelity doesn't save time**: Poor initial guidance leads to worse solutions requiring more fallback runs
3. **Net effect is negative**: Both time AND accuracy are worse than fixed 40%

**Comparison to Fixed Approaches** (from baseline experiment):
| Config | Score | Time |
|--------|-------|------|
| Fixed 25% | 1.1219 | 30.3 min |
| Fixed 40% | 1.1362 | 39.0 min |
| Fixed 40% + 8 NM polish | 1.1688 | 58.4 min |
| Adaptive 25%->40% | 1.1473 | 66.0 min |
| Adaptive 35%->40% | 1.1635 | 69.9 min |

Adaptive approaches are WORSE than both fixed alternatives!

**Recommendation**: ABANDON adaptive timestep approaches. Fixed 40% + NM polish is optimal. Multi-fidelity requires specialized algorithms (not CMA-ES).

---

### [W1] Experiment: physics_informed_init | Score: 1.1639 @ 69.6 min
**Algorithm**: Physics-informed initialization using temperature gradients
**Tuning Runs**: 2 runs (A/B test: gradient init vs smart init)
**Result**: **FAILED** - Gradient init is WORSE than simple hottest-sensor approach
**Key Finding**: Temperature gradients at sensors don't accurately point to sources. Heat diffusion and boundary effects corrupt the gradient signal.

**A/B Test Results**:
| Initialization | Score | Time (min) | RMSE Mean |
|----------------|-------|------------|-----------|
| Gradient + Tri | 1.1593 | 71.9 | 0.1360 |
| Smart + Tri | 1.1639 | 69.6 | 0.1369 |
| **Delta** | **-0.0046** | **+2.3** | -0.0009 |

**Why Gradient Init Failed**:
1. **Heat diffusion corrupts gradients**: By the time heat reaches sensors, gradients don't point to sources
2. **Boundary effects**: Add complexity that confuses gradient estimation
3. **Sensor spacing**: Too coarse (~0.1-0.2 units) for accurate local gradient estimation
4. **Multiple sources**: Create interference patterns that mislead gradient analysis

**Conclusion**: Simple hottest-sensor (smart init) approach is already optimal. The initialization family is now EXHAUSTED.


---

### [W2] Experiment: pod_reduced_order_model (EXP_POD_SURROGATE_001) | ABORTED
**Algorithm**: POD (Proper Orthogonal Decomposition) as fast surrogate
**Result**: **ABORTED** - POD not viable for this problem
**Key Finding**: Sample-specific physics (kappa, bc, T0) prevents building a universal POD basis. Each sample is unique.

**Feasibility Check Results** (from prior analysis):
| Config | 10 modes | 20 modes |
|--------|----------|----------|
| 1-source error | 4.8% | 2.1% |
| 2-source error | 5.9% | 2.9% |

POD CAN mathematically capture temperature fields, but:
1. **Variable physics**: Each sample has different kappa - can't pre-build universal basis
2. **Online POD defeats purpose**: Building sample-specific POD requires simulations
3. **Temporal fidelity already works**: 40% timesteps gives 2.5x speedup with 5% error

**Recommendation**: ABANDON surrogate approaches for this problem. Focus on temporal fidelity extensions.

---

### [W2] Experiment: two_source_specialized (EXP_2SOURCE_FOCUS_001) | Score: 1.1620 @ 69.3 min | FAILED
**Algorithm**: Specialized feval allocation for 1-source vs 2-source samples
**Tuning Runs**: 1 run (16/42 fevals)
**Result**: **FAILED** - Specialized allocation hurts performance on all metrics
**Key Finding**: Baseline 20/36 feval split is already optimal. 2-source is harder due to 4D search space, not under-optimization.

**Results**:
| Config | Score | Time | 1-src RMSE | 2-src RMSE | Status |
|--------|-------|------|------------|------------|--------|
| Baseline (20/36) | 1.1688 | 58.4 | 0.104 | 0.138 | - |
| Specialized (16/42) | 1.1620 | 69.3 | 0.1157 | 0.1558 | FAILED |

**Why Specialized Allocation Failed**:
1. **Reducing 1-src fevals hurts accuracy**: 16 fevals → 0.1157 RMSE (vs 0.104 baseline)
2. **Increasing 2-src fevals doesn't help**: 42 fevals → 0.1558 RMSE (vs 0.138 baseline)
3. **Time overhead prohibitive**: +10.9 min for worse accuracy

**Recommendation**: ABANDON source-specific feval allocation. Baseline is already optimal.

---

### [W1] Experiment: larger_cmaes_population | Score: 1.1666 @ 73.0 min
**Algorithm**: Larger CMA-ES population size (popsize=12)
**Tuning Runs**: 1 run (abort criteria met)
**Result**: **FAILED** - Larger popsize adds time without improving accuracy
**Key Finding**: Default popsize is already optimal. Larger popsize reduces number of generations with fixed feval budget.

**Results**:
| Popsize | Score | Time (min) | Delta vs Baseline |
|---------|-------|------------|-------------------|
| Default (~6/8) | 1.1688 | 58.4 | (baseline) |
| 12/12 | 1.1666 | 73.0 | -0.0022, +14.6 min |

**Why Larger Popsize Failed**:
1. **Fixed feval budget limits generations**: With popsize=12 and max_fevals=20, only ~2 generations run. CMA-ES needs multiple generations to adapt covariance.
2. **Default formula is optimal**: 4+floor(3*ln(n)) is already well-tuned for small dimensions
3. **More simulations per generation** without more accuracy = pure overhead

**Conclusion**: cmaes_accuracy family should be marked as EXHAUSTED. Default CMA-ES settings are already optimal.


---

### [W2] Experiment: active_cmaes_covariance (EXP_ACTIVE_CMAES_001) | ABORTED
**Algorithm**: Active CMA-ES variant with negative covariance update
**Result**: **ABORTED** - Wrong premise. CMA_active already True by default.
**Key Finding**: pycma's CMA_active option defaults to True. The baseline ALREADY uses active CMA-ES. No experiment needed.

```python
import cma
print(cma.CMAOptions()['CMA_active'])
# Output: 'True  # negative update, conducted after the original update'
```

---

### [W1] Experiment: adaptive_sigma_schedule | Status: ABORTED
**Algorithm**: Sigma scheduling (high early, low late)
**Tuning Runs**: 0 (aborted before testing)
**Result**: **ABORTED** - Prior evidence shows this approach will fail
**Key Finding**: CMA-ES already adapts sigma naturally. Manual scheduling cannot improve without adding time.

**Rationale for Abort**:
1. EXP_TEMPORAL_HIGHER_SIGMA_001 showed high sigma adds 5+ min without in-budget accuracy gain
2. EXP_ADAPTIVE_TIMESTEP_001 showed changing conditions mid-run disrupts CMA-ES
3. The fundamental tradeoff (high sigma = more exploration = more time) cannot be bypassed

**Recommendation**: sigma_scheduling family should be marked as EXHAUSTED. Focus on different approaches.


---

### [W2] Experiment: progressive_polish_fidelity (EXP_PROGRESSIVE_POLISH_FIDELITY_001) | ABORTED
**Algorithm**: Progressive timestep fidelity during NM polish (60% early, 100% final)
**Result**: **ABORTED** - Prior experiment showed truncated polish HURTS accuracy
**Key Finding**: early_timestep_filtering Run 8 showed 40% polish = 1.1342 vs full polish = 1.1688 (-0.0346). 60% would similarly hurt.

**From early_timestep_filtering SUMMARY.md:**
> "NM polish on truncated timesteps HURTS... The truncated signal is a proxy; polishing the proxy overfits to noise"

Full timestep polish is optimal. Do not use reduced fidelity during polish.


---

### [W1] Experiment: boundary_aware_initialization (EXP_BOUNDARY_AWARE_INIT_001) | ABORTED
**Algorithm**: Initialize heat sources away from domain boundaries
**Tuning Runs**: 0 (aborted based on data analysis)
**Result**: **ABORTED** - Data analysis shows 24% of samples have boundary sources
**Key Finding**: Biasing initialization away from boundaries would hurt 24% of samples.

**Data Analysis**:
```
Hottest sensor location (proxy for source location):
  Interior (10-90% of domain): 61/80 (76.2%)
  Near boundary (<10% margin): 19/80 (23.8%)
```

**Why Boundary-Aware Init Would Fail**:
1. **24% of samples have boundary hotspots** - biasing away would hurt these cases
2. **Abort criteria explicitly met**: "Boundary constraint hurts cases with actual boundary sources"
3. **Smart init already optimal**: Hottest sensor directly targets most likely source location
4. **EXP_PHYSICS_INIT_001 already showed**: Initialization modifications don't help

**Recommendation**: initialization_v2 family marked EXHAUSTED. Smart init (hottest sensor) is optimal.

---

## Session Summary - W1 Experiments Today

### Experiments Completed This Session:
| Experiment | Status | Key Finding |
|------------|--------|-------------|
| larger_cmaes_population | FAILED | Popsize=12 adds 14 min, -0.0022 score. Default is optimal. |
| adaptive_sigma_schedule | ABORTED | CMA-ES already adapts sigma. Manual scheduling can't bypass time/accuracy tradeoff. |
| weighted_parameter_scaling | ABORTED | Wrong premise - CMA-ES only optimizes (x,y), q is analytical. |
| boundary_aware_initialization | ABORTED | 24% samples have boundary sources. Biasing away would hurt. |

### Families Marked EXHAUSTED This Session:
- cmaes_accuracy (default popsize optimal)
- sigma_scheduling (CMA-ES adaptation is optimal)
- problem_specific (wrong premise)
- initialization_v2 (24% boundary sources exist)

### Current State:
- **Best Score**: 1.1688 @ 58.4 min (W2 temporal fidelity baseline)
- **Queue Status**: EMPTY - All experiments completed
- **17+ families EXHAUSTED** - Most optimization avenues explored
- **Remaining Approaches**: Need new experiment ideas or focus on final submission

---

### [W1] Experiment: boundary_aware_initialization (EXP_BOUNDARY_AWARE_INIT_001) | ABORTED
**Algorithm**: Initialize heat sources away from domain boundaries
**Result**: **ABORTED** - Data analysis shows 24% of samples have boundary hotspots
**Key Finding**: Biasing initialization away from boundaries would hurt cases where actual sources ARE near boundaries.

**Abort Criteria Met**: "Boundary constraint hurts cases with actual boundary sources"

The initialization_v2 family is now EXHAUSTED.

---

## Session Summary: Queue Exhausted (2026-01-20)

**Status**: ALL EXPERIMENTS COMPLETE

**Best Result**: 1.1688 @ 58.4 min (40% timesteps + 8 NM polish, from early_timestep_filtering)

**Total Experiments Tested**: 30+
**Successful**: 1 (early_timestep_filtering)
**Failed**: 25+ (various reasons)
**Aborted**: 8 (wrong premise, prior evidence)

**Key Learnings Across All Experiments**:

1. **Temporal fidelity is the key insight**: 40% timesteps provides 2.5x speedup with minimal accuracy loss (RMSE correlation 0.95+)

2. **CMA-ES requires consistency**:
   - Covariance adaptation needs stable fitness landscape
   - Adaptive fidelity, sigma scheduling all HURT performance
   - Default CMA-ES parameters are already well-optimized

3. **Spatial multi-fidelity DOESN'T work**: Coarse grids have fundamentally different RMSE landscapes

4. **Initialization is already optimal**: Smart init (hottest sensor) beats physics-based, gradient-based, and boundary-aware approaches

5. **More fevals/polish/time DOESN'T improve score proportionally**: 8 NM polish iterations is the sweet spot

6. **Diversity is not the bottleneck**: Baseline already achieves 80% 3-candidate samples

**Families Exhausted**:
- evolutionary_cmaes, evolutionary_other, gradient_based, gradient_free_local
- surrogate, surrogate_lq, surrogate_pod, decomposition, ensemble, hybrid
- bayesian_opt, multi_fidelity, budget_allocation, meta_learning, preprocessing
- diversity, initialization, initialization_v2, temporal_fidelity_extended
- sigma_scheduling, cmaes_variants, cmaes_accuracy, source_specific, refinement, problem_specific

**Recommendation**: Current baseline (1.1688 @ 58.4 min) represents a local optimum for this problem formulation. Further gains would require fundamentally new approaches not yet in the queue.


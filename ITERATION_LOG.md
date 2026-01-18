# Iteration Log - Heat Signature Zero

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


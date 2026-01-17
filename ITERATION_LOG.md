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


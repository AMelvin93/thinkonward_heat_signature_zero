# Iteration Log - Heat Signature Zero

> **NOTE**: Full history (Sessions 1-14) archived in `ITERATION_LOG_full_backup.md`

## Current State
- **Current Best**: 1.1233 @ 55.3 min (36 fevals + Coarse Refinement 3-iter top-2)
- **Leaderboard Position**: #7 (tied with Team Okpo)
- **Gap to Top 5**: 0.030 (2.6%)
- **Gap to Top 2**: 0.104 (9%)

### Leaderboard Context (2026-01-13)
```
#1  Team Jonas M     1.2268
#2  Team kjc         1.2265
#3  Matt Motoki      1.2005   ← Big jump! Was #4
#4  MGöksu           1.1585
#5  Team vellin      1.1533   ← Target
#6  Team StarAtNyte  1.1261
#7  Team Okpo        1.1232   ← WE ARE HERE (1.1233)
#8  Koush            1.1096
#9  ekarace          1.1061
#10 Team lukasks     1.0996
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


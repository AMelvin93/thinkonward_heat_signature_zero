# Experiment Summary: perturbed_local_restart

## Metadata
- **Experiment ID**: EXP_PERTURBED_LOCAL_RESTART_001
- **Worker**: W1
- **Date**: 2026-01-27
- **Algorithm Family**: basin_hopping_v1

## Status: SUCCESS

## Objective
Test basin hopping concept: after CMA-ES finds candidates, perturb the top solutions slightly and apply local NM optimization to escape nearby local minima and find better solutions.

## Hypothesis
The CMA-ES + NM pipeline may converge to local minima. By perturbing the best candidates and re-running local optimization, we might discover better nearby solutions that the original optimization missed.

## Baseline
- **Current best**: 1.1373 @ 42.6 min (solution_verification_pass with verify_top_n=1)
- **Old baseline**: 1.1246 @ 32.6 min (early_timestep_filtering, no verification)

## Results Summary

| Run | perturb_top_n | n_perturbations | Other Config | Score | Time (min) | Budget | Status |
|-----|---------------|-----------------|--------------|-------|------------|--------|--------|
| 1   | 3 | 3 | ts=0.40 | 1.1434 | 69.4 | 116% | OVER |
| 2   | 1 | 2 | ts=0.40 | **1.1452** | **47.7** | 79% | **BEST** |
| 3   | 2 | 2 | ts=0.40 | 1.1429 | 57.3 | 96% | IN |

**Best in-budget**: Run 2 with perturb_top_n=1, n_perturbations=2 (Score 1.1452 @ 47.7 min)

## Tuning Efficiency Metrics
- **Runs executed**: 3
- **Time utilization**: 79% (47.7/60 min used at best in-budget)
- **Parameter space explored**: perturb_top_n=[1,2,3], n_perturbations=[2,3]
- **Pivot points**: Run 1->2 (reduced perturbation params after budget overrun)

## Key Findings

### 1. Perturbation Helps Score (+0.0079 vs baseline)
The perturbed local restart approach improves upon the baseline:
- Baseline (solution_verification_pass): 1.1373 @ 42.6 min
- Best perturbed restart: **1.1452 @ 47.7 min** (+0.0079, +5.1 min)

### 2. Optimal Perturbation Configuration
- **perturb_top_n=1**: Only perturb the single best candidate
- **n_perturbations=2**: Create 2 perturbed versions per candidate
- **perturbation_scale=0.05**: Small perturbations (5% of parameter range)
- **perturb_nm_iters=3**: Quick NM polish for perturbed points

More perturbations (perturb_top_n=3, n_perturbations=3) do NOT help:
- Run 1: perturb_top_n=3, n_perturbations=3 → 1.1434 @ 69.4 min (OVER)
- Run 2: perturb_top_n=1, n_perturbations=2 → 1.1452 @ 47.7 min (BETTER score, less time)

### 3. Perturbation Selection Rate
- Run 2: 20 perturbed candidates selected across 20/80 samples (25%)
- This indicates perturbation finds better solutions in ~25% of cases

### 4. Critical Bug Fix During Development
Initial implementation mistakenly used `objective_fine` (100x50 grid) for NM polish instead of `objective_coarse` (50x25 grid). This caused:
- Samples taking 100-300 seconds instead of 30-80 seconds
- Projected runtime of ~187 minutes (3x over budget)

**Fix**: Changed both main NM refinement and perturbation phase to use `objective_coarse`:
```python
# Before (BUG):
result = minimize(objective_fine, ...)

# After (FIXED):
result = minimize(objective_coarse, ...)
```

This reduced projected runtime from 187 min to 47-69 min while maintaining accuracy (coarse grid is sufficient for local polish).

## Algorithm Details

### Perturbation Approach
1. Run standard CMA-ES + NM pipeline to get initial candidates
2. Take top `perturb_top_n` candidates
3. For each candidate, create `n_perturbations` perturbed versions:
   - Add small random noise: `x_perturbed = x + N(0, scale * (ub - lb))`
   - Clip to bounds
4. Run NM polish on each perturbed point (using coarse grid)
5. Add improved perturbed solutions to candidate pool
6. Apply dissimilarity filtering as usual

### Cost Analysis
Each perturbation adds:
- `n_perturbations` NM optimizations with `perturb_nm_iters` max iterations
- For perturb_top_n=1, n_perturbations=2: ~2 extra NM runs

Time overhead vs baseline:
- Baseline (no perturbation): ~42.6 min
- With perturbation: ~47.7 min
- Overhead: ~5 min (+12%)

Score improvement: +0.0079 (+0.7%)

## Comparison to Other Approaches

| Approach | Score | Time | Delta vs baseline |
|----------|-------|------|-------------------|
| solution_verification_pass (baseline) | 1.1373 | 42.6 min | - |
| extended_verification_all_candidates | 1.1432 | 40.0 min | +0.0059, -2.6 min |
| **perturbed_local_restart** | **1.1452** | **47.7 min** | **+0.0079, +5.1 min** |

## Budget Analysis

| Run | Score | Time | Budget Remaining | Decision |
|-----|-------|------|------------------|----------|
| 1   | 1.1434| 69.4 | -9.4 min | PIVOT (reduce perturbation) |
| 2   | 1.1452| 47.7 | +12.3 min | INVEST (try intermediate config) |
| 3   | 1.1429| 57.3 | +2.7 min | RUN 2 REMAINS BEST |

## Recommendations

1. **ADOPT perturbed_local_restart with perturb_top_n=1, n_perturbations=2**
   - Best score achieved: 1.1452 @ 47.7 min
   - Beats baseline by +0.0079 score
   - Stays well within budget (79% utilization)

2. **Consider combining with solution_verification_pass**
   - Perturbation + gradient verification might stack
   - Would need testing to verify overhead is acceptable

3. **Do NOT increase perturbation params beyond (1, 2)**
   - More perturbations increase time without improving score
   - Diminishing returns due to dissimilarity filtering

## Conclusion

**SUCCESS** - Perturbed local restart improves submission score by +0.0079 compared to the baseline (1.1452 vs 1.1373), while staying within budget at 47.7 min (79% utilization). The optimal configuration perturbs only the top candidate with 2 perturbations, achieving the best score/time tradeoff.

This approach should be adopted as the new best-in-budget configuration.

## Raw Data
- Experiment directory: `experiments/perturbed_local_restart/`
- Files: `optimizer.py`, `run.py`, `STATE.json`

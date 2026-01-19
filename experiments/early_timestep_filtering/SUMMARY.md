# EXP_TEMPORAL_FIDELITY_001: Early Timestep Filtering

## Result: MAJOR SUCCESS

**Best Configuration:** 40% timesteps + 8 NM polish iterations (full timesteps)
- **Score:** 1.1688 (+0.0441 vs original baseline 1.1247)
- **Time:** 58.4 min (within 60 min budget)
- **Improvement:** +3.9% accuracy, still in budget

## Summary of Tuning

| Run | Configuration | Score | Time | Delta vs Baseline |
|-----|--------------|-------|------|-------------------|
| 1 | 25% timesteps | 1.1219 | 30.3 | -0.0028 |
| 2 | 35% timesteps | 1.1143 | 36.3 | -0.0104 |
| 3 | **40% timesteps** | 1.1362 | 39.0 | +0.0115 |
| 4 | 45% timesteps | 1.1267 | 40.0 | +0.0020 |
| 5 | 50% timesteps | 1.1250 | 36.3 | +0.0003 |
| 6 | 75% timesteps | 1.1224 | 48.6 | -0.0023 |
| 7 | 50% + more fevals | 1.1155 | 43.8 | -0.0092 |
| 8 | 40% + 8 NM (truncated) | 1.1342 | 45.8 | +0.0095 |
| 9 | 40% + 5 NM polish (full) | 1.1555 | 51.5 | +0.0308 |
| **10** | **40% + 8 NM polish (full)** | **1.1688** | **58.4** | **+0.0441** |

## Key Discoveries

### 1. Temporal Fidelity Works
- 40% of timesteps provides sufficient thermal diffusion information
- Spearman correlation 0.95+ between truncated and full RMSE
- Spatial grid (100x50) remains intact, unlike spatial coarsening which failed

### 2. The Critical Insight: NM Polish on Full Timesteps
- NM refinement on **truncated** timesteps HURTS (Run 8: 1.1342)
- NM polish on **full** timesteps HELPS significantly (Run 10: 1.1688)
- The truncated signal is a proxy; polishing the proxy overfits to noise
- Full-timestep polish refines the actual objective

### 3. RMSE Breakdown

| Configuration | 1-source RMSE | 2-source RMSE |
|--------------|---------------|---------------|
| Original baseline | ~0.14 | ~0.18 |
| 40% no polish | 0.14 | 0.21 |
| 40% + 8 NM polish | **0.104** | **0.138** |

The 2-source RMSE improvement is dramatic: from 0.21 to 0.14 (33% reduction).

## Implementation

```python
# Key parameters
timestep_fraction = 0.40  # Use 40% of timesteps for CMA-ES
final_polish_maxiter = 8  # NM iterations on best candidate with FULL timesteps
```

### Workflow
1. CMA-ES optimization using 40% timesteps (fast proxy)
2. Evaluate top candidates on full timesteps
3. Final NM polish on BEST candidate using full timesteps
4. Return polished result

## Why This Works

1. **CMA-ES with truncated timesteps**: Fast exploration, covariance adaptation still effective
2. **Full evaluation**: Accurate ranking of candidates
3. **NM polish with full timesteps**: Fine-tunes position to true optimum

The combination leverages:
- CMA-ES's global exploration (with 60% speed boost)
- NM's local refinement ability (on accurate signal)

## Comparison to Previous Best

| Metric | Previous Best | This Experiment |
|--------|--------------|-----------------|
| Score | 1.1247 (robust_fallback) | **1.1688** |
| Time | 57.2 min | 58.4 min |
| Delta | - | **+0.0441** |

## Files

- `optimizer_with_polish.py`: TemporalFidelityWithPolishOptimizer
- `run_polish.py`: Run script with `--final-polish-maxiter` parameter
- `optimizer.py`: Base TemporalFidelityOptimizer (no polish)
- `run.py`: Base run script
- `correlation_test.py`: Feasibility test

## Recommendation

**Adopt 40% timesteps + 8 NM polish as the new production optimizer.**

This achieves:
- Best score ever in-budget (1.1688)
- Significant 2-source improvement
- Within time budget (58.4 min)

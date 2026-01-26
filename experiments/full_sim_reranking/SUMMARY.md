# Experiment Summary: full_sim_reranking

## Metadata
- **Experiment ID**: EXP_CANDIDATE_RERANKING_001
- **Worker**: W1
- **Date**: 2026-01-25
- **Algorithm Family**: candidate_selection_v2

## Status: ABORTED (Already Implemented)

## Objective
After CMA-ES (40% fidelity), re-evaluate top-5 candidates with full simulation to select true best for polish.

## Why Aborted

**This functionality is ALREADY IMPLEMENTED in the baseline optimizer.**

### Evidence from Code

From `experiments/early_timestep_filtering/optimizer_with_polish.py`:

```python
# Lines 345-364: Evaluate candidates on FINE grid with FULL timesteps
candidates_raw = []
for pos_params, rmse_coarse, init_type in refined_solutions:
    # ... full timestep evaluation ...
    final_rmse = compute_optimal_intensity_*src(...)
    candidates_raw.append((sources, full_params, final_rmse, ...))

# Lines 366-369: Find best candidate based on FULL timestep RMSE
if self.final_polish_maxiter > 0 and candidates_raw:
    # Find best candidate
    best_idx = min(range(len(candidates_raw)), key=lambda i: candidates_raw[i][2])
```

### Baseline Workflow Already Does This

1. CMA-ES with 40% timesteps (coarse exploration)
2. **Evaluate ALL refined candidates with FULL timesteps** ← ALREADY DONE
3. **Select best based on FULL-timestep RMSE** ← ALREADY DONE
4. NM polish on the best candidate with full timesteps

## Why Re-Ranking Would Be Redundant

### 0.95+ Correlation Already Ensures Correct Ranking

Quote from ITERATION_LOG.md:
> "40% timesteps provides 2.5x speedup with minimal accuracy loss (RMSE correlation 0.95+)"

Even if we didn't do full-sim evaluation, the coarse ranking would be:
- 95%+ likely to identify the correct best candidate
- Within 5% error margin for borderline cases

But we DO full-sim evaluation anyway, making this moot.

### Additional Re-Ranking Would Add Overhead

If we evaluated top-5 instead of currently-evaluated set:
- Each full-sim evaluation costs ~0.4 seconds per sample
- 5 evaluations × 80 samples × 0.4s = 160 seconds overhead
- Zero benefit since baseline already does this

## Technical Analysis

### Current Candidate Pipeline

```
CMA-ES (40% timesteps)
    ↓
Top-N refined solutions (coarse RMSE)
    ↓
Full-timestep evaluation (accurate RMSE)  ← ALREADY IMPLEMENTED
    ↓
Select best (by accurate RMSE)           ← ALREADY IMPLEMENTED
    ↓
NM polish (full timesteps)
    ↓
Return polished best
```

### What the Experiment Proposed

```
CMA-ES (40% timesteps)
    ↓
Top-5 candidates (coarse RMSE)
    ↓
Full-sim re-ranking (accurate RMSE)      ← DUPLICATE
    ↓
Polish best
```

## Algorithm Family Status

- **candidate_selection_v2**: This specific experiment is a DUPLICATE
- The family is not exhausted - but this particular approach adds nothing

## Recommendations

1. **Do NOT implement** - functionality already exists
2. **Verify baseline code** before proposing similar experiments
3. **The baseline is already optimal** for candidate selection

## Conclusion

The full_sim_reranking experiment was aborted because it proposes functionality that is ALREADY IMPLEMENTED in the baseline optimizer. The baseline already evaluates all refined candidates with full timesteps before selecting the best for polish. No additional re-ranking is needed or would provide benefit.

# Experiment Summary: top3_ensemble_averaging

## Status: FAILED

## Experiment ID: EXP_TOP3_ENSEMBLE_001
## Worker: W1
## Date: 2026-01-26

## Hypothesis
Averaging the positions of the top-3 candidates (by RMSE) might create a more robust solution by:
- Reducing variance from optimization noise
- Finding a more central/stable solution
- Benefiting from multiple starting point diversity

## Approach
1. After CMA-ES optimization, get the candidate pool
2. Sort by RMSE and take top-3
3. For 1-source: average (x, y) positions
4. For 2-source: sort sources by x-coordinate for consistent ordering, then average corresponding sources
5. Compute optimal intensity for the averaged position
6. Add ensemble solution as additional candidate

## Results

| Metric | Value | Baseline | Delta |
|--------|-------|----------|-------|
| **Score** | 1.1129 | 1.1688 | **-4.8%** |
| Projected 400 min | 43.4 | 58.4 | -25.7% |
| RMSE 1-source | 0.1508 | - | - |
| RMSE 2-source | 0.2110 | - | - |
| Ensemble in final | 29/80 | - | - |

## Why It Failed

### 1. Averaging Blurs Precise Locations
The top candidates often represent different local optima or slightly different positions. Averaging them creates a position that is neither here nor there:
- If candidates are clustered around the true solution, averaging helps slightly
- If candidates represent different basins, averaging creates a position between basins (worse)

### 2. Source Permutation Issues for 2-Source
Even with x-coordinate sorting for consistent ordering, the 2-source case has issues:
- Two sources at similar x-coordinates may be swapped across candidates
- Averaging "source 1" across candidates may mix different actual sources
- This is especially problematic when sources are at similar x-coordinates

### 3. Ensemble Adds to Candidate Pool, Not Replaces
The ensemble was added as an additional candidate and included in 29/80 final filtered results. However:
- It often had similar or worse RMSE than existing candidates
- When included, it may have displaced a better diverse candidate
- The dissimilarity filtering didn't benefit from ensemble presence

### 4. Post-Processing Too Late
By the time we do post-processing, the CMA-ES and NM polish have already converged. Averaging at this stage:
- Doesn't help exploration (already done)
- Doesn't improve refinement (positions already refined)
- Just adds noise to already-optimized solutions

## Conclusion
**Ensemble averaging of candidate positions is NOT beneficial for this problem.**

The candidates returned by CMA-ES optimization are already reasonably good solutions. Averaging their positions:
- Destroys the precise refinement achieved by NM polish
- Mixes potentially distinct local optima
- Adds computational overhead (1-2 extra simulations) for no benefit

## Recommendation
**Do NOT use position averaging as post-processing.** Keep candidates as discrete solutions from optimization. If diversity is needed, rely on the existing dissimilarity filtering.

## Files
- `optimizer.py`: Implementation with ensemble averaging
- `run.py`: Run script with configurable top-k parameter
- `STATE.json`: Experiment state and results

# Multi-Candidate Perturbation Experiment

## Status: INCONCLUSIVE

## Hypothesis
Current perturbation only explores around best candidate. 2nd-best candidate may be in a different basin with better local optimum.

## Configuration
- Perturb top 2 candidates with 1 perturbation each (2 total perturbations)
- Standard parameters: sigma 0.18/0.22, 8 NM polish, 0.05 perturbation scale

## Results

| Run | Config | Score | RMSE | Projected Time | Status |
|-----|--------|-------|------|----------------|--------|
| 1 | perturb_top_n=2, n_perturb=1 | 1.1526 | 0.1369 | 150.4 min | OVER BUDGET |

## Analysis

### Promising Signal
- Score: 1.1526 (vs baseline 1.1468, delta +0.0058)
- RMSE: 0.1369 (lower is better)
- Sample 7 improved from 0.5374 (best_of_2_seeds) to 0.3242

### But Over Budget
- Time: 150.4 min (2.5x over 60 min budget)
- Machine is running 2-3x slower than baseline machine
- Cannot validate if this works within budget on production machine

### Comparison to best_of_2_seeds
| Experiment | Score | Time | Sample 7 RMSE |
|------------|-------|------|---------------|
| best_of_2_seeds | 1.1162 | 133.9 min | 0.5374 |
| multi_candidate_perturbation | 1.1526 | 150.4 min | 0.3242 |

Multi-candidate perturbation significantly outperforms multi-seed approach.

## Key Finding
**Perturbing 2nd-best candidate shows promise.** The score improvement (+0.0058) suggests the 2nd-best candidate's basin sometimes leads to better solutions than the best candidate's basin.

However, this adds ~10 min overhead (150 vs 133 min scaled by machine speed), which may push it over budget on production machines.

## Conclusion
INCONCLUSIVE due to machine constraints. Shows promise but cannot validate within 60-minute budget. Recommend testing on faster machine.

## Recommendation
If production machine is faster, this approach may be viable. The perturbation of 2nd-best candidate adds overhead but potentially improves accuracy.

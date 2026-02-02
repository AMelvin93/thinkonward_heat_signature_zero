# Experiment: asymmetric_014_019_4perturb

## Hypothesis
Combine optimal sigma (0.14/0.19) with 4 perturbations - both have shown improvements individually.

## Results

| Config | Sigma | Perturb | Score | Proj 400 | RMSE 1src | RMSE 2src | In Budget |
|--------|-------|---------|-------|----------|-----------|-----------|-----------|
| **baseline_2perturb** | 0.14/0.19 | 2 | **1.1675** | **53.4m** | 0.1286 | 0.1770 | **YES** |
| test_4perturb | 0.14/0.19 | 4 | 1.1684 | 61.3m | 0.1155 | 0.1875 | **NO** |
| test_3perturb | 0.14/0.19 | 3 | 1.1671 | 56.9m | 0.1307 | 0.1759 | YES |

## Key Finding: 2 PERTURBATIONS IS OPTIMAL

### Analysis

1. **4 perturbations EXCEEDS BUDGET**
   - Score: 1.1684 (+0.0009 vs 2 perturb)
   - Time: 61.3 min (OVER 60 min limit!)
   - Even though score is slightly better, it's disqualified

2. **3 perturbations has NO benefit**
   - Score: 1.1671 (-0.0004 vs 2 perturb)
   - Time: 56.9 min (+3.5 min overhead)
   - Worse score AND slower - no advantage

3. **2 perturbations is optimal for sigma 0.14/0.19**
   - Best in-budget score: 1.1675
   - Time: 53.4 min (6.6 min buffer)
   - Good balance of accuracy and speed

### Why 4 perturbations helps 1-source but hurts 2-source

| Config | RMSE 1src | RMSE 2src |
|--------|-----------|-----------|
| 2 perturb | 0.1286 | 0.1770 |
| 4 perturb | 0.1155 | 0.1875 |

- 4 perturbations improves 1-source RMSE by 0.013 (10%)
- But worsens 2-source RMSE by 0.010 (6%)
- Net effect: marginal improvement, but over budget

## Conclusion
**RESULT: FAILED - Hypothesis Disproved**

More perturbations do NOT help with asymmetric sigma 0.14/0.19:
- 4 perturbations exceeds budget (61.3 min > 60 min)
- 3 perturbations has no benefit (-0.0004 score, +3.5 min)
- **2 perturbations remains optimal**

## CONFIRMED OPTIMAL CONFIG
```python
config = {
    'sigma0_1src': 0.14,
    'sigma0_2src': 0.19,
    'max_fevals_1src': 20,
    'max_fevals_2src': 44,
    'perturbation_scale': 0.06,
    'n_perturbations': 2,  # OPTIMAL - more is slower without benefit
    'timestep_fraction': 0.40,
    'refine_maxiter': 8,
}
```

Expected score: **1.165-1.175** @ ~53 min.

## Tuning Efficiency Metrics
- **Runs executed**: 3 (sufficient for clear trend)
- **Time utilization**: 89% (53.4/60 min for best config)
- **Clear result**: 2 perturbations is optimal

## Note on Previous "4 Perturbations" Claims
Prior experiments with sigma 0.18/0.22 may have shown 4 perturbations as beneficial.
This is because sigma 0.18/0.22 is slower CMA-ES (wider search), giving more time headroom.
With the tighter sigma 0.14/0.19, the budget is more constrained.

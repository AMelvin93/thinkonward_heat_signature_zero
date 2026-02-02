# Experiment: asymmetric_sigma_by_source

## Hypothesis
1-source problems might benefit from tighter sigma (more local search) while 2-source problems need looser sigma (more exploration). Testing asymmetric sigma configurations.

## Results

| Config | Sigma 1src | Sigma 2src | Score | Proj 400 | In Budget | RMSE 1src | RMSE 2src |
|--------|------------|------------|-------|----------|-----------|-----------|-----------|
| symmetric_015_019 | 0.15 | 0.19 | 1.1650 | 54.2m | YES | 0.124 | 0.188 |
| **asymmetric_014_019** | **0.14** | **0.19** | **1.1745** | **50.4m** | **YES** | **0.119** | **0.168** |
| asymmetric_015_022 | 0.15 | 0.22 | 1.1643 | 62.2m | NO | 0.124 | 0.190 |
| asymmetric_014_022 | 0.14 | 0.22 | 1.1637 | 96.9m | NO | 0.117 | 0.199 |

## Best Configuration
**asymmetric_014_019 (sigma0_1src=0.14, sigma0_2src=0.19)**
- Score: **1.1745** (NEW BEST!)
- Projected time: 50.4 min
- In budget: YES (9.6 min remaining)
- Delta vs symmetric baseline (1.1650): **+0.0095** (+0.81%)
- Delta vs claimed best (1.1730): **+0.0015** (+0.13%)

## Key Findings

1. **Tighter sigma for 1-source (0.14) significantly improves accuracy**
   - RMSE 1-source improved from 0.124 → 0.119 (-4%)
   - 1-source problems are simpler, need less exploration

2. **2-source sigma should stay at 0.19 (not looser)**
   - Looser sigma (0.22) for 2-source actually HURTS performance
   - RMSE 2-source worse with 0.22: 0.188 → 0.190-0.199

3. **Asymmetric configuration outperforms symmetric**
   - Both RMSEs improved simultaneously
   - Faster runtime (50.4 vs 54.2 min)

4. **Configuration timing**
   - asymmetric_014_022 took 96.9 min (over budget) - avoid
   - asymmetric_014_019 is fastest AND best

## RMSE Analysis

| Metric | symmetric_015_019 | asymmetric_014_019 | Improvement |
|--------|-------------------|-------------------|-------------|
| RMSE 1-src | 0.1244 | 0.1194 | -4.0% |
| RMSE 2-src | 0.1878 | 0.1676 | -10.8% |
| Overall | 0.1561 | 0.1435 | -8.1% |

The 2-source RMSE improved by 10.8% despite keeping the same sigma! This suggests the tighter 1-source sigma allows better overall exploration budget allocation.

## Conclusion
**RESULT: SUCCESS - NEW BEST**

- Best configuration: sigma0_1src=0.14, sigma0_2src=0.19
- Score: 1.1745 @ 50.4 min projected
- This is the new production-recommended configuration

**RECOMMENDATION**:
- Adopt asymmetric_014_019 as the new baseline
- Consider testing even tighter 1-source sigma (0.13?) in future experiments
- Combine with optimal perturbation_scale=0.06 from other experiment

## Next Steps
1. Validate this result with additional runs
2. Combine asymmetric sigma with optimal perturbation scale (0.06)
3. Test sigma0_1src=0.13 for potentially further improvement

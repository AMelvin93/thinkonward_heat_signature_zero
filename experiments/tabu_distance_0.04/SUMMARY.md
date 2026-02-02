# Experiment: tabu_distance_0.04

## Status: PARTIAL SUCCESS - Found better config but needs validation

## Hypothesis
Test larger tabu distance (0.04, 0.05) vs baseline 0.03. Larger distance may improve diversity of perturbation solutions.

## Results

| Config | Sigma | Tabu | Score | Time (min) | vs Baseline | Budget Remaining |
|--------|-------|------|-------|------------|-------------|------------------|
| tabu_004_sigma_014_019 | 0.14/0.19 | 0.04 | 1.1389 | 50.7 | -0.0075 | 9.3 min |
| **tabu_004_sigma_018_022** | **0.18/0.22** | **0.04** | **1.1535** | 53.1 | **+0.0071** | 6.9 min |
| tabu_005_sigma_018_022 | 0.18/0.22 | 0.05 | 1.1435 | 53.0 | -0.0029 | 7.0 min |

**Baseline**: 1.1464 @ 51.2 min (sigma 0.18/0.22, tabu_distance=0.03)

## Key Findings

### 1. tabu_distance=0.04 with sigma 0.18/0.22 Beats Baseline!
- Score: 1.1535 vs 1.1464 baseline (+0.0071)
- Time: 53.1 min (within budget with 6.9 min remaining)
- RMSE 1src: 0.1223, RMSE 2src: 0.1849

### 2. sigma 0.14/0.19 Continues to Underperform
- With tabu 0.04: 1.1389 (-0.0075 vs baseline)
- Confirms validation experiment finding that this sigma is not reliable

### 3. tabu_distance=0.05 is Too Large
- Score: 1.1435 (-0.0029 vs baseline)
- Larger distance doesn't help; 0.04 is optimal

### 4. Optimal Configuration Found
```python
config = {
    'sigma0_1src': 0.18,
    'sigma0_2src': 0.22,
    'tabu_distance': 0.04,  # Increased from 0.03 baseline
    'max_fevals_1src': 20,
    'max_fevals_2src': 44,
    'n_perturbations': 2,
    'perturbation_scale': 0.05,
    ...
}
```

## RMSE Analysis

| Config | RMSE 1src | RMSE 2src | Avg Candidates |
|--------|-----------|-----------|----------------|
| tabu_004_sigma_014_019 | 0.1138 | 0.1920 | 2.73 |
| **tabu_004_sigma_018_022** | **0.1223** | **0.1849** | **2.86** |
| tabu_005_sigma_018_022 | 0.1172 | 0.2084 | 2.84 |

The best config has:
- Slightly higher 1-src RMSE (0.1223 vs 0.1138)
- Lower 2-src RMSE (0.1849 vs 0.1920)
- More candidates (2.86 vs 2.73) - better diversity

## Why tabu_distance=0.04 Works Better

The tabu memory prevents revisiting explored regions. With distance=0.04 (vs 0.03):
- Perturbations explore more diverse regions
- Less overlap between explored areas
- Better chance of finding distinct local optima
- Slightly more candidates generated (2.86 vs baseline ~2.75)

## Caveats

This is a **single run** result. Given the high variance observed earlier (std ~0.008), this +0.0071 improvement may not be statistically significant. However:
- The improvement direction is consistent (0.04 > 0.03 > 0.05)
- The sigma effect is consistent with other experiments

## Recommendation

**NEEDS VALIDATION**: Run 3-5 more tests with tabu_distance=0.04 and sigma 0.18/0.22 to confirm this finding is reproducible.

If validated, this becomes the new production config:
- Change tabu_distance from 0.03 to 0.04
- Keep sigma 0.18/0.22
- Expected improvement: ~+0.007 score

## Tuning Efficiency Metrics
- **Runs executed**: 3 (systematic exploration)
- **Time utilization**: 88.5% (53.1/60 min for best config)
- **Parameter space explored**:
  - tabu_distance: [0.04, 0.05]
  - sigma: [0.14/0.19, 0.18/0.22]

## Budget Analysis
All configurations within budget (50.7-53.1 min projected for 400 samples).

---
**Worker**: W2
**Completed**: 2026-02-02
**Runs**: 3 (systematic tuning)
**Result**: POTENTIAL IMPROVEMENT FOUND - needs validation

# Asymmetric NM Iterations Experiment

## Status: INCONCLUSIVE

## Hypothesis
2-source problems have more complex landscapes and may benefit from more NM polish iterations than 1-source problems.

## Configuration
- Baseline: 8 NM iterations for both 1-src and 2-src
- Test: 6 NM iterations for 1-src, 10 for 2-src

## Results

| Run | NM Config | Samples | Score | RMSE 1-src | RMSE 2-src | Projected Time |
|-----|-----------|---------|-------|------------|------------|----------------|
| 1 | 8/8 | 20 (1-src only) | 1.1507 | 0.125 | N/A | 122.2 min |
| 2 | 6/10 | 40 | 1.1479 | 0.137 | 0.216 | 142.9 min |
| 3 | 8/8 | 40 | 1.1355 | 0.140 | 0.203 | 163.6 min |

## Analysis

### Direct Comparison (Runs 2 vs 3, same 40 samples):
- **Asymmetric (6/10)**: Score 1.1479, 1-src RMSE=0.137, 2-src RMSE=0.216
- **Baseline (8/8)**: Score 1.1355, 1-src RMSE=0.140, 2-src RMSE=0.203

### Observations:
1. **Asymmetric scored better overall** (1.1479 vs 1.1355) due to slightly better 1-src performance
2. **Baseline had better 2-src RMSE** (0.203 vs 0.216), contradicting hypothesis
3. **1-src dominates** the overall score (32 samples vs 8 samples)
4. **All runs over budget** - this machine runs 2-3x slower than baseline machine

### Why Inconclusive:
- Machine too slow to test within 60-minute budget
- Cannot draw meaningful conclusions about production performance
- The hypothesis was about 2-src benefiting from more polish, but the data shows mixed results

## Key Findings
1. Reducing 1-src NM from 8 to 6 may slightly improve 1-src accuracy (counter-intuitive)
2. Increasing 2-src NM from 8 to 10 did NOT improve 2-src accuracy
3. The fixed 8 NM iterations remain the safe choice for 2-src problems

## Recommendation
**Do not pursue asymmetric NM allocation.** The evidence does not support the hypothesis. The baseline configuration of 8 NM iterations for all problems remains optimal based on prior experiments.

## Baseline Comparison
- Baseline: 1.1468 @ 54.2 min (perturbation_plus_verification)
- Best this experiment: 1.1479 @ 142.9 min (over budget)

## Conclusion
Experiment INCONCLUSIVE due to machine constraints. However, the limited data does not support the hypothesis that 2-src problems benefit from more NM polish iterations. Marking the `polish_allocation` family as exhausted.

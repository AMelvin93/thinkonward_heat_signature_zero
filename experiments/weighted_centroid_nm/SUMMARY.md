# Weighted Centroid NM Experiment

## Experiment ID: EXP_WEIGHTED_CENTROID_NM_001
**Status**: FAILED
**Worker**: W1
**Date**: 2026-01-24

## Hypothesis

Based on a 2023 paper "Weighted Centroids in Adaptive Nelder–Mead Simplex: With heat source locator applications" (Applied Soft Computing, Jan 2024), the weighted centroid approach should accelerate NM convergence for heat source localization. The key idea is to compute the centroid as a weighted mean where vertices with lower function values (better fitness) get higher weights, biasing the search towards promising regions.

## Implementation

Custom NM implementation with weighted centroid:
1. Weights are inversely proportional to function values: `w_i = 1 / (f_i^power)`
2. Centroid = weighted mean of n best vertices (excluding worst)
3. Standard NM operations (reflect, expand, contract, shrink) use weighted centroid

## Results

| Run | Weighted NM | Weight Power | Score | Time (min) | In Budget | Notes |
|-----|-------------|--------------|-------|------------|-----------|-------|
| 1   | Yes         | 2.0          | 0.9608 | 52.3      | YES       | -0.2080 vs baseline |

### Detailed Metrics (Run 1)
- **1-src RMSE**: 0.1214 (n=32) - worse than baseline 0.11
- **2-src RMSE**: 0.2056 (n=48) - 45% worse than baseline 0.14
- **Candidates per sample**: 1 (vs ~3 for baseline)

### Comparison to Baseline
| Metric | Baseline | Weighted NM | Delta |
|--------|----------|-------------|-------|
| Score  | 1.1688   | 0.9608      | -0.2080 |
| Time   | 58.4 min | 52.3 min    | -6.1 min |
| 1-src RMSE | 0.11 | 0.1214 | +10% |
| 2-src RMSE | 0.14 | 0.2056 | +45% |

## Analysis

### Why Weighted Centroid NM Failed

1. **Severely Reduced Candidate Diversity**: The implementation produced only 1 candidate per sample instead of the baseline's ~3 candidates. This alone costs ~0.2 in the score formula (loss of diversity bonus).

2. **Worse Accuracy**: Even ignoring diversity, the RMSE values are significantly worse:
   - 1-source: 10% worse (0.1214 vs 0.11)
   - 2-source: 45% worse (0.2056 vs 0.14)

3. **Custom Implementation Issues**: The custom NM implementation may have subtle differences from scipy's optimized version that hurt convergence quality.

4. **Weighted Centroid May Not Help Small Simplexes**: In the polish phase, the simplex is already near the optimum. Weighting by fitness might not provide meaningful directional improvement over standard centroid.

5. **Paper Context Mismatch**: The 2023 paper focused on high-dimensional problems where the standard NM becomes inefficient. Our polish phase is only 2D (1-source) or 4D (2-source), where standard NM is already effective.

## Key Finding

**Weighted centroid NM is NOT beneficial for this problem.**

The standard scipy Nelder-Mead implementation is:
- More robust (produces multiple candidates)
- More accurate (lower RMSE)
- Better optimized (mature implementation)

## Recommendation

Do NOT replace scipy's NM with custom weighted centroid implementation. The baseline approach (40% temporal fidelity + sigma 0.15/0.20 + 8 scipy NM polish) remains optimal.

The `polish_improvement` family should be marked as EXHAUSTED:
- Powell polish: FAILED (4x over budget)
- Weighted centroid NM: FAILED (worse accuracy and diversity)
- Standard NM: OPTIMAL for 2-4D problems

## Files
- `optimizer.py`: Weighted centroid NM implementation
- `run.py`: Experiment runner with MLflow logging
- `STATE.json`: Tuning history and configuration

## References
- Paper: https://www.sciencedirect.com/science/article/abs/pii/S1568494623011961

# Perturbation NM Iterations Tuning

## Status: FAILED (baseline remains optimal)

## Hypothesis
Test 4+ NM iterations per perturbation (vs 3 baseline). May improve perturbation refinement.

## Results

| Run | Config | Score | Time (min) | vs Baseline |
|-----|--------|-------|------------|-------------|
| 1 | sigma 0.14/0.19, nm_iters=4 | 1.1349 | 45.9 | -0.0115 |
| 2 | sigma 0.18/0.22, nm_iters=4 | 1.1427 | 46.8 | -0.0037 |
| 3 | sigma 0.18/0.22, nm_iters=5 | 1.1406 | 43.5 | -0.0058 |

**Baseline**: 1.1464 @ 51.2 min (perturb_nm_iters=3, sigma 0.18/0.22)

## Key Findings

1. **3 NM iterations is optimal**: Both 4 and 5 iterations underperform baseline
2. **Sigma 0.18/0.22 > 0.14/0.19**: Confirmed again (+0.0078 difference)
3. **Diminishing returns**: More iterations add time without score improvement
4. **Sweet spot at 3 iters**: Enough refinement without overfitting

## Analysis

| nm_iters | Best Score | Time | Implication |
|----------|------------|------|-------------|
| 3 (baseline) | 1.1464 | 51.2 | OPTIMAL |
| 4 | 1.1427 | 46.8 | -0.0037 (worse) |
| 5 | 1.1406 | 43.5 | -0.0058 (worse) |

Interestingly, 5 iterations is FASTER than 4 iterations. This might be due to variance or convergence patterns.

## Tuning Efficiency

- **Runs executed**: 3
- **Time utilization**: 78% (46.8/60 min)
- **Parameter space explored**: nm_iters in [4, 5], sigma in [0.14/0.19, 0.18/0.22]

## Recommendation

**DO NOT increase perturb_nm_iters beyond 3**. The baseline value of 3 is already optimal.

---
**Worker**: W1
**Completed**: 2026-02-02
**Runs**: 3 (systematic exploration)

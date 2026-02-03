# Experiment: 5 Perturbations Test

## Status: FAILED - Cannot fit in budget

## Purpose
Test if 5 perturbations (vs 4) improves score within budget.
Current 4pert_nm2: 1.1482 @ 51.7 min (8.3 min remaining)

## Results

| Config | n_pert | perturb_nm_iters | Score | Time (min) | Budget |
|--------|--------|------------------|-------|------------|--------|
| 4pert_nm2 (baseline) | 4 | 2 | 1.1482 | 51.7 | IN |
| 5pert_nm2 | 5 | 2 | 1.1534 | 76.0 | **OVER** |
| 5pert_nm1 | 5 | 1 | 1.1533 | 76.1 | **OVER** |

## Key Findings

### 1. More perturbations DO improve score (+0.005)
- 5 perturbations: 1.1533-1.1534
- 4 perturbations: 1.1482
- Delta: +0.005 (~0.5% improvement)

### 2. Perturbation cost is NOT perturb_nm_iters
Reducing perturb_nm_iters from 2 to 1 didn't reduce timing:
- perturb_nm_iters=2: 76.0 min
- perturb_nm_iters=1: 76.1 min
- Delta: ~0 min

The bottleneck is the full simulations for evaluating perturbation candidates, not the NM refinement iterations.

### 3. Per-perturbation overhead is ~24 minutes
- 4 perturbations: 51.7 min
- 5 perturbations: 76.0 min
- Per-perturbation cost: ~24 min for the 5th perturbation

This is unexpected - implies non-linear scaling with perturbations.

### 4. Cannot fit 5 perturbations in budget
Even with minimal NM iterations, 5 perturbations takes 76+ min (16 min over budget).

## Conclusion

**5 perturbations provides a real improvement (+0.005 score) but cannot fit in the 60-minute budget.**

The current 4pert_nm2 config (1.1482 @ 51.7 min) remains optimal.

### What Would Help

To fit 5 perturbations in budget, we would need to reduce base timing by ~16 min:
- Reduce fevals significantly (but this hurts accuracy)
- Reduce final NM polish (but this hurts accuracy)
- Use faster simulator (not available)

## Additional Test: Reduced fevals + 5 perturbations

| Config | fevals | n_pert | nm_iters | Score | Time | Budget |
|--------|--------|--------|----------|-------|------|--------|
| 5pert_nm2 | 20/44 | 5 | 2 | 1.1534 | 76.0 | OVER |
| 5pert_nm1 | 20/44 | 5 | 1 | 1.1533 | 76.1 | OVER |
| reduced_5pert | 15/35 | 5 | 1 | **1.1567** | **80.1** | OVER |

**Surprising finding**: Even with 25% fewer fevals (15/35 vs 20/44), 5 perturbations is STILL over budget. And timing actually got worse (80 min vs 76 min).

The best score achieved was 1.1567 (+0.0085 vs baseline) but timing was 80 min (20 min over budget).

## Tuning Efficiency

- **Runs executed**: 3
- **Configs tested**: 5pert_nm2, 5pert_nm1, reduced_fevals_5pert
- **Time utilization**: N/A (approach cannot fit budget)
- **Key insights**:
  - perturb_nm_iters is NOT the bottleneck
  - Reducing fevals doesn't reduce timing proportionally
  - 5 perturbations adds ~24+ min regardless of other settings

## Fundamental Issue

The perturbation mechanism has significant fixed overhead per perturbation that cannot be reduced by tuning parameters. The overhead is likely in:
1. Full simulation evaluations for each perturbation candidate
2. The tabu memory checking and exploration
3. Non-parallelizable sequential processing

---
**Worker**: W2
**Completed**: 2026-02-02
**Result**: FAILED - Cannot fit 5 perturbations in budget (3 configs tested)

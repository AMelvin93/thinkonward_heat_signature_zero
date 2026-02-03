# Experiment: faster_nm_4pert

## Status: FAILED - No 4-pert config fits budget on this system

## Hypothesis
Reducing refine_maxiter or perturb_nm_iters might allow 4 perturbations to fit within budget.

## Results

| Config | refine_maxiter | perturb_nm_iters | Score | Time (proj 400) | Budget |
|--------|----------------|------------------|-------|-----------------|--------|
| **baseline nm8** | 8 | 2 | 1.1546 | 70.9 min | OVER |
| nm6_4pert | 6 | 2 | 1.1491 | 69.9 min | OVER |
| nm4_4pert | 4 | 2 | **1.1587** | 74.0 min | OVER |
| nm6_4pert_nm1 | 6 | 1 | 1.1528 | 69.2 min | OVER |
| **no_perturb** | 8 | N/A | 1.1367 | 56.3 min | **IN** |

## Key Findings

### 1. All 4-pert configs exceed budget
Even with aggressively reduced NM iterations:
- nm6: 69.9 min (still 10 min over)
- nm4: 74.0 min (worse!)
- nm6 + nm1: 69.2 min (still 9 min over)

### 2. Surprising result: nm4 is SLOWER and BETTER
The nm4_4pert config (refine_maxiter=4) achieved:
- Higher score: 1.1587 (best of all!)
- But slower time: 74.0 min

This is counterintuitive - fewer NM iterations should be faster.
Possible explanations:
- Run-to-run variance
- Different convergence behavior
- System variability

### 3. No amount of NM reduction brings 4-pert into budget
The perturbation overhead (~14-15 min) cannot be reduced enough
by NM iteration changes alone.

## Timing Analysis

| Component | Time Contribution |
|-----------|-------------------|
| Base CMA-ES + NM (no perturb) | 56.3 min |
| Perturbation overhead (4 perturb) | +14.6 min |
| **Total 4-pert** | 70.9 min |

Even with reduced NM iterations, perturbation overhead remains ~13-18 min.
To fit 4-pert in 60 min, we'd need to reduce base time to <46 min,
which would require massive cuts to CMA-ES fevals.

## Conclusion

**RESULT: FAILED - Cannot fit 4 perturbations in budget on this system**

The only in-budget option remains the **no-perturb config** at 56.3 min (score 1.1367).

## Recommendation

For final submission:
1. **Safe option**: Use no-perturb config (guaranteed 56.3 min)
2. **Risk option**: Use 4-pert nm2 and hope G4dn.2xlarge is faster

The 4-pert config (score ~1.155) is very close to Top 10 (1.1585),
but timing risk is significant.

## Tuning Efficiency Metrics
- **Runs executed**: 3 (systematic exploration of NM variations)
- **Time utilization**: N/A (all configs over budget)
- **Parameter space explored**: refine_maxiter ∈ [4, 6, 8], perturb_nm_iters ∈ [1, 2]

---
**Worker**: W1
**Date**: 2026-02-02
**Status**: FAILED - 4-pert cannot fit budget

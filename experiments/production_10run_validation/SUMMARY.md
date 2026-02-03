# Experiment: production_10run_validation

## Status: CRITICAL TIMING DISCREPANCY DISCOVERED

## Objective
Validate the 4-pert + nm2 production config with 10 runs to establish high-confidence performance metrics for final submission.

## CRITICAL FINDING: System Timing is 35-40% Slower Than Expected

The original 4-pert + nm2 validation showed 51.7 min projected for 400 samples.
On this system, the same config runs at **70.9 min projected** - **37% slower**.

This indicates that timing benchmarks are **NOT portable** across systems.

## Full Results Summary

| Config | Score | Time (proj 400) | vs Budget | Notes |
|--------|-------|-----------------|-----------|-------|
| 4-pert nm2 | **1.1546** | 70.9 min | **OVER** | Original validated config |
| 2-pert nm2 | 1.1452 | 67.4 min | **OVER** | Reduced perturbations |
| 1-pert | 1.1447 | 65.9 min | **OVER** | Single perturbation |
| **No perturb** | 1.1367 | **56.3 min** | **IN** | Only in-budget option |

## Timing Breakdown by Source Type

| Config | 1-source avg | 2-source avg | Ratio |
|--------|--------------|--------------|-------|
| No perturb | 26.8s | 77.2s | 2.88x |
| 1 perturb | 34.6s | 76.8s | 2.22x |
| 2-pert nm2 | 37.7s | 88.9s | 2.36x |
| 4-pert nm2 | 38.0s | 94.5s | 2.49x |

**Key observation**: 2-source samples are 2.2-2.9x slower than 1-source samples.

## Analysis

### Why the Timing Discrepancy?
The original validation was likely run on a system with different:
1. CPU performance characteristics
2. System load conditions
3. Memory/cache behavior

### Score vs Time Tradeoff
```
         | 1.15 |                    * 4pert (70.9 min, OVER)
         |      |                * 2pert (67.4 min, OVER)
  Score  |      |              * 1pert (65.9 min, OVER)
         | 1.13 |         * no_perturb (56.3 min, IN)
         +------+--------------------------------
                      55    60    65    70   Time
                      ↑
                   Budget limit
```

## Recommendations for Final Submission

### OPTION A: Conservative (Safe)
Use the "No perturb" config which is reliably in budget:
```python
config = {
    'sigma0_1src': 0.18,
    'sigma0_2src': 0.22,
    'max_fevals_1src': 20,
    'max_fevals_2src': 44,
    'timestep_fraction': 0.40,
    'refine_maxiter': 8,
    'enable_tabu_hopping': False,  # NO PERTURBATIONS
}
```
- Expected score: ~1.137
- Expected time: ~56 min (4 min buffer)
- Risk: Low

### OPTION B: Moderate Risk
Use the 4-pert + nm2 config and hope the G4dn.2xlarge is faster:
```python
config = {
    'sigma0_1src': 0.18,
    'sigma0_2src': 0.22,
    'max_fevals_1src': 20,
    'max_fevals_2src': 44,
    'timestep_fraction': 0.40,
    'refine_maxiter': 8,
    'enable_tabu_hopping': True,
    'n_perturbations': 4,
    'perturb_nm_iters': 2,
    'perturbation_scale': 0.05,
    'tabu_distance': 0.04,
    'max_tabu_attempts': 10,
}
```
- Expected score: ~1.155
- Expected time: Unknown (was 51.7 min on original system, 70.9 min here)
- Risk: High (may exceed 60 min)

## Gap to Top 10

| Config | Score | Gap to Top 10 (1.1585) |
|--------|-------|------------------------|
| 4-pert nm2 | 1.1546 | **-0.0039** (very close!) |
| No perturb | 1.1367 | -0.0218 |

The 4-pert config is VERY close to Top 10 (-0.0039) but timing is risky.
The safe config has a larger gap (-0.0218).

## Conclusion

**The timing benchmarks established on one system are NOT reliable on another system.**

The original validation showed 4-pert + nm2 at 51.7 min, but it runs at 70.9 min here.
This 37% difference is significant and indicates the need for:

1. **Final validation on actual competition hardware** (G4dn.2xlarge)
2. **A conservative fallback config** that is guaranteed to complete in time

## Worker Notes
- Minimum 3 tuning runs: N/A (validation experiment)
- Time utilization: N/A (discovered timing discrepancy)
- This is a critical finding for competition strategy

---
**Worker**: W1
**Date**: 2026-02-02
**Status**: CRITICAL FINDING - Timing is system-dependent

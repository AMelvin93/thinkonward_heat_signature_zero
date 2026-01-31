# Position-Only Perturbation Experiment

## Status: ABORTED - REDUNDANT

## Hypothesis
Since VarPro (Variable Projection) handles intensity optimally, only perturb position parameters (x,y) to escape local position minima without disrupting optimal intensity.

## Key Finding: ALREADY IMPLEMENTED

**The current optimizer already implements exactly what this experiment proposes.**

### Evidence from Code Analysis

1. **Perturbation is already position-only**:
   - In `perturbation_plus_verification/optimizer.py`, the `_generate_perturbations()` method (lines 278-286) operates on `pos_params`
   - `pos_params` contains only position coordinates: (x, y) for 1-source, (x1, y1, x2, y2) for 2-source
   - Intensity parameters are NEVER part of `pos_params`

2. **Intensity is already computed analytically via VarPro**:
   - `compute_optimal_intensity_1src()` (lines 88-107): Computes q = (Y_unit · Y_obs) / (Y_unit · Y_unit)
   - `compute_optimal_intensity_2src()` (lines 110-136): Solves 2x2 linear system for [q1, q2]
   - This is **Variable Projection**: optimize position, compute intensity analytically

3. **Workflow already matches experiment proposal**:
   - Step 1 "Modify perturbation to only add noise to x,y coordinates" → Already done
   - Step 3 "Re-optimize intensity analytically after position perturbation" → Already done via VarPro

## Prior Evidence

From coordination.json (2026-01-27):
> "W0: intensity_refinement_only NOT_FEASIBLE. CRITICAL INSIGHT: Baseline uses Variable Projection (VarPro) - CMA-ES only optimizes position, intensity computed analytically via closed-form least-squares. This is ALREADY globally optimal for intensity."

Related perturbation experiments that failed:
- `adaptive_perturbation_scale`: Score 1.1419 vs baseline 1.1464 (WORSE)
- `multi_round_perturbation`: Score 1.1372 vs baseline (WORSE)

## Conclusion

**This experiment is REDUNDANT** - the current best optimizer (`perturbation_plus_verification`) already implements:
- Position-only perturbation
- VarPro for optimal intensity computation
- The exact workflow described in the experiment proposal

There is nothing new to test.

## Tuning Efficiency Metrics
- **Runs executed**: 0 (aborted before running)
- **Reason for abort**: Code analysis showed experiment is already implemented
- **Time utilization**: N/A

## Family Status
`perturbation_targeted` family: EXHAUSTED
- Position-only perturbation is already the standard approach
- No variations remain to test

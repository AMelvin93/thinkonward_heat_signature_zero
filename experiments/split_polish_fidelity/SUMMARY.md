# Experiment Summary: split_polish_fidelity

## Metadata
- **Experiment ID**: EXP_SPLIT_POLISH_FINE_001
- **Worker**: W2
- **Date**: 2026-01-25
- **Algorithm Family**: fidelity_polish

## Objective
Test if initial polish iterations can use 40% fidelity (faster) while final iterations use 100% fidelity.

## Results Summary
- **Status**: **ABORTED** - Prior evidence invalidates hypothesis

## Why Aborted
This experiment is equivalent to the previously tested "progressive_polish_fidelity" experiment, which was also ABORTED based on prior evidence.

**Key Prior Finding:**
> "CRITICAL: NM polish must use FULL timesteps, not truncated (polishing proxy overfits to noise)"
> "2-source RMSE dropped from 0.21 to 0.14 (33% reduction) with full-timestep polish"

**Measured Impact:**
Using truncated timesteps during polish resulted in **-0.0346 score** penalty.

## Root Cause
The NM polish step refines source positions from CMA-ES's best solution. This refinement requires:
1. Accurate RMSE signal to guide descent direction
2. Truncated timesteps create a PROXY RMSE that differs from true RMSE
3. Polish moves toward proxy optimum, NOT true optimum
4. Result: Final positions are suboptimal when evaluated at full fidelity

## Conclusion
**Full timesteps are REQUIRED for all polish iterations.** Any form of fidelity reduction during polish (progressive, split, adaptive) will cause accuracy loss due to proxy overfitting.

The fidelity_polish family is EXHAUSTED.

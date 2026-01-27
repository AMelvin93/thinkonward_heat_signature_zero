# Experiment Summary: pinn_initialization

## Status: ABORTED (Family Exhausted)

## Experiment ID: EXP_PHYSICS_INFORMED_NN_INIT_001
## Worker: W1
## Date: 2026-01-26

## Hypothesis
A small pre-trained PINN could provide better initial guesses than triangulation, improving CMA-ES convergence.

## Why Aborted

### 1. PINN/neural_operator Family is EXHAUSTED

Prior experiment `pinn_inverse_heat_source` (EXP_PINN_DIRECT_001) conclusively showed:
- PINN requires efficient gradient computation
- The ADI solver blocks autodiff (implicit scheme)
- Finite difference gradients are too slow (99 min projected)
- Adjoint implementation failed (gradients 1e-6 of correct value)

Quote from SUMMARY:
> "ABORTED - PINN and all gradient-based approaches require efficient gradient computation. The neural_operator family should be marked EXHAUSTED."

### 2. Initialization Family is EXHAUSTED

Prior experiment `physics_informed_init` (EXP_PHYSICS_INIT_001) showed:
- Physics-based initialization DOES NOT improve over simple hottest-sensor approach
- Score: -0.0046 vs smart init (gradient init is WORSE)

Quote from SUMMARY:
> "The initialization family should be marked as EXHAUSTED for this problem domain."

### 3. Cross-Sample Learning is Impossible

Prior experiment `pretrained_nn_surrogate` showed:
- RMSE landscape is completely sample-specific
- Sample correlation: -0.167 average (NEGATIVE)
- No generalizable patterns exist across samples

## Recommendation
**Do NOT pursue PINN-based approaches.** The combination of:
1. No efficient gradient computation
2. Negative cross-sample correlation
3. Simple initialization already optimal

Makes any PINN approach fundamentally unviable for this problem.

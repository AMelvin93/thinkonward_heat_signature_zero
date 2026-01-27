# Experiment Summary: particle_filter_inversion

## Status: ABORTED (Family Exhausted)

## Experiment ID: EXP_PARTICLE_FILTER_001
## Worker: W1
## Date: 2026-01-26

## Hypothesis
Particle filter naturally handles non-Gaussian posteriors and can provide multiple solution candidates via Sequential Monte Carlo.

## Why Aborted

### Explicit Warning from EKF Experiment

The `extended_kalman_filter_inversion` experiment (EXP_KALMAN_ESTIMATION_001) explicitly recommended:

> **"Do NOT pursue particle_filter_inversion - Similar issues: requires many particles (simulations) to represent posterior well."**

### State Estimation Family is EXHAUSTED

EKF failed catastrophically:
- Score: 1.0058 vs baseline 1.1247 (10.6% worse)
- Time: 358 min vs baseline 57 min (6.3x slower)
- Required 233 simulations/sample vs CMA-ES's 20-56

### Why Particle Filter Would Be Even Worse

1. **Particle count requirement**: To adequately represent the posterior in 2-4D space, need hundreds to thousands of particles
2. **Each particle = 1 simulation**: Far more expensive than CMA-ES
3. **Resampling doesn't help**: Low effective sample size in peaked posteriors
4. **Local optima same as EKF**: Particles cluster around local modes

### Comparison

| Method | Sims/Sample | Score vs Baseline | Why It Fails |
|--------|-------------|-------------------|--------------|
| CMA-ES | 20-56 | Reference | Population-based, covariance adapted |
| EKF | 233 | -10.6% | Linearization, Jacobian cost |
| Particle Filter (est.) | 500-1000+ | Expected -15%+ | Many particles needed |

## Recommendation
**Do NOT pursue any state estimation approaches** (EKF, UKF, Particle Filter, Ensemble Kalman). They all require too many simulations and don't handle multi-modal landscapes well.

CMA-ES's population-based search with covariance adaptation is fundamentally superior for this problem.

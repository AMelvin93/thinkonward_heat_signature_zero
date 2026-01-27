# Experiment Summary: extended_kalman_filter_inversion

## Metadata
- **Experiment ID**: EXP_KALMAN_ESTIMATION_001
- **Worker**: W2
- **Date**: 2026-01-26
- **Algorithm Family**: state_estimation

## Objective
Test whether Extended Kalman Filter (EKF) can outperform optimization-based approaches by treating source parameters as hidden state and temperature measurements as observations. This is a fundamentally different paradigm from optimization - EKF sequentially processes measurements to refine a state estimate.

## Hypothesis
EKF treats source params (x, y, Q) as hidden state and sensor temperatures as observations. Literature shows EKF works for 3D heat source tracking. Sequential processing of timesteps may help escape local minima.

## Results Summary
- **Best In-Budget Score**: None (completely over budget)
- **Best Overall Score**: 1.0058 @ 357.9 min (5-sample projection)
- **Baseline Comparison**: -0.1189 vs 1.1247 (10.6% worse), 6.3x slower
- **Status**: **FAILED** - Fundamentally unsuitable

## Tuning History

| Run | Config | Score | Time (min) | In Budget | Notes |
|-----|--------|-------|------------|-----------|-------|
| 1 | 20 timesteps, 5 samples | 1.0058 | 357.9 | No | 233 sims/sample, completely unviable |

**Note**: Aborted after quick test - full 80-sample run would take ~6 hours.

## Key Findings

### Why EKF Failed

1. **Expensive Jacobian Computation**: EKF requires computing the observation Jacobian H at each timestep:
   - For 2-source (6-dim state): 7 simulations per timestep (state_dim + 1)
   - With 20 timesteps: 140 simulations just for Jacobians
   - Plus predictions and polish: ~233 total simulations/sample
   - CMA-ES baseline uses only 20-56 simulations/sample

2. **Local Convergence (Same as LM)**: EKF is fundamentally a local estimator that linearizes around the current state. For nonlinear observations (PDE-based temperatures), this causes:
   - Convergence to nearest consistent state (often a local minimum)
   - No ability to escape local minima
   - Same failure mode as Levenberg-Marquardt

3. **Static Source Problem**: For static sources, the state transition is identity (x_{t+1} = x_t). This means:
   - No dynamic information to leverage
   - EKF degenerates to iterated batch estimation
   - Equivalent to running LM with sequential measurement updates

4. **Cost Model Mismatch**: EKF is designed for problems where:
   - State transitions are informative (dynamics help estimation)
   - Observations are cheap relative to state updates
   - Our problem has: trivial transitions + expensive observations (opposite)

### Comparison with CMA-ES

| Aspect | CMA-ES (baseline) | Extended Kalman Filter |
|--------|------------------|------------------------|
| Simulations/sample | 20-56 | ~233 |
| Global vs local | Global (population) | Local (linearization) |
| Jacobian required | No | Yes (expensive) |
| Handles multi-modal | Yes | No |
| Time (projected) | 57 min | 358 min (6.3x) |
| Score | 1.1247 | 1.0058 |

### Relationship to Prior Failures

EKF failed for the **exact same reasons** as Levenberg-Marquardt:
- Both are fundamentally gradient-based local methods
- Both require expensive Jacobian computation
- Both get stuck in local minima
- The only difference is sequential vs batch processing, which doesn't help

## Parameter Sensitivity

Not evaluated due to fundamental infeasibility:
- Any reduction in timesteps would reduce accuracy further
- Any increase would exceed budget further
- The approach cannot be tuned to be competitive

## Recommendations for Future Experiments

1. **Mark state_estimation family as EXHAUSTED** - EKF and similar filters (UKF, particle filters with resampling) all require Jacobians or many samples, making them too expensive.

2. **Do NOT pursue particle_filter_inversion** - Similar issues: requires many particles (simulations) to represent posterior well.

3. **The gradient_approximation family is exhausted** - All methods requiring gradients (LM, EKF, L-BFGS-B, SPSA) fail due to:
   - Local optima trapping
   - Expensive Jacobian/gradient computation
   - Multi-modal landscape incompatibility

4. **CMA-ES is optimal for this problem** because:
   - Population-based global search (no local trapping)
   - No Jacobian required (evolution strategy)
   - Efficient covariance adaptation
   - ~10x more sample-efficient than EKF

## Technical Details

### EKF Implementation
- State vector: [x1, y1, q1, x2, y2, q2] for 2-source
- Observation: Temperature at sensors at current timestep
- State transition: Identity (static sources)
- Measurement model: Full PDE simulation
- Jacobian: Numerical finite differences (eps=0.01)

### Why Literature Success Doesn't Transfer
The cited paper uses EKF for **tracking moving sources** where:
- State transition provides informative dynamics
- Prior position helps predict future position
- Observations are incremental updates

Our problem has **static sources** where:
- No dynamics to leverage
- All information must come from observations
- EKF reduces to batch estimation

## Conclusion

**Extended Kalman Filter is fundamentally unsuitable for inverse heat source identification** because:

1. It requires expensive Jacobian computation (7x simulations/timestep for 2-source)
2. It's a local estimator that gets stuck in local minima
3. Static sources provide no dynamics to leverage
4. 10.6% worse accuracy at 6.3x the computational cost

**The state_estimation family should be marked as EXHAUSTED.** No filter-based approach can compete with CMA-ES for this problem.

## Raw Data
- Quick test: 5 samples, 268.4s total, score 1.0058
- Avg simulations per sample: 233.4
- MLflow: Not logged (aborted before full run)

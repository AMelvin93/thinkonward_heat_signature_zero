# Experiment Summary: frequency_domain_optimization

## Metadata
- **Experiment ID**: EXP_FREQUENCY_DOMAIN_001
- **Worker**: W2
- **Date**: 2026-01-24
- **Algorithm Family**: frequency_domain

## Objective
Exploit spectral structure of heat equation to enable faster RMSE computation through Fourier/frequency domain analysis.

## Hypothesis
Heat equation simplifies in frequency domain. Temperature signals are low-frequency, so using FFT could reduce computation while maintaining RMSE correlation.

## Results Summary
- **Best In-Budget Score**: N/A (not implemented)
- **Best Overall Score**: N/A
- **Baseline Comparison**: N/A
- **Status**: **ABORTED** - Simulation bottleneck unchanged

## Key Findings

### Finding 1: Temperature Signals Are Low-Frequency ✓

| Metric | Value |
|--------|-------|
| Total frequency components | 2002 (1001 timesteps × 2 sensors) |
| Components for 95% energy | 5 |
| Components for 99% energy | 30 |
| Potential FFT compression | 99.8% |

The temperature signals are extremely low-frequency. FFT-based compression is theoretically attractive.

### Finding 2: Frequency RMSE Correlates Perfectly ✓

| Metric | Correlation |
|--------|-------------|
| Full frequency RMSE vs time RMSE | r = 1.0000 |
| Truncated (10%) freq RMSE vs time RMSE | r = 1.0000 |

This is expected from Parseval's theorem: the energy in frequency domain equals the energy in time domain. Optimizing one optimizes the other.

### Finding 3: Simulation Is The Bottleneck ✗

| Operation | Time | Relative |
|-----------|------|----------|
| **Simulation** | 1207 ms | **14,721x** |
| Time-domain RMSE | 0.082 ms | 1x |
| Frequency-domain RMSE | 0.183 ms | 2.2x |

**Critical insight: RMSE computation is already negligible (0.082ms).** The bottleneck is the ADI simulation (1207ms), not the RMSE calculation.

## Why Frequency Domain Doesn't Help

### The Fundamental Problem

```
Standard approach:
1. Simulate Y_sim(t) at source position   [1207 ms - BOTTLENECK]
2. Compute RMSE(Y_sim, Y_obs)             [0.082 ms - negligible]

Frequency domain approach:
1. Simulate Y_sim(t) at source position   [1207 ms - STILL NEEDED]
2. Compute FFT(Y_sim), FFT(Y_obs)         [0.05 ms]
3. Compute freq_RMSE                       [0.1 ms]
```

Both approaches require the full simulation. Frequency domain adds FFT overhead but doesn't reduce the simulation that dominates runtime.

### What Would Actually Help

For frequency domain to help, we would need:
1. **Analytical frequency response** - Compute temperature at sensors directly in frequency domain without time-stepping
2. **This requires**: Fourier transform of heat equation Green's function
3. **Problem**: Our boundary conditions (Dirichlet/Neumann) don't yield simple closed-form frequency responses

### Comparison to Temporal Fidelity (40% Timesteps)

| Approach | How it reduces simulation | Works? |
|----------|--------------------------|--------|
| Temporal fidelity | Run fewer timesteps | ✓ 2.5x speedup |
| Frequency domain | Same timesteps, different RMSE | ✗ No speedup |

Temporal fidelity works because it reduces the number of ADI iterations. Frequency domain doesn't change the simulation at all.

## Abort Criteria Met

From experiment specification:
> "Spectral RMSE doesn't correlate with time-domain RMSE OR boundary conditions complicate transform"

The actual abort reason:
> **Simulation is 14,721x slower than RMSE computation. Frequency domain optimization addresses the wrong bottleneck. The ADI simulation must still run to completion regardless of how RMSE is computed.**

## Recommendations

### 1. frequency_domain Family Should Be Marked EXHAUSTED
Frequency analysis doesn't reduce simulation cost. The heat equation simplification in frequency domain is true analytically, but not applicable when we need numerical simulation.

### 2. Focus on Reducing Simulation Count, Not RMSE Computation
The baseline already achieves ~0.1ms RMSE computation. Improvements must come from:
- Reducing number of simulations (CMA-ES efficiency)
- Reducing simulation time (temporal fidelity)
- Smarter initialization (fewer iterations)

### 3. Green's Function May Still Help (Different Approach)
The new EXP_GREENS_FUNCTION_001 experiment takes a different approach: analytical solution via Green's function rather than numerical simulation. This could bypass the ADI bottleneck entirely.

## Conclusion

**ABORTED** - Frequency domain RMSE computation requires running the full ADI simulation first. RMSE computation is already negligible (0.082ms vs 1207ms simulation). Optimizing the frequency domain component addresses the wrong bottleneck. The frequency_domain family is fundamentally inapplicable to this problem.

## Files
- `feasibility_analysis.py`: Timing and correlation analysis
- `STATE.json`: Experiment state tracking

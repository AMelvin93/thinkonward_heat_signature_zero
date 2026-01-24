# Experiment Summary: modal_identification_method

## Metadata
- **Experiment ID**: EXP_MODAL_IDENTIFICATION_001
- **Worker**: W2
- **Date**: 2026-01-24
- **Algorithm Family**: model_reduction

## Objective
Build low-order modal model from sensor data for fast source estimation using Modal Identification Method (MIM).

## Hypothesis
MIM builds reduced-order model directly from temperature measurements. Can estimate time-varying heat sources through inversion of low-order model without full simulation.

## Results Summary
- **Best In-Budget Score**: N/A (not implemented)
- **Best Overall Score**: N/A
- **Baseline Comparison**: N/A
- **Status**: **ABORTED** - Simulation is bottleneck, modal space doesn't help

## Key Findings

### Finding 1: Only 2 Sensors = Only 2 Modes Observable

| Metric | Value |
|--------|-------|
| Number of sensors | 2 |
| Maximum observable modes | 2 |
| Energy in mode 1 | 99.6% |
| Energy in mode 2 | 0.4% |

With only 2 sensors, SVD can only extract 2 modes. The thermal field has infinitely many modes, but sensor sparsity fundamentally limits what we can observe.

### Finding 2: Simulation is the Bottleneck

| Operation | Time |
|-----------|------|
| ADI simulation | 1,167 ms |
| SVD modal extraction | 0.02 ms |
| RMSE computation | < 0.1 ms |

Modal identification adds SVD overhead without reducing simulation count. We still need to simulate each candidate's temperature response.

### Finding 3: Modal Inversion is Still Nonlinear

```
Temperature at sensor: T_i(t) = q × ∫ G(sensor_i, source, t-τ) dτ

With 2 sensors: 2 equations
Unknowns: x_s, y_s, q = 3 variables

This is underdetermined! Adding temporal structure doesn't help
because all temporal info is already captured in the 2 modes.

Source position appears nonlinearly in mode coefficients.
We still need CMA-ES or similar optimizer.
```

### Finding 4: MIM Literature Assumptions Don't Hold

The Modal Identification Method literature assumes:
1. ✗ KNOWN modal basis (from FEM or analytical) - Our physics vary per sample
2. ✗ DENSE sensor network (many sensors) - We only have 2 sensors
3. ✗ LINEAR source parameterization - Our source position is nonlinear

When these assumptions fail, MIM reduces to standard inverse problem.

## Why Modal Identification Doesn't Help

### The Fundamental Problem

```
MIM approach:
1. Extract modes from Y_obs (SVD/DMD)        [0.02 ms]
2. For each candidate source:
   a. Simulate temperature response          [1167 ms - STILL NEEDED]
   b. Extract modes from simulation          [0.02 ms]
   c. Compare with observed modes            [~0 ms]
3. Still requires iterative optimization!

Baseline approach:
1. For each candidate source:
   a. Simulate temperature response          [1167 ms]
   b. Compute RMSE                           [0.1 ms]
2. CMA-ES selects best candidates

Both approaches require the same number of simulations!
```

### What Would Make MIM Work

For MIM to provide speedup:
1. ✗ Pre-computed universal modal basis (blocked by sample-specific physics)
2. ✗ Many sensors for richer modal content (only 2 sensors)
3. ✗ Fast mode-to-source mapping (still nonlinear optimization)

## Abort Criteria Met

From experiment specification:
> "Sensor sparsity prevents reliable mode identification OR modal truncation loses source information"

Actual abort reason:
> **Only 2 sensors limits observable modal content to 2 modes. Simulation is the bottleneck (1167ms), not RMSE computation. Modal space doesn't reduce simulation cost. MIM assumptions don't hold for this problem.**

## Recommendations

### 1. model_reduction Family Should Be Marked EXHAUSTED
Any modal decomposition approach (POD, MIM, DMD) fails because:
- Sensor sparsity limits observable content
- Simulation is the bottleneck, not RMSE/mode computation
- Sample-specific physics prevent universal basis

### 2. Simulation Cost is the Only Lever
All approaches that don't reduce simulation count cannot help:
- Frequency domain RMSE ✗ (still needs simulation)
- Green's function ✗ (slower than ADI)
- Modal identification ✗ (still needs simulation)
- Surrogate models ✗ (sample-specific RMSE landscapes)

Only temporal fidelity (40% timesteps) successfully reduces simulation cost.

## Conclusion

**ABORTED** - Modal Identification Method provides no computational advantage because simulation is the bottleneck, not RMSE computation or mode extraction. With only 2 sensors, we can only observe 2 modes, and source inversion remains nonlinear. The model_reduction family is fundamentally inapplicable to this sparse-sensor inverse problem.

## Files
- `feasibility_analysis.py`: SVD analysis and timing comparison
- `STATE.json`: Experiment state tracking

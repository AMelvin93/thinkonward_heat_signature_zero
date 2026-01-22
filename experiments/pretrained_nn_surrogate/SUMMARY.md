# Experiment Summary: pretrained_nn_surrogate

## Metadata
- **Experiment ID**: EXP_PRETRAINED_SURROGATE_001
- **Worker**: W2
- **Date**: 2026-01-22
- **Algorithm Family**: neural_operator

## Objective
Pre-train a neural network on 10K+ simulations to create a fixed surrogate that predicts RMSE from source positions, enabling fast candidate filtering during optimization.

## Hypothesis
Previous NN surrogate failed due to online learning + parallel issues. Offline pre-training creates a fixed surrogate that works with parallel processing.

## Results Summary
- **Status**: **ABORTED** - Fundamental infeasibility discovered
- **Root Cause**: RMSE landscape is completely sample-specific

## Feasibility Analysis

### The Test
Evaluated RMSE for a 3x3 grid of source positions across 5 1-source samples with different kappa values.

### The Result
| Sample Pair | Spearman r |
|-------------|-----------|
| 0 vs 1 | -0.700 |
| 0 vs 2 | +0.633 |
| 0 vs 3 | -0.833 |
| 0 vs 4 | +0.617 |
| 1 vs 2 | -0.867 |
| 1 vs 3 | +0.900 |
| 1 vs 4 | -0.817 |
| 2 vs 3 | -0.817 |
| 2 vs 4 | +0.983 |
| 3 vs 4 | -0.767 |

**Average correlation: -0.167**

### Interpretation
- Many sample pairs have **negative correlation** (opposite RMSE landscapes!)
- A position that gives low RMSE for one sample may give high RMSE for another
- This is because RMSE depends on:
  1. Source positions (x, y) - what we'd input to surrogate
  2. **Observed sensor data (Y_observed)** - different for each sample
  3. **Physical parameters (kappa, bc, T0)** - different for each sample

A surrogate that only takes (x, y) as input CANNOT predict RMSE without knowing the sample.

## Why This Experiment Was Doomed

### Fundamental Flaw in Proposal
The original implementation plan:
```
1. Generate training data: 10K random source configs -> RMSE values
2. Train small MLP: (x, y, [x2, y2]) -> predicted_RMSE
```

This assumes RMSE = f(x, y) is a fixed function. But actually:
```
RMSE = f(x, y, Y_observed, kappa, bc, T0)
```

### Could We Fix It?
Potential approaches that were NOT tried:
1. **Per-sample surrogate**: Train a new surrogate for each sample
   - Problem: Training takes too long (defeats the purpose)
2. **Conditional surrogate**: Include sample features as input
   - Problem: What features? Sensor readings are high-dimensional (nt × n_sensors)
3. **Meta-learning**: Train a "surrogate generator"
   - Problem: Still needs sample information, complexity not worth it

## Key Findings

### Why Neural Operator Papers Succeed
Research papers on neural surrogates for PDEs typically:
- Fix the physical parameters (same kappa, bc for all samples)
- Use the SAME observed data with different source guesses
- Or predict temperature fields, not RMSE directly

Our problem has variable physics per sample, making surrogates much harder.

### Previous Surrogate Failures Explained
| Experiment | Why It Failed |
|------------|--------------|
| Neural Network Surrogate | Online learning + parallel issues |
| lq-CMA-ES | API mismatch (returns 1 solution) |
| POD (Proper Orthogonal Decomposition) | Sample-specific physics prevents universal basis |
| **This experiment** | RMSE landscape is sample-specific |

**Pattern**: All surrogate approaches fail because each sample has unique physics.

## Recommendations for Future Experiments

### 1. Mark neural_operator Family EXHAUSTED
No surrogate approach will work without sample-specific information, which defeats the purpose (training per sample is too slow).

### 2. Focus on Temporal Fidelity
The current best approach (40% timesteps) is already a form of multi-fidelity that works because it uses the SAME physics, just fewer timesteps.

### 3. Avoid Surrogates
The inverse heat problem with variable physics is fundamentally unsuitable for surrogate-based optimization.

## Technical Details

### Sample Variability
```
Sample 0 (kappa=0.0500): RMSE range [0.653, 3.845]
Sample 1 (kappa=0.0500): RMSE range [0.103, 0.802]
Sample 2 (kappa=0.0500): RMSE range [0.676, 3.049]
Sample 3 (kappa=0.1000): RMSE range [0.083, 1.352]
Sample 4 (kappa=0.1000): RMSE range [0.155, 0.845]
```

Even samples with the same kappa have vastly different RMSE ranges (due to different Y_observed).

## Conclusion
**ABORTED** - The premise is fundamentally flawed. RMSE landscape is sample-specific due to variable physics (kappa, bc, T0) and observed data. Average correlation between sample RMSE landscapes is -0.167 (essentially anti-correlated). A universal surrogate cannot work without sample information, and per-sample training is too slow. The neural_operator family should be marked EXHAUSTED for this problem.

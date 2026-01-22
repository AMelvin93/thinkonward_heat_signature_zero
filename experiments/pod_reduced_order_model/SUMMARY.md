# Experiment Summary: pod_reduced_order_model

## Metadata
- **Experiment ID**: EXP_POD_SURROGATE_001
- **Worker**: W2
- **Date**: 2026-01-19
- **Algorithm Family**: surrogate_pod

## Objective
Use POD (Proper Orthogonal Decomposition) as a fast surrogate for thermal simulation, potentially achieving 70-100x speedup.

## Hypothesis
POD-based reduced order models can compress thermal field information into 5-10 dominant modes, enabling fast approximate simulations.

## Results Summary
- **Best In-Budget Score**: N/A (not implemented)
- **Status**: ABORTED - POD not viable for this problem

## Feasibility Check Results

The feasibility check (from prior work) showed POD CAN mathematically capture temperature fields:

| Configuration | 10 modes | 20 modes |
|--------------|----------|----------|
| 1-source: energy captured | 93.05% | 98.54% |
| 1-source: reconstruction error | 4.8% | 2.1% |
| 2-source: energy captured | 92.68% | 98.27% |
| 2-source: reconstruction error | 5.9% | 2.9% |

## Why POD Was Aborted (Key Findings)

### 1. Fundamental Problem: Sample-Specific Physics
Each sample has unique physical parameters:
- **kappa**: Thermal diffusivity varies between samples
- **bc**: Boundary conditions may differ
- **T0**: Initial temperature varies
- **sensors_xy**: Sensor locations are sample-specific

POD requires pre-computed snapshots from the SAME physical system. With varying physics, we cannot build a universal POD basis that works across samples.

### 2. Online POD Would Defeat the Purpose
If we built a sample-specific POD during optimization:
1. We'd need to run simulations to collect snapshots
2. This is exactly what we're trying to avoid
3. The overhead would likely exceed the savings

### 3. Temporal Fidelity Already Works
The temporal fidelity approach (40% timesteps) already provides similar benefits with zero complexity:

| Approach | Speedup | Error | Complexity |
|----------|---------|-------|------------|
| Temporal fidelity (40%) | 2.5x | ~5% | Trivial (truncate simulation) |
| POD (if it worked) | 2-3x | 5-10% | High (SVD, mapping, etc.) |

### 4. Comparison to Failed NN Surrogate
This shares the same fundamental flaw as the failed neural network surrogate:
- Both require learning a mapping during optimization
- Parallel processing makes online learning difficult
- The overhead of learning exceeds simulation savings

## Conclusion

POD is not viable for the heat source identification problem because:
1. **Variable physics** prevents pre-building a universal basis
2. **Online construction** requires simulations, defeating the purpose
3. **Temporal fidelity** already achieves similar speedup with zero complexity
4. **Implementation complexity** is high with uncertain payoff

## Recommendations for Future Experiments

1. **ABANDON surrogate approaches** - The problem structure (sample-specific physics) makes surrogates fundamentally challenging
2. **Focus on temporal fidelity extensions** - This approach works and is simple
3. **Consider adjoint methods** - If we had access to gradients, we could use gradient-based optimization (but this requires simulator modification)

## Files
- `pod_feasibility.py`: Feasibility check script (completed prior)
- `STATE.json`: Experiment state with analysis

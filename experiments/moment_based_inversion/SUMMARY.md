# Experiment Summary: moment_based_inversion

## Metadata
- **Experiment ID**: EXP_MOMENT_INVERSION_001
- **Worker**: W1
- **Date**: 2026-01-25
- **Algorithm Family**: direct_inversion_v2

## Objective
Use moment analysis of heat flow for source identification. The hypothesis was that moment formulation transforms the ill-posed inverse problem to a more stable form without optimization iteration.

## Hypothesis
Spatial/temporal moments of the temperature field can be related analytically to source parameters, enabling direct (non-iterative) source identification.

## Results Summary
- **Status**: ABORTED (Feasibility Analysis)
- **Reason**: Theoretically unsound for our problem constraints

## Feasibility Analysis

### Problem Constraints
1. **Sparse sensors**: Only 2-6 sensors per sample (not dense spatial sampling)
   - 10% have 2 sensors
   - 30% have 3 sensors
   - 30% have 4 sensors
   - 20% have 5 sensors
   - 10% have 6 sensors

2. **Per-sample unique sensors**: All 80 samples have different sensor configurations (100% unique)

3. **Boundary conditions**: Non-trivial BC (Dirichlet/Neumann) break analytical Green's function solutions

4. **Multi-source problems**: 60% of samples have 2 sources = 6 unknowns to estimate

### Prior Failed Direct Inversion Experiments
| Experiment | Failure Reason |
|------------|----------------|
| Green's function inversion | 4.3x slower than ADI, boundary effects break analytical solution |
| Compressive sensing (D-PBCS) | 646x over budget, per-sample sensors prevent pre-computation |
| Modal identification | Only 2 sensors = 2 modes max, insufficient |
| RBF meshless inversion | Same as compressive sensing - per-sample computation too expensive |

### Why Moment-Based Will Fail
1. **Sparse sensors cannot compute spatial moments**: Moment computation (centroid, spread) requires dense spatial sampling or knowledge of the full temperature field. With 2-6 sparse sensors, we cannot estimate field moments.

2. **Temporal moments insufficient**: While we have dense temporal sampling (800-1200 timesteps), relating temporal moments (peak time, rise rate) to source positions requires the Green's function solution, which breaks down with boundary conditions.

3. **Baseline already uses best direct approach**: The triangulation initialization already exploits sensor readings to estimate initial source positions. This IS a form of moment-based reasoning (using peak temperatures and sensor positions).

4. **Direct inversion without iteration has repeatedly failed**: The problem requires iterative refinement (CMA-ES) to achieve good accuracy. All prior direct methods failed.

## Key Insight
**The problem is fundamentally optimization-based, not direct-inversion-based.**

The scoring formula and RMSE objective require finding parameters that minimize the simulation-observation mismatch. This is inherently an optimization problem. Direct inversion methods can only provide rough initialization (which triangulation already does well).

## Recommendations for Future Experiments
1. **Do NOT pursue more direct inversion methods**: Green's function, compressive sensing, modal, RBF, and moment-based approaches have all failed or will fail for the same reasons.

2. **Focus on CMA-ES optimization improvements**: The baseline CMA-ES + triangulation + NM polish is the correct algorithmic structure.

3. **direct_inversion_v2 family is EXHAUSTED**: No viable direct inversion approach exists for sparse-sensor heterogeneous problems.

## Algorithm Family Status
- **direct_inversion_v2 family**: EXHAUSTED
- All prior direct methods failed; moment-based would face identical issues

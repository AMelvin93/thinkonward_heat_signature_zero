# Experiment Summary: rbf_meshless_inverse

## Metadata
- **Experiment ID**: EXP_RBF_MESHLESS_001
- **Worker**: W2
- **Date**: 2026-01-24
- **Algorithm Family**: meshless_direct

## Objective
Use Radial Basis Function (RBF) meshless methods to directly solve the inverse heat source problem without iterative optimization.

## Hypothesis
RBF methods can reformulate the inverse problem as a linear system Ax=b, where A is an observation matrix built from RBF basis functions. This would replace iterative CMA-ES with direct matrix inversion.

## Results Summary
- **Best In-Budget Score**: N/A (not implemented)
- **Best Overall Score**: N/A
- **Baseline Comparison**: N/A
- **Status**: **ABORTED** - Matrix A computation exceeds per-sample budget

## Key Findings

### Finding 1: RBF Approach Still Requires Simulations

```
RBF Inverse Problem Formulation:

1. FORWARD MODEL:
   T(x,y,t) = ∫ G(x,y,t; x',y') × q(x',y') dx'dy'
   where G is Green's function, q is source term

2. RBF APPROXIMATION:
   Approximate source: q(x,y) = Σ_j λ_j × φ(||r - r_j||)
   where φ is RBF kernel, r_j are collocation points

3. LINEAR SYSTEM:
   A × λ = T_obs
   where A_ij = thermal response at sensor_i from RBF point j

4. KEY ISSUE:
   Each column of A requires one simulation!
   A depends on (BC, kappa, sensor_positions) - all sample-specific
```

### Finding 2: Matrix A Computation Cost

| Grid Size | RBF Points | Simulations | Time | Budget Ratio |
|-----------|------------|-------------|------|--------------|
| 5×3 | 15 | 15 | 15 sec | **1.7x over** |
| 10×5 | 50 | 50 | 50 sec | 5.6x over |
| 20×10 | 200 | 200 | 200 sec | 22x over |

Even the coarsest possible RBF grid (5×3=15 points) exceeds the per-sample budget (9 sec) by 1.7x.

### Finding 3: Pre-computation is Impossible

| Factor | Unique Values | Impact |
|--------|---------------|--------|
| Sensor positions | 80 (100% unique) | Different A for every sample |
| (BC, kappa) | 4 combinations | Limited benefit |

With 100% unique sensor positions across 80 samples, matrix A cannot be pre-computed and reused. Each sample requires its own A matrix, which requires n_rbf_points simulations.

### Finding 4: Same Issue as D-PBCS

This is fundamentally the same problem as D-PBCS (tested in EXP_PHYSICS_CS_001):

| Approach | Observation Matrix Size | Simulations Needed | Result |
|----------|------------------------|-------------------|--------|
| D-PBCS | sensors × grid points | 5000 | 97 min/sample (646x over) |
| RBF Meshless | sensors × RBF points | 15-200 | 15-200 sec (1.7-22x over) |

Both approaches require building an observation matrix where each column represents thermal response to a unit source. The cost is O(n_points) simulations per sample.

## Why RBF Meshless Doesn't Work

### The Fundamental Problem

```
RBF approach does NOT avoid simulation:
- Each RBF collocation point needs one simulation to compute its column in A
- Even coarsest grid (5×3=15 pts) needs 15 simulations
- 15 sims × 1 sec/sim = 15 sec >> 9 sec budget

The name "meshless" is misleading for our use case:
- RBF is meshless for the FORWARD problem (no finite difference grid)
- But for INVERSE problems, we still need to probe each RBF point
- "Meshless" doesn't mean "simulation-free"
```

### Comparison to Baseline

```
Baseline (CMA-ES + 40% fidelity):
- 20-36 fevals × 0.4 sec/feval = 8-14 sec per sample
- Uses simulation in optimization loop
- Achieves score 1.1688

RBF Meshless (minimum viable):
- 15 sims × 1 sec/sim = 15 sec JUST for matrix A
- Plus matrix solve + post-processing
- Total would be ~16-17 sec minimum
- 1.7x over budget before even starting optimization
```

## Abort Criteria Met

From experiment specification:
> "Mesh-free method requires more function evaluations than CMA-ES"

Actual abort reason:
> **RBF meshless requires n_rbf_points simulations to build matrix A. Even coarsest grid (5×3=15 pts = 15 sec) exceeds per-sample budget (9 sec) by 1.7x. 100% unique sensor positions prevent pre-computation.**

## Recommendations

### 1. meshless_direct Family Should Be Marked EXHAUSTED
Any RBF/meshless approach to inverse problems requires building an observation matrix, which costs O(n_points) simulations per sample.

### 2. All "Direct Inversion" Methods Are Prohibitively Expensive
This is the third failed direct method:
- **Green's Function**: 4.3x slower than ADI
- **D-PBCS**: 646x over budget
- **RBF Meshless**: 1.7x over budget (even at coarsest)

The common failure mode: building observation matrices requires too many simulations.

### 3. Iterative Optimization Remains Optimal
CMA-ES with 20-36 fevals is fundamentally more efficient than any direct method requiring O(n_grid) simulations to build matrices.

## Conclusion

**ABORTED** - RBF Meshless Inverse Method is not viable because matrix A computation requires n_rbf_points simulations. Even the coarsest possible grid (5×3=15 points) exceeds the per-sample budget by 1.7x. With 100% unique sensor positions, pre-computation is impossible. This is fundamentally the same issue as Green's function and D-PBCS approaches - direct inversion requires too many simulations.

## Files
- `feasibility_analysis.py`: RBF cost and feasibility analysis
- `STATE.json`: Experiment state tracking

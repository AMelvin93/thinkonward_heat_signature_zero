# Experiment Summary: physics_compressive_sensing

## Metadata
- **Experiment ID**: EXP_PHYSICS_CS_001
- **Worker**: W2
- **Date**: 2026-01-24
- **Algorithm Family**: inverse_method

## Objective
Use Decomposed Physics-Based Compressive Sensing (D-PBCS) to directly solve the inverse heat source problem without iterative optimization, leveraging sparsity constraints and physics.

## Hypothesis
D-PBCS combines compressed sensing sparsity constraints with heat equation physics. Since heat sources are sparse (1-2 point sources), formulating as a CS problem could enable direct solution via convex optimization instead of iterative CMA-ES.

## Results Summary
- **Best In-Budget Score**: N/A (not implemented)
- **Best Overall Score**: N/A
- **Baseline Comparison**: N/A
- **Status**: **ABORTED** - Observation matrix construction prohibitively expensive

## Key Findings

### The D-PBCS Formulation

To formulate as compressive sensing, we need:
- **Measurement vector y**: Sensor temperatures over time (n_sensors × n_timesteps)
- **Observation matrix A**: Temperature response at sensors for unit source at each grid cell
- **Source vector x**: Sparse vector indicating source intensity at each grid cell

The relationship: **y = Ax + noise**, where x is k-sparse (k ≤ 2 sources)

### Sparsity Condition: ✓ SATISFIED

| Metric | Value | Requirement |
|--------|-------|-------------|
| Grid cells (n) | 5000 | - |
| Max sources (k) | 2 | k << n |
| Sparsity ratio | 0.04% | Very sparse ✓ |
| Measurements (m) | 2000-6000 | - |
| CS bound | 68 | m >> c·k·log(n) ✓ |

The sparsity condition is well-satisfied: we have 2000-6000 measurements for a 2-sparse signal in 5000 dimensions.

### Observation Matrix Construction: ✗ PROHIBITIVE

| Metric | Value | Impact |
|--------|-------|--------|
| Simulations for A | 5000 | One per grid cell |
| Time per simulation | 1.16 sec | Full-fidelity ADI solver |
| **Time for A** | **97 min/sample** | Critical bottleneck |
| Budget per sample | 9 sec | 60 min / 400 samples |
| **Budget exceedance** | **646x** | Fatal |

### Coarse Grid Alternative: ✗ STILL FAILS

Even with drastically coarser grids:

| Grid | Cells | Time/Sample | Total (400 samples) |
|------|-------|-------------|---------------------|
| 100×50 | 5000 | 97 min | 38,850 min |
| 50×25 | 1250 | 24 min | 9,695 min |
| 25×12 | 300 | 6 min | 2,327 min |
| 20×10 | 200 | 4 min | 1,551 min |
| **10×5** | **50** | **1 min** | **388 min** |

Even the coarsest viable grid (10×5 = 50 cells) would take ~1 min/sample, still **6x over budget**.

### Pre-computation Strategy: ✗ NOT POSSIBLE

Could we pre-compute A once for all samples?

**NO**, because:
1. **Sample-specific sensor locations**: Each sample has UNIQUE sensor positions (80 samples → 80 unique configs)
2. **Sample-specific physics**: kappa varies (0.05 or 0.1)
3. **Sample-specific measurements**: Observation matrix depends on both sensor positions AND physics

The observation matrix A must be rebuilt for **every single sample**.

## Why This Approach Fundamentally Fails

The core insight from the research paper on D-PBCS assumes:
1. A fixed sensor configuration (same locations across measurements)
2. Either known physics OR ability to pre-compute responses
3. Multiple samples from the SAME system

Our problem violates all three assumptions:
1. ❌ Each sample has different sensor locations
2. ❌ Each sample has different physics (kappa)
3. ❌ Each sample is from a different system

### Comparison to Baseline CMA-ES

| Approach | Simulations/Sample | Time/Sample | Score |
|----------|-------------------|-------------|-------|
| D-PBCS | 5000+ (build A) | 97+ min | N/A |
| CMA-ES baseline | 60-90 | ~45 sec | 1.1688 |

CMA-ES is **130x faster** because it only evaluates candidate solutions, not all possible source locations.

## Abort Criteria Met

From experiment specification:
> "Heat source not actually sparse (continuous distribution) OR physics constraints don't match problem"

The actual abort reason:
> **Observation matrix construction requires 646x more time than available budget. Even coarse grids exceed budget by 6x. Sample-specific sensors prevent any pre-computation.**

## Recommendations for Future Experiments

### 1. Do NOT Pursue CS-Based Approaches
Any approach requiring exhaustive basis function computation (one per potential source location) will fail:
- Standard compressive sensing: FAILED (this experiment)
- Dictionary learning: Would have same matrix construction cost
- Sparse Bayesian learning: Same issue

### 2. inverse_method Family Should Be Marked EXHAUSTED
Direct inverse methods that require pre-computing system responses at all possible source locations are not viable:
- Time budget: 9 sec/sample
- Minimum matrix construction: >60 sec/sample (even very coarse)
- Fundamental mismatch

### 3. Focus on Iterative Methods
Methods that only evaluate candidate solutions (not all possible locations) are the only viable path:
- CMA-ES (baseline): 60-90 evaluations/sample ✓
- Frequency domain: Transforms the problem, may reduce evaluation cost

## Conclusion

**ABORTED** - D-PBCS requires constructing an observation matrix with 5000+ simulations per sample (97 min). The budget allows only 9 seconds per sample - a 646x shortfall. Coarse grids (10×5) still exceed budget by 6x. Sample-specific sensor locations prevent pre-computation. The inverse_method family is fundamentally incompatible with the time budget constraints.

## Files
- `feasibility_analysis.py`: Detailed cost analysis script
- `STATE.json`: Experiment state tracking

# Experiment Summary: gappy_clustering_pod

## Metadata
- **Experiment ID**: EXP_GAPPY_CPOD_001
- **Worker**: W2
- **Date**: 2026-01-24
- **Algorithm Family**: surrogate_v2

## Objective
Use Gappy Clustering-POD to handle heterogeneous physics across samples by clustering similar samples and building cluster-specific POD bases.

## Hypothesis
Standard POD failed because samples have varying physics (kappa, BC, T0). Gappy C-POD clusters samples by physics parameters first, then builds separate POD bases per cluster. This could enable fast surrogates for thermal simulation.

## Results Summary
- **Best In-Budget Score**: N/A (not implemented)
- **Best Overall Score**: N/A
- **Baseline Comparison**: N/A
- **Status**: **ABORTED** - Sensor location heterogeneity defeats the approach

## Key Findings

### What Worked: Physics Parameter Clustering ✓

The physics parameters are actually well-suited for clustering:

| Parameter | Unique Values | Distribution |
|-----------|---------------|--------------|
| kappa | 2 | 0.05 (45%), 0.10 (55%) |
| T0 | 1 | All 0.0 (100%) |
| n_sources | 2 | 1-src (40%), 2-src (60%) |
| BC type | 2 | Neumann (51%), Dirichlet (49%) |

Combined (kappa, T0, n_sources) clustering yields only **4 clusters** with viable sizes:
- Cluster sizes: [25, 23, 19, 13]
- All clusters have ≥5 samples (sufficient for POD basis)

### What Failed: Sensor Location Heterogeneity ✗

**CRITICAL FINDING: Every single sample has UNIQUE sensor locations!**

| Statistic | Value |
|-----------|-------|
| Total samples | 80 |
| Unique sensor configurations | 80 (100%) |
| Repeated configurations | 0 |
| Sensors per sample | 2-6 (varies) |

Within each physics cluster, every sample has different sensors:
- Cluster (kappa=0.05, n_sources=1): 13 samples, 13 unique sensor configs
- Cluster (kappa=0.05, n_sources=2): 23 samples, 23 unique sensor configs
- Cluster (kappa=0.10, n_sources=1): 19 samples, 19 unique sensor configs
- Cluster (kappa=0.10, n_sources=2): 25 samples, 25 unique sensor configs

## Why This Defeats Gappy C-POD

### 1. Gappy POD Requires Consistent Observation Locations
The "gappy" in Gappy POD refers to having a subset of known observation points (sensors) and using the POD basis to reconstruct the full field. This requires:
- A pre-computed POD basis from full-field snapshots
- **Consistent sensor locations** so the gappy reconstruction problem is the same

With 80 unique sensor configurations, we'd need 80 different gappy reconstruction problems.

### 2. Cannot Transfer POD Bases Across Samples
Even within a cluster of same-physics samples:
- Sample A has sensors at [(1.2, 0.3), (0.8, 0.7), ...]
- Sample B has sensors at [(0.5, 0.4), (1.9, 0.2), ...]

The POD coefficients from Sample A cannot predict anything about Sample B because the observation points differ entirely.

### 3. Reduces to Sample-by-Sample POD
If each sample needs its own gappy setup:
1. Generate training snapshots **for that specific sample**
2. Build POD basis **for that specific sample**
3. Use for optimization **only on that specific sample**

This is exactly what the previous POD experiment rejected - **online POD requires simulations, defeating the purpose**.

## Comparison to Prior POD Experiment

| Issue | Standard POD (EXP_POD_SURROGATE_001) | Gappy C-POD (This) |
|-------|-------------------------------------|---------------------|
| Physics heterogeneity | FAILED - varying kappa | **SOLVED** by clustering |
| Sensor heterogeneity | Not analyzed | **FAILED** - 100% unique sensors |
| Outcome | ABORTED | ABORTED |

Gappy C-POD solves the physics clustering problem but introduces a new unsolvable constraint: **sensor consistency**.

## Abort Criteria Met

From experiment specification:
> "Clusters too small for reliable POD OR kappa variation too continuous"

The actual abort reason is different but equally fatal:
> **Sensor locations are 100% unique across samples, making cross-sample POD basis reuse impossible.**

## Recommendations for Future Experiments

### 1. Do NOT Pursue Any POD-Based Approach
- Standard POD: FAILED (physics heterogeneity)
- Gappy C-POD: FAILED (sensor heterogeneity)
- Online POD: Defeats purpose (requires simulations)

The sensor location uniqueness is a fundamental property of the problem that prevents any POD surrogate approach.

### 2. surrogate_v2 Family Should Be Marked EXHAUSTED
Both attempted surrogate approaches in this family have failed:
- Pre-trained NN surrogate: FAILED (RMSE landscape is sample-specific)
- Gappy C-POD: FAILED (sensor locations are sample-specific)

### 3. Focus on Non-Surrogate Approaches
Remaining viable directions:
- EXP_PHYSICS_CS_001: D-PBCS (direct solve, no surrogate)
- EXP_FREQUENCY_DOMAIN_001: Frequency domain (different formulation)
- EXP_RBF_MESHLESS_001: RBF meshless (direct inverse)

## Conclusion

**ABORTED** - While physics parameters cluster well (only 4 clusters), every sample has unique sensor locations (80 configs for 80 samples). Gappy C-POD requires consistent observation points to transfer POD bases across samples. With 100% unique sensors, each sample needs individual treatment, which reduces to sample-by-sample online POD - exactly what was rejected in the previous POD experiment.

## Files
- `analyze_samples.py`: Physics parameter distribution analysis
- `check_sensors.py`: Sensor location consistency analysis
- `STATE.json`: Experiment state tracking

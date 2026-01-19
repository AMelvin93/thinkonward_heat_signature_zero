# Experiment Summary: multi_fidelity_proper_ratio (Pyramid)

## Metadata
- **Experiment ID**: EXP_MULTIFID_OPT_001
- **Worker**: W1
- **Date**: 2026-01-19
- **Algorithm Family**: multi_fidelity
- **Folder**: `experiments/multi_fidelity_pyramid/`

## Objective
Test multi-fidelity optimization with literature-recommended 4:1 cell ratios between grid levels to achieve speedup while maintaining accuracy.

## Hypothesis
4:1 cell ratio between fidelity levels (25x12 -> 50x25 -> 100x50) gives 5-10x speedup per level, allowing faster exploration on coarse grids with accuracy preserved through refinement on finer grids.

## Results Summary
- **Best In-Budget Score**: RMSE 0.2589 @ 44.9 min (Run 1)
- **Best Overall Score**: RMSE 0.2320 @ 92.2 min (Run 2 - over budget)
- **Baseline Comparison**: **45-85% WORSE** than baseline RMSE ~0.17
- **Status**: **FAILED**

## Tuning History

| Run | Config Changes | RMSE | Time (min) | In Budget | Notes |
|-----|---------------|------|------------|-----------|-------|
| 1 | Initial: 25x12 coarse, 10/16 fevals | 0.2589 | 44.9 | Yes | 45% worse accuracy |
| 2 | Larger: 30x15 coarse, 15/24 fevals | 0.2320 | 92.2 | No | 21% worse, 53% over budget |
| 3 | Same as Run 2 | 0.3039 | 73.3 | No | High variance, 79% worse |

### Detailed Results by Source Type
| Run | 1-src RMSE | Baseline 1-src | 2-src RMSE | Baseline 2-src |
|-----|------------|----------------|------------|----------------|
| 1 | 0.2039 | 0.14 | 0.2955 | 0.21 |
| 2 | 0.1698 | 0.14 | 0.2734 | 0.21 |
| 3 | 0.2674 | 0.14 | 0.3283 | 0.21 |

## Key Findings

### What Didn't Work
1. **Coarse-to-fine transfer fails for inverse problems**: The RMSE landscape on coarse grids (25x12, 30x15) is fundamentally different from fine grids (100x50). Optimal positions found on coarse grids do NOT correspond to optimal positions on fine grids.

2. **3-level pyramid adds complexity without benefit**: The medium grid (50x25) level didn't provide meaningful refinement. Solutions either transferred poorly from coarse level or needed full fine-grid optimization.

3. **High variance between runs**: Run 2 and Run 3 had identical configs but RMSE varied from 0.2320 to 0.3039 (31% difference). The coarse grid exploration is unstable.

4. **Time overhead from multi-level evaluation**: Despite fewer fine-grid simulations, the total overhead of coarse + medium + fine levels often exceeded single-level baseline.

### Why Multi-Fidelity Fails for This Problem

The thermal inverse problem has a key property that makes multi-fidelity ineffective:

**The RMSE landscape depends on grid resolution.**
- On coarse grids, heat diffuses differently (larger cells = different thermal dynamics)
- Sensor interpolation errors compound at coarse resolutions
- The "optimal" source position on a 25x12 grid is NOT the same as on 100x50 grid

This is different from forward simulation where coarse grids approximate fine grid behavior. For inverse problems, the objective function (RMSE) itself changes with resolution.

### Critical Insights
1. **Coarse grid = different optimization landscape**: Not just lower accuracy, but fundamentally different local minima
2. **Transfer learning assumption fails**: Can't learn from cheap approximations
3. **Single-fidelity CMA-ES is more robust**: Baseline spends all budget on correct resolution

## Parameter Sensitivity
- **Most impactful parameter**: Grid resolution (coarse grid size)
- **Time-sensitive parameters**: fevals per level, transfer_top_n
- **Negative impact**: Larger coarse grids (30x15) increased time without proportional accuracy gain

## Recommendations for Future Experiments

### What to AVOID
1. **Multi-fidelity with different physics resolutions** - The optimization landscape changes too much between resolutions
2. **Coarse grid pre-filtering** - Risks eliminating good candidates that look bad on coarse grid
3. **Hierarchical refinement** - Better to invest all budget in target resolution

### What MIGHT Work Instead
1. **Surrogate with SAME resolution but fewer time steps** - Keep spatial resolution, reduce temporal fidelity
2. **Early-stopping evaluation** - Simulate fewer time steps on fine grid for initial filtering
3. **Adjoint-based gradients** - True gradient information instead of coarse approximation

### For W0's Reference
- Multi-fidelity via grid coarsening is **ABANDONED** for this problem class
- Focus optimization efforts on single-resolution techniques
- Consider alternative cheap approximations (fewer timesteps, not coarser grid)

## Raw Data
- **MLflow run IDs**:
  - Run 1: `4ab5762f50754d5b8cd966cd8317ca71`
  - Run 2: `61f972c591254902adc3e4827ada3445`
  - Run 3: `b31e248d5a9b48ad906af975253d0276`
- **Best in-budget config**: Run 1 - `25x12 -> 50x25 -> 100x50` with `10/16` coarse fevals
- **Baseline to beat**: RMSE 0.17 @ 57 min

## Conclusion
**FAILED**: Multi-fidelity pyramid optimization does not work for thermal inverse problems. The optimization landscape changes too much between grid resolutions. Coarse grid solutions do not transfer to fine grid. Recommend abandoning multi-fidelity approaches based on spatial grid coarsening.

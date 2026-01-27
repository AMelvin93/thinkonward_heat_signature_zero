# Experiment Summary: mini_batch_sensor_eval

## Metadata
- **Experiment ID**: EXP_MINI_BATCH_SENSORS_001
- **Worker**: W2
- **Date**: 2026-01-26
- **Algorithm Family**: evaluation_v2

## Objective
Test whether using a random subset of sensors during CMA-ES exploration (similar to SGD mini-batches) can reduce per-evaluation cost and speed up optimization, while using full sensors only for final polish and ranking.

## Hypothesis
A random 50% of sensors may be sufficient to estimate candidate quality during CMA-ES exploration, enabling faster candidate screening with full evaluation only for the best candidates.

## Results Summary
- **Best In-Budget Score**: None (experiment FAILED to meet time budget)
- **Best Overall Score**: 1.1530 @ 80.8 min
- **Baseline Comparison**: +0.0283 vs 1.1247 (better accuracy, but 37% slower)
- **Status**: **FAILED** - approach fundamentally doesn't save time

## Tuning History

| Run | Config Changes | Score | Time (min) | In Budget | Notes |
|-----|---------------|-------|------------|-----------|-------|
| 1 | sensor_fraction=0.5 | 1.1530 | 80.8 | No | Slower than baseline despite using 50% sensors |

## Key Findings

### Why This Approach Fundamentally Fails

1. **PDE Simulation Dominates Cost**: The computational bottleneck is the PDE solver (Heat2D), not the sensor RMSE calculation. Using fewer sensors only reduces the RMSE calculation time, which is already negligible (<1% of total time).

2. **Sensors Only Affect Post-Simulation Step**: The sensor sampling happens AFTER the full PDE solve:
   ```python
   # PDE solve (expensive) - happens regardless of sensor count
   times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources)

   # Sensor sampling (cheap) - only this is affected by sensor_fraction
   Y_unit = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])
   ```

3. **Noise May Destabilize CMA-ES**: Using random sensor subsets introduces stochastic noise into the fitness landscape, potentially causing CMA-ES to require more iterations to converge.

4. **Full Evaluation Still Required**: Final candidate ranking and polish stages still use full sensors, so no time is saved there.

### What Didn't Work
- Mini-batch sensor evaluation cannot accelerate optimization because simulation cost is independent of sensor count
- The 50% sensor approach made timing WORSE (80.8 min vs 58.4 min baseline)

### Critical Insights
- **Do NOT pursue sensor subsampling for speedup** - it's a dead end
- The only way to reduce per-evaluation cost is to reduce simulation fidelity (coarse grids, fewer timesteps), NOT sensor count
- The baseline's multi-fidelity approach (coarse for exploration, fine for polish) already optimally addresses this

## Parameter Sensitivity
- **sensor_fraction**: Irrelevant - changing this parameter cannot provide meaningful speedup

## Recommendations for Future Experiments

1. **Do NOT explore sensor subsampling variations** - fundamentally flawed approach
2. **Focus on simulation fidelity** for speedup opportunities (grid resolution, timesteps)
3. **Do NOT confuse sensor reduction with simulation reduction** - they are completely different cost structures
4. **Cross off evaluation_v2 family** - stochastic evaluation approaches are not viable

## Conclusion

This experiment definitively proves that mini-batch sensor evaluation is NOT a viable optimization strategy for this problem. The computational bottleneck is PDE simulation, not RMSE calculation. Any future experiments targeting speedup should focus on simulation fidelity reduction or algorithmic efficiency, NOT sensor subsampling.

**Recommendation**: Mark evaluation_v2 family as EXHAUSTED - no viable paths forward in this direction.

## Raw Data
- MLflow run IDs: acff28eee6f145d4a06b2f9584542fde
- Best config: `{"sensor_fraction": 0.5}`

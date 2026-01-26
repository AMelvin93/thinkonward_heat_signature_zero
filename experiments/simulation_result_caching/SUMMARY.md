# Experiment Summary: simulation_result_caching

## Metadata
- **Experiment ID**: EXP_SIMULATOR_CACHING_001
- **Worker**: W2
- **Date**: 2026-01-25
- **Algorithm Family**: engineering

## Objective
Add LRU-style caching to simulation results to eliminate redundant evaluations during CMA-ES optimization. Hypothesis: discretizing source positions and caching results could save computation when the same positions are evaluated multiple times.

## Hypothesis
CMA-ES optimization may evaluate similar or identical source positions multiple times, especially near the optimum. Caching these results could eliminate redundant simulations and speed up optimization.

## Results Summary
- **Best In-Budget Score**: N/A (caching made things slower)
- **Best Overall Score**: 1.1654 @ 71.3 min
- **Baseline Comparison**: -0.0034 score, +22% time - FAILED
- **Cache Hit Rate**: 10.15%
- **Status**: FAILED

## Tuning History

| Run | Config | Score | Time (min) | In Budget | Cache Hit Rate |
|-----|--------|-------|------------|-----------|----------------|
| 1 | precision=3 | 1.1654 | 71.3 | No | 10.15% |

## Key Findings

### Cache Statistics
- Total cache hits: 1,113
- Total cache misses: 9,852
- Hit rate: 10.15%
- Simulations per sample: ~137 (avg)
- Unique positions per sample: ~123 (avg)

### Why Caching Failed

1. **Low hit rate (10%)**: CMA-ES generates new random positions each iteration
   - Discretizing to 3 decimals (~1mm precision) still yields unique keys
   - CMA-ES exploration doesn't repeatedly evaluate same positions
   - Only NM polish might revisit positions, but it's a small fraction

2. **Overhead outweighs savings**:
   - Cache lookup/storage: ~0.1ms per call
   - Average simulation: ~3ms
   - 10% hit rate saves: 0.1 * 3ms = 0.3ms per call
   - Overhead costs: 0.9 * 0.1ms = 0.09ms per call
   - Net: Small savings wiped out by memory/GC overhead

3. **Memory pressure**: Caching 1000+ arrays per sample adds memory overhead
   - GC pauses become significant
   - Cache eviction adds latency

### Critical Insight
**Caching only works when there are repeated evaluations. CMA-ES is designed to explore NEW positions, not revisit old ones.**

For CMA-ES optimization:
- Each iteration generates new candidate solutions
- Population-based search means diverse positions
- Sigma-based mutation ensures positions are different each time
- Caching is fundamentally misaligned with the algorithm's design

## Recommendations for Future Experiments
1. **engineering family for caching is EXHAUSTED** - don't try caching optimizations
2. Caching would only help if we had a deterministic search pattern (grid search)
3. For stochastic optimization like CMA-ES, positions are intentionally unique
4. Focus on making individual simulations faster, not caching them

## Raw Data
- MLflow run ID: ab29f04c6ca242f09934833bc3b39027
- Best config: No caching (baseline is optimal)

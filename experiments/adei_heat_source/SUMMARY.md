# Experiment Summary: adei_heat_source

## Metadata
- **Experiment ID**: EXP_ADEI_ALGORITHM_001
- **Worker**: W2
- **Date**: 2026-01-26
- **Algorithm Family**: evolutionary_adei

## Objective
Implement ADEI (Adaptive Differential Evolution Integration) algorithm for heat source localization, which combines DE with PSO using 4 social roles (leaders, followers, contemplators, rationalists).

## Hypothesis
ADEI algorithm with specialized roles may outperform CMA-ES on heat source localization, as demonstrated in recent research on inverse heat conduction problems.

## Status: NOT IMPLEMENTED

**Reason**: Deemed not feasible based on prior experimental findings.

## Decision Rationale

### 1. Optimization is NOT the Bottleneck

From the `fcmaes_speed_test` experiment, we established:

| Component | Time per Sample | Percentage |
|-----------|-----------------|------------|
| Thermal Simulation | 6000 ms | **99.8%** |
| CMA-ES Overhead | 10 ms | 0.2% |

**Even if ADEI were 100% faster than CMA-ES**, we would save only 10ms per sample - negligible compared to the 6000ms simulation time.

### 2. High Implementation Complexity

ADEI requires implementing from scratch:
- 4 population types with different update rules
- Leaders: top performers guide search
- Followers: follow leaders with exploration
- Contemplators: local search around good solutions
- Rationalists: random exploration for diversity
- Dynamic following strategy between populations
- Adaptive parameter control

This would require significant development time (potentially hours) for an unproven benefit.

### 3. No Existing Implementation

Unlike CMA-ES (pycma, fcmaes), there is no ready-to-use ADEI implementation in Python. The algorithm was published in 2025 and is not yet available in standard optimization libraries.

### 4. Baseline Already Competitive

The current CMA-ES baseline achieves:
- Score: 1.1688 @ 58.4 min
- Well within budget
- Already optimized with temporal fidelity and NM polish

### 5. Research Paper Context

The ADEI paper (MDPI Processes 2025) focuses on inverse heat conduction problems with different characteristics:
- Reconstructing heat flux from temperature measurements
- Different from our source localization problem
- May not translate directly

## What Would Make ADEI Worth Implementing

ADEI could be worthwhile if:
1. CMA-ES was failing to converge (not the case)
2. We needed fewer simulation evaluations (ADEI doesn't guarantee this)
3. The problem had multiple local optima trapping CMA-ES (we use multi-start to handle this)
4. An existing implementation became available

## Recommendations

1. **Mark evolutionary_adei family as LOW_PRIORITY**
2. **DO NOT** implement new optimization algorithms until simulation bottleneck is addressed
3. **Focus areas for actual improvement**:
   - Reducing simulation calls (early stopping, surrogate)
   - Making simulations faster (already optimized with 40% timesteps)
   - Better initialization (already using triangulation + smart init)

## Alternative Considered

Scipy's `differential_evolution` is available but:
- Standard DE, not ADEI
- Still faces same bottleneck issue
- No expected benefit over well-tuned CMA-ES

## Conclusion

**NOT IMPLEMENTED** - Implementing ADEI would be a significant engineering effort for minimal expected benefit. The fundamental insight from fcmaes profiling is that optimizer algorithm choice is largely irrelevant when simulations consume 99.8% of runtime.

Future optimization efforts should target simulation reduction, not optimizer improvements.

## Sources

- [ADEI Paper - MDPI Processes](https://www.mdpi.com/2227-9717/13/5/1293)
- [ResearchGate PDF](https://www.researchgate.net/publication/391115095)

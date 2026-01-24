# Powell Polish Instead of Nelder-Mead

## Experiment ID: EXP_POWELL_POLISH_001

## Status: FAILED

## Hypothesis
Powell's method performs coordinate-wise line searches which may be more efficient than Nelder-Mead's simplex operations for the 2-4D search space of heat source localization.

## Results

### Run 1: Powell with baseline config
| Metric | Value |
|--------|-------|
| Score | 1.1413 |
| RMSE | 0.1746 |
| RMSE (1-src) | 0.1334 |
| RMSE (2-src) | 0.2021 |
| Time (80 samples) | 39.2 min |
| Projected (400) | **244.9 min** |
| Budget Status | **4.2x OVER** |

### Comparison to Baseline
| Metric | Baseline (NM) | Powell | Delta |
|--------|---------------|--------|-------|
| Score | 1.1688 | 1.1413 | **-0.0275** |
| Time (400) | 58.4 min | 244.9 min | **+186.5 min** |
| Sims (2-src) | ~200 | 1000-3500 | **5-17x more** |

## Key Findings

### 1. Powell Uses Dramatically More Function Evaluations
- 1-source samples: 109-555 sims (vs ~80 for NM baseline)
- 2-source samples: **718-3466 sims** (vs ~200 for NM baseline)
- Average 2-source: ~1600 sims (8x more than NM)

### 2. Coordinate-wise Line Searches Are Inefficient for Low-D
Powell's method:
- Performs O(n) line searches per iteration (n = dimensions)
- For 4D 2-source problems: 4 line searches per iteration
- Each line search requires many evaluations to bracket the minimum

Nelder-Mead simplex:
- Updates all dimensions simultaneously
- ~5 evaluations per iteration (reflect, expand, contract, shrink)
- Much more efficient for n ≤ 10 dimensions

### 3. Time Scales Poorly with Dimensionality
| Sample Type | Dims | Powell Sims | Time Range |
|-------------|------|-------------|------------|
| 1-source | 2 | 109-555 | 27-168s |
| 2-source | 4 | 718-3466 | 175-645s |

The jump from 2D to 4D causes ~3x more simulations on average.

### 4. Worst Cases Are Extreme
Several 2-source samples took 500-645 seconds with 2000-3500 simulations:
- Sample 63: 3466 sims, 630.9s
- Sample 78: 2256 sims, 645.4s
- Sample 57: 2836 sims, 597.5s

## Root Cause Analysis

Powell's method is designed for higher-dimensional problems where computing gradients is expensive. For low-dimensional problems (2-4D), the overhead of multiple line searches per iteration dominates:

```
Powell per iteration:
  - 4 line searches (one per dimension)
  - Each line search: ~10-30 evaluations (bracket + bisection)
  - Total: 40-120 evaluations per iteration
  - 8 iterations × 3 candidates = 960-2880 evaluations

Nelder-Mead per iteration:
  - ~5 evaluations (reflect/expand/contract/shrink)
  - 8 iterations × 3 candidates = ~120 evaluations
```

This explains the observed 8-17x difference in simulation counts.

## Recommendation

**DO NOT use Powell for polish.** Nelder-Mead is optimal for this 2-4D search space.

### Why NM Wins for Heat Source Localization:
1. **Low dimensionality** (2D for 1-src, 4D for 2-src) - NM excels here
2. **Smooth objective** - NM simplex efficiently explores smooth landscapes
3. **No gradient overhead** - Both methods are derivative-free, but NM uses fewer evaluations
4. **Fixed budget works** - 8 NM iterations consistently improve without explosion

## Alternative Polish Methods to Consider

Instead of trying other scipy optimizers, consider:
1. **Gradient descent with finite differences** - only if simulation is very cheap
2. **BFGS with numerical gradient** - may converge faster than NM for smooth problems
3. **Trust region methods** - could provide more controlled convergence

However, given the time constraints and NM's proven effectiveness, keeping NM is the safest choice.

## MLflow Run
- Run ID: `8df835f50b164be6bd6e6d50fe4422ba`

## Conclusion

Powell's method is fundamentally unsuited for low-dimensional polish in this problem. The coordinate-wise line search strategy requires too many function evaluations compared to NM's simplex approach. Keep using Nelder-Mead with 8 iterations for final polish.

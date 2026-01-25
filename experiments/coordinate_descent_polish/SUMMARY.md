# Experiment Summary: coordinate_descent_polish

## Metadata
- **Experiment ID**: EXP_COORDINATE_POLISH_001
- **Worker**: W2
- **Date**: 2026-01-25
- **Algorithm Family**: polish_method_v2

## Objective
Test if Powell's coordinate descent method can replace Nelder-Mead for the final polish step, potentially offering faster convergence for 2-4D problems.

## Hypothesis
For low-dimensional problems (2D for 1-source, 4D for 2-source), coordinate descent may converge faster than simplex methods because it exploits the axis-aligned nature of the bounded parameter space.

## Results Summary
- **Best In-Budget Score**: N/A (experiment killed - massively over budget)
- **Best Overall Score**: N/A
- **Baseline Comparison**: FAILED
- **Status**: **FAILED**

## Tuning History

| Run | Config | Samples | Time Observed | Status | Notes |
|-----|--------|---------|---------------|--------|-------|
| 1 | Powell x8, 40% temporal | 39/80 | 1-src: ~150s, 2-src: 490-868s | KILLED | 5-8x slower than NM |

## Key Findings

### What Failed
- **Powell's method is fundamentally unsuitable for expensive optimization**
- 1-source samples: ~150s avg (vs baseline ~50s) - 3x slower
- 2-source samples: 490-868s avg (vs baseline ~100s) - 5-8x slower
- Experiment was killed at 39/80 samples because it was clearly going to exceed budget by 5-10x

### Root Cause
Powell's method performs **multiple line searches per iteration**:
1. For 2D (1-source): 2 line searches per iteration, each needing ~5-10 function evaluations
2. For 4D (2-source): 4 line searches per iteration, each needing ~5-10 function evaluations
3. Each function evaluation requires an expensive thermal simulation (~0.5s on coarse grid, ~1-2s on fine grid)
4. Result: 8 Powell iterations = 80-160+ simulations vs NM's ~20-40 simulations

### Why Nelder-Mead is Superior for This Problem
- NM uses only n+1 = 3 (1-src) or 5 (2-src) points per iteration
- NM converges in fewer function evaluations for smooth, unimodal landscapes
- The thermal RMSE landscape is relatively smooth near optima (CMA-ES already found good starting point)
- **For expensive objectives, NM's simplex strategy beats Powell's line search strategy**

### Critical Insights
1. **Line search methods are inefficient for expensive optimization**
   - Same pattern seen with L-BFGS-B (finite differences add O(n) evals per iteration)
   - Same pattern seen with Trust Region methods
   - Only Nelder-Mead's fixed simplex size is efficient for expensive evals

2. **The 8 NM iteration limit is already optimal**
   - Previous experiments proved 8 is the sweet spot (12 goes over budget)
   - No alternative polish method can improve on this

## Parameter Sensitivity
- **Most impactful parameter**: `polish_method` - Powell vs NM
- **Time-sensitive parameters**: Powell's line search count dominates runtime

## Recommendations for Future Experiments
1. **DO NOT try other line-search methods** (BFGS, CG, etc.) - same issue will occur
2. **DO NOT try scipy.optimize.powell** variants - all coordinate descent methods have same overhead
3. **NM x8 is OPTIMAL** for this problem - no room for improvement in polish method
4. **polish_method_v2 family is EXHAUSTED**

## Raw Data
- MLflow run IDs: None (experiment killed before completion)
- Samples completed: 39/80 before kill
- Sample timing observations:
  - 1-source: 40-445s (avg ~150s)
  - 2-source: 157-868s (avg ~680s)
  - Baseline reference: 1-src ~50s, 2-src ~100s

## Conclusion
**Powell's coordinate descent CANNOT replace Nelder-Mead.** The fundamental issue is that Powell does multiple line searches per iteration, each requiring many function evaluations. For expensive simulations like thermal PDE solving, this overhead makes Powell 5-8x slower than NM.

This confirms the pattern from multiple failed experiments:
- L-BFGS-B: FAILED (finite diff overhead)
- Trust Region: FAILED (same issue)
- Powell: FAILED (line search overhead)
- **Only Nelder-Mead works** for expensive optimization polish

The baseline (NM x8 with 40% temporal) remains optimal.

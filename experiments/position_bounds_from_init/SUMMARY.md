# Experiment Summary: position_bounds_from_init

## Metadata
- **Experiment ID**: EXP_POSITION_BOUNDS_FROM_INIT_001
- **Worker**: W2
- **Date**: 2026-01-26
- **Algorithm Family**: search_space_v2

## Objective
Use triangulation init to set tighter position bounds (±0.3 around estimate) instead of global bounds.

## Hypothesis
If triangulation gives approximate position, searching in a smaller region might help CMA-ES converge faster.

## Feasibility Analysis

### Why This is NOT_FEASIBLE

#### 1. Baseline Already Achieves Local Search
The baseline already combines:
- **Triangulation initialization**: CMA-ES starts at the triangulation estimate
- **Sigma control**: sigma0=0.15-0.20 naturally limits search to ~0.3 around init

This achieves the same effect as tight bounds, but with soft constraints instead of hard cutoffs.

#### 2. Hard Bounds Add Failure Mode
If triangulation estimate is off by more than 0.3 (which happens for difficult samples):
- Tight bounds would cut off the true solution
- CMA-ES couldn't recover by exploring beyond the bounds
- These samples would get stuck at wrong local minima

#### 3. No Benefit Over Sigma Control
| Approach | Local Search | Recovery from Bad Init | Complexity |
|----------|--------------|------------------------|------------|
| Tight bounds | Yes | NO | Higher |
| Sigma control | Yes | YES | Lower |

The baseline's sigma control achieves local search while allowing recovery if triangulation is wrong.

### What Would Happen
- **Easy samples**: Same performance (triangulation is accurate)
- **Hard samples**: WORSE performance (tight bounds cut off solution)
- **Net effect**: Likely negative

## Recommendation

**MARK AS NOT_FEASIBLE**

The experiment proposes a more brittle version of what the baseline already does:
1. Triangulation → Initialization (baseline has this)
2. Sigma → Local search (baseline has this)
3. Hard bounds → Risk without benefit (don't add)

## Conclusion

**NOT_FEASIBLE** - The baseline already achieves local search around triangulation estimate via sigma control. Adding hard bounds would only add a failure mode for samples where triangulation is inaccurate, without providing any benefit for samples where it's accurate.

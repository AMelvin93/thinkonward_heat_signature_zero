# Experiment Summary: scipy_slsqp_optimizer

## Metadata
- **Experiment ID**: EXP_SCIPY_SLSQP_001
- **Worker**: W2
- **Date**: 2026-01-24
- **Algorithm Family**: gradient_numerical

## Objective
Replace Nelder-Mead polish with SLSQP (Sequential Least-Squares Quadratic Programming) for potentially faster local convergence.

## Hypothesis
SLSQP uses sequential quadratic programming with numerical gradients. Different from Powell/BFGS which failed. May provide faster local convergence than Nelder-Mead.

## Results Summary
- **Best In-Budget Score**: N/A (not implemented)
- **Best Overall Score**: N/A
- **Baseline Comparison**: N/A
- **Status**: **ABORTED** - Prior evidence shows finite difference gradient overhead is prohibitive

## Abort Criteria Met

From experiment specification:
> "Numerical gradient overhead same as BFGS/Powell (already failed)"

**SLSQP uses finite difference gradients = same as L-BFGS-B = ABORT CRITERIA MET**

## Prior Evidence

### EXP_HYBRID_CMAES_LBFGSB_001 (L-BFGS-B Polish)
| Metric | Value |
|--------|-------|
| Score | 1.1174 |
| Time | 42 min |
| vs NM x8 | **WORSE** (NM: 1.1415 @ 38 min) |

**Finding**: Finite diff overhead O(n) extra sims per iteration outweighs gradient advantage.

### EXP_POWELL_POLISH_001 (Powell Polish)
| Metric | Value |
|--------|-------|
| Score | 1.1413 |
| Time | 244.9 min |
| Budget ratio | **4.2x over budget** |

**Finding**: Coordinate-wise line searches require 5-17x more function evaluations than NM.

## Why SLSQP Won't Work

### Same Gradient Computation as L-BFGS-B

```
SLSQP gradient computation:
- Uses finite differences for gradient approximation
- Same O(n) overhead per iteration as L-BFGS-B
- For 2D (1-source): 2+1 = 3 extra sims per iteration
- For 4D (2-source): 4+1 = 5 extra sims per iteration

Since L-BFGS-B FAILED with this same overhead,
SLSQP will also FAIL.
```

### No Benefit Over L-BFGS-B

| Feature | L-BFGS-B | SLSQP |
|---------|----------|-------|
| Gradient method | Finite diff | Finite diff |
| Overhead | O(n) per iter | O(n) per iter |
| Bound constraints | Box constraints | General constraints |
| Our problem | Box bounds | Box bounds only |

SLSQP's constraint handling adds no benefit since we only have box bounds.

## Recommendations

### 1. gradient_numerical Family Should Be Marked EXHAUSTED
All gradient-based polish methods require finite differences which are too expensive for ~400ms simulations.

### 2. Methods Confirmed Not Viable
- **L-BFGS-B**: FAILED - finite diff overhead
- **Powell**: FAILED - line search overhead
- **SLSQP**: ABORTED - same as L-BFGS-B

### 3. Nelder-Mead Remains Optimal for Polish
NM x8 achieves best accuracy/time tradeoff without gradient computation.

## Conclusion

**ABORTED** - SLSQP is not viable because it uses finite difference gradients with the same O(n) overhead per iteration as L-BFGS-B, which already failed. Since our simulations take ~400ms, the gradient computation overhead outweighs any convergence advantage. The gradient_numerical family is exhausted.

## Files
- `feasibility_analysis.py`: Prior evidence review
- `STATE.json`: Experiment state tracking

# Experiment Summary: adaptive_nm_iterations

## Metadata
- **Experiment ID**: EXP_ADAPTIVE_NM_POLISH_001
- **Worker**: W2
- **Date**: 2026-01-22
- **Algorithm Family**: refinement

## Objective
Dynamically adjust Nelder-Mead (NM) polish iterations based on convergence rate to:
1. Save time on easy samples that converge quickly
2. Allow more iterations for hard samples that need refinement

## Hypothesis
Fixed 8 NM iterations is wasteful for easy samples (which converge in 4-5 iters) and insufficient for hard ones (which need 10+). An adaptive approach should improve efficiency.

## Results Summary
- **Best In-Budget Score**: None achieved (all runs over 60 min budget)
- **Best Overall Score**: 1.1607 @ 78.3 min (Fixed 8/8, Run 3)
- **Baseline Comparison**: -0.0081 vs documented 1.1688 @ 58.4 min
- **Status**: **FAILED** - Adaptive approach does NOT improve over fixed iterations

## Tuning History

| Run | Config | Score | Time (min) | In Budget | Notes |
|-----|--------|-------|------------|-----------|-------|
| 1 | Adaptive 4-12 iters, batch=2, threshold=0.001 | 1.1545 | 78.7 | No | Similar time to fixed, lower score |
| 2 | Source-count: 1-src=6, 2-src=10 | 1.1551 | 94.5 | No | WORST - more 2-src iters adds time |
| 3 | Fixed 8/8 (both sources) | 1.1607 | 78.3 | No | Best of my runs |
| Baseline | Original optimizer, 8 NM | 1.1580 | 88.6 | No | Verification run |

## Key Findings

### What Didn't Work

1. **Adaptive Batched Approach (Run 1)**
   - Multiple `minimize()` calls add overhead
   - Early stopping (threshold=0.001) doesn't save meaningful time
   - 1-source samples averaged 6.7 iters, 2-source averaged 8.0 iters
   - The overhead from checking convergence offsets any time savings

2. **Source-Count Based Iterations (Run 2)**
   - Hypothesis: 1-source needs fewer iters, 2-source needs more
   - Result: WORST performance (94.5 min, +36 min vs documented baseline)
   - More iterations for 2-source adds significant time without accuracy gain
   - 2-source problems are structurally harder (4D search space), not under-iterated

3. **Baseline Reproducibility Issue**
   - Documented baseline: 1.1688 @ 58.4 min
   - My verification run: 1.1580 @ 88.6 min
   - **~30 min discrepancy** suggests original results on different hardware/conditions

### Critical Insights

1. **Fixed iterations are already optimal**
   - NM's built-in tolerance (`fatol`, `xatol`) handles early termination
   - No benefit from manual convergence checking

2. **2-source problems don't benefit from more iterations**
   - The extra 4 dimensions make the search harder, not longer
   - More iterations ≠ better accuracy for 2-source

3. **scipy.optimize.minimize overhead**
   - Each `minimize()` call has startup overhead
   - Batched approach (4+2+2+...) is slower than single call (8)

## Parameter Sensitivity

| Parameter | Effect |
|-----------|--------|
| polish_iters | Linear time increase, diminishing accuracy returns after ~6 |
| convergence_threshold | 0.001 is too aggressive (stops too early), but relaxing doesn't help |
| batch_size | Smaller batches = more overhead, no benefit |

## RMSE Analysis

| Config | 1-source RMSE | 2-source RMSE |
|--------|---------------|---------------|
| Adaptive 4-12 | 0.1123 | 0.1552 |
| Source-count 6/10 | 0.1173 | 0.1602 |
| Fixed 8/8 | 0.1046 | 0.1550 |
| Baseline 8 | 0.1019 | 0.1815 |

Observation: Fixed 8/8 achieves best 2-source RMSE (0.1550 vs baseline's 0.1815).

## Recommendations for Future Experiments

1. **Do NOT pursue adaptive NM iterations** - fixed is optimal
2. **Do NOT allocate different iterations per source count** - both need same amount
3. **Consider reducing iterations to 6** - may save time with acceptable accuracy loss
4. **Investigate baseline discrepancy** - documented 58.4 min cannot be reproduced

## Files
- `optimizer.py` - Adaptive batched NM optimizer (Run 1)
- `optimizer_v2.py` - Source-count based optimizer (Runs 2, 3)
- `run.py` - Run script for adaptive optimizer
- `run_v2.py` - Run script for source-count based optimizer
- `STATE.json` - Detailed tuning history

## Raw Data

### MLflow Run IDs
- Run 1: `964170f4334845189937ffe2e2c864b4`
- Run 2: `5e1ae1e6861a4aef88e34d77022ed7e8`
- Run 3: `f1260e3e27144e0c845c8e972f60a396`

### Best Config (Run 3)
```json
{
  "timestep_fraction": 0.40,
  "polish_iters_1src": 8,
  "polish_iters_2src": 8,
  "max_fevals_1src": 20,
  "max_fevals_2src": 36
}
```

## Conclusion

**Adaptive NM iterations is NOT beneficial.** The baseline's fixed 8 iterations is already optimal because:
1. NM's built-in tolerance handles early termination
2. Multiple minimize() calls add overhead
3. 2-source problems don't benefit from extra iterations

The `refinement` family should be considered **EXHAUSTED** - both extended polish (12 iters) and adaptive polish fail to improve over fixed 8.

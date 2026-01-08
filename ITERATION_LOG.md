# Iteration Log - Heat Signature Zero

**Session Start**: 2026-01-08 04:05:01
**Max Iterations**: 10
**Current Best**: 1.0116 @ 58.6 min (SmartInitOptimizer)

---

## Iteration Summary
| # | Approach | Config | Score | Time | Status |
|---|----------|--------|-------|------|--------|
| 1 | Early CMA-ES Termination | 12/22, thresh=1e-4 | 1.0218 | 57.4 min | ✅ Best but no early stops |
| 1b | Early CMA-ES Termination | 12/22, thresh=0.01 | 1.0115 | 57.3 min | ~Same score, 25% early stops |
| 1c | Early CMA-ES Termination | 15/28, thresh=0.005 | 1.0156 | 65.9 min | ❌ Over budget |

---

## Detailed Log

## Iteration 1 - 2026-01-08 04:05
- **Approach**: Early CMA-ES Termination (Priority 6)
- **Hypothesis**: Stop CMA-ES early on easy samples to save time for hard ones
- **Implementation**: `experiments/early_termination/`

### Test Results

| Config | Score | 1-src RMSE | 2-src RMSE | Time | Early Stop % | Status |
|--------|-------|------------|------------|------|--------------|--------|
| 12/22, thresh=1e-4, pat=3 | **1.0218** | 0.183 | 0.295 | 57.4 min | 0% | ✅ Best |
| 12/22, thresh=0.01, pat=2 | 1.0115 | 0.215 | 0.307 | 57.3 min | 25% | ~Same |
| 15/28, thresh=0.005, pat=2 | 1.0156 | 0.205 | 0.289 | 65.9 min | 36% | ❌ Over budget |

### Key Findings

1. **Conservative threshold (1e-4) = no early termination** - CMA-ES improvements are always larger than 1e-4
2. **Aggressive threshold (0.01) = marginal savings** - 1.3 min faster, same score
3. **First run scored 1.0218** - Likely run-to-run variance (no early stops triggered)
4. **Early termination is NOT a significant improvement path**

### Root Cause Analysis

- **Why no early termination with 1e-4?** CMA-ES with small populations (4-6) makes large jumps per generation. With only 2-5 generations (12-22 fevals / popsize), improvement per gen > 1e-4.
- **Why aggressive threshold doesn't help much?** Easy samples finish quickly anyway. Hard 2-source samples don't stagnate - they need all fevals.
- **Why score variance?** Random seed affects sample ordering and init selection.

### Conclusion

Early termination provides **marginal time savings** (~1-2 min) but **no significant score improvement**. The baseline SmartInitOptimizer (12/22) remains the best approach.

**Recommendation**: Mark Priority 6 as TESTED - NOT EFFECTIVE. Move to next priority.

---


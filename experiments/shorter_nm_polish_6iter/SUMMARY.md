# Shorter NM Polish Experiment

## Status: MIXED - 7 iterations optimal in-budget, but baseline unreproducible

## Hypothesis
8 NM iterations may be overkill. Testing if 6-7 iterations can save time with minimal accuracy loss.

## Results Summary

| Run | Config | Score | Time (min) | Budget Status |
|-----|--------|-------|------------|---------------|
| 1 | nm_6iter | 1.1489 | 53.5 | IN budget |
| 2 | nm_7iter | 1.1516 | 56.1 | IN budget |
| 3 | nm_8iter_baseline | 1.1554 | 65.8 | OVER budget |

**Claimed Baseline**: 1.1689 @ 58.2 min (from hopping_with_tabu_memory)

## Key Findings

### 1. 8 Iterations Goes Over Budget!
The baseline config with 8 NM iterations actually takes **65.8 min**, not the claimed 58.2 min. This suggests:
- The previous baseline run may have had different system conditions
- Or variance in processing time is higher than expected

### 2. All Scores Below Claimed Baseline
- Best: 1.1554 (8 iter, over budget)
- All configs scored 0.0135-0.0200 lower than claimed baseline of 1.1689

### 3. Diminishing Returns
| NM Iterations | Score | Time | Score/min |
|---------------|-------|------|-----------|
| 6 | 1.1489 | 53.5 | 0.0215 |
| 7 | 1.1516 | 56.1 | 0.0205 |
| 8 | 1.1554 | 65.8 | 0.0176 |

More iterations = better accuracy but diminishing returns.

### 4. Best In-Budget Configuration
**nm_7iter @ 1.1516, 56.1 min** is the best option that stays within budget.

## Analysis

### RMSE Breakdown
| Config | RMSE 1-src | RMSE 2-src |
|--------|------------|------------|
| nm_6iter | 0.134832 | 0.221059 |
| nm_7iter | 0.130246 | 0.218347 |
| nm_8iter | 0.134085 | 0.204070 |

7 iterations gives the best 1-source RMSE, while 8 iterations has the best 2-source RMSE.

## Conclusion

**The claimed baseline of 1.1689 @ 58 min appears unreproducible.** In this run:
- 8 iterations took 65.8 min (over budget)
- Best achievable score was 1.1554 (over budget) or 1.1516 (in budget with 7 iter)

## Recommendations

1. **Use 7 NM iterations** for in-budget runs (~56 min, score 1.1516)
2. **Re-verify baseline** - the claimed 1.1689 may have been a variance outlier
3. **Document run-to-run variance** - scores vary by ~0.01-0.02 between runs
4. **Current best in-budget**: nm_7iter @ 1.1516, 56.1 min

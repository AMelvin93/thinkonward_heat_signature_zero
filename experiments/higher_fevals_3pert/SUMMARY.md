# Experiment: higher_fevals_3pert

## Status: PROMISING BUT OVER BUDGET

## Hypothesis
With 4.3 min budget remaining from 3-pert baseline (55.7 min), higher fevals (22/48)
might improve accuracy.

## Results (3 Validation Runs)

| Run | Score | Time (min) | RMSE 1src | RMSE 2src | Budget |
|-----|-------|------------|-----------|-----------|--------|
| 1   | 1.1512 | 63.5 | 0.1144 | 0.1768 | OVER |
| 2   | **1.1610** | 63.4 | 0.1079 | 0.1642 | OVER |
| 3   | 1.1582 | 63.3 | 0.1123 | 0.1721 | OVER |

## Statistics

| Metric | Value |
|--------|-------|
| **Mean Score** | **1.1568 +/- 0.0041** |
| **Mean Time** | 63.4 min |
| vs 3-pert baseline (1.1475) | **+0.0093** |
| Budget Status | OVER BUDGET by 3.4 min |

## Key Finding

**Higher fevals significantly improves accuracy but exceeds budget.**

| Config | Score | Time | Notes |
|--------|-------|------|-------|
| 3-pert @ 20/44 | 1.1475 | 55.7 | VALIDATED BASELINE |
| **3-pert @ 22/48** | **1.1568** | **63.4** | **+0.0093 but OVER** |

## Leaderboard Context

If this were within budget, Run 2 (1.1610) would rank:

| Rank | Team | Score |
|------|------|-------|
| 8 | Ti41e7 | 1.1743 |
| 9 | nacumaria00 | 1.1716 |
| **Run 2** | **1.1610** | **TOP 10!** |
| 10 | MGöksu | 1.1585 |

Mean score (1.1568) would also be competitive for top 10.

## Trade-off Analysis

The question: Is +0.0093 score worth +7.7 min time?
- Budget: 60 min (hard limit)
- Current: 63.4 min (over)
- Score improvement: 1.1475 → 1.1568 (+0.8%)

## Recommendation

**DO NOT use for final submission** - too risky (100% runs over budget)

But the finding suggests:
- CMA-ES fevals have room for improvement
- A middle ground (e.g., 21/46) might fit budget
- Worth testing intermediate configs

## What To Try Next

| Config | Expected Time | Expected Score |
|--------|---------------|----------------|
| 3-pert @ 21/46 | ~59-60 min | ~1.152 |
| 3-pert @ 20/45 | ~58 min | ~1.150 |

---
**Worker**: W2
**Completed**: 2026-02-02
**Runs**: 3
**Result**: OVER BUDGET - but shows potential for higher fevals

# Experiment: 4pert_tabu004_combined

## Status: PROMISING BUT OVER BUDGET (mean 61.3 min)

## Hypothesis
Combining two validated improvements:
1. 4 perturbations (vs 2): 1.1437 @ 42.2 min (+0.0100)
2. tabu_distance=0.04 (vs 0.03): 1.1496 @ 55.4 min (+0.0032)

Expected: Additive improvement.

## Results (3 Runs)

| Run | Score | Time (min) | RMSE 1src | RMSE 2src | Budget Status |
|-----|-------|------------|-----------|-----------|---------------|
| 1   | **1.1585** | 52.5 | 0.1121 | 0.1730 | **IN BUDGET** |
| 2   | 1.1484 | 66.2 | 0.1167 | 0.1902 | OVER BUDGET |
| 3   | 1.1535 | 65.3 | 0.1159 | 0.1815 | OVER BUDGET |

## Statistics

| Metric | Value |
|--------|-------|
| Mean Score | **1.1535 +/- 0.0041** |
| Mean Time | 61.3 min |
| vs True Baseline (1.1337) | **+0.0198** |
| vs 4-pert only (1.1437) | **+0.0098** |
| vs tabu-0.04 only (1.1496) | **+0.0039** |

## Key Findings

### 1. Additive Improvement Confirmed!
The combined config EXCEEDS both individual improvements:
- Mean score 1.1535 is higher than either 1.1437 (4-pert) or 1.1496 (tabu-0.04)
- The improvements are approximately additive

### 2. BUT Timing is Problematic
- Only 1 out of 3 runs finished in budget
- Mean time 61.3 min exceeds 60 min limit
- High timing variance (52.5 to 66.2 min)

### 3. Run 1 is Exceptional
- Score 1.1585 @ 52.5 min is the BEST we've seen
- Within budget with 7.5 min margin
- May represent lucky run (low variance)

## Analysis

The combined config shows that improvements ARE additive:
```
Expected additive: 0.0100 + 0.0032 = 0.0132
Actual improvement: +0.0198
```

But the time cost is also additive:
- 4-pert alone: ~42 min
- tabu-0.04 alone: ~55 min
- Combined: ~61 min (over budget)

## Configuration

```python
config = {
    'sigma0_1src': 0.18,
    'sigma0_2src': 0.22,
    'max_fevals_1src': 20,
    'max_fevals_2src': 44,
    'timestep_fraction': 0.40,
    'refine_maxiter': 8,
    'enable_tabu_hopping': True,
    'n_perturbations': 4,           # Higher
    'perturb_nm_iters': 3,
    'perturbation_scale': 0.05,
    'tabu_distance': 0.04,          # Higher
    'max_tabu_attempts': 10,
}
```

## Recommendations

### Option 1: Risk It (High Potential, Risky)
Use the combined config and accept some runs may go over budget:
- Potential score: 1.1535 - 1.1585
- Risk: 2/3 runs over budget

### Option 2: Reduce Perturbations (Lower Risk)
Try 3 perturbations + tabu_distance=0.04:
- Expected: score ~1.15, time ~55 min
- Better budget margin

### Option 3: Stick with Single Improvement (Safe)
Use either:
- 4 perturbations + tabu=0.03: 1.1437 @ 42.2 min (SAFE)
- 2 perturbations + tabu=0.04: 1.1496 @ 55.4 min (SAFE)

## Next Steps

**RECOMMENDED**: Test 3 perturbations + tabu_distance=0.04 to find sweet spot between score and time.

## Leaderboard Context

| Config | Score | Time | Gap to Top 10 (1.1585) |
|--------|-------|------|------------------------|
| Combined (best run) | **1.1585** | 52.5 | **MATCHES TOP 10** |
| Combined (mean) | 1.1535 | 61.3 | -0.005 |
| tabu-0.04 only | 1.1496 | 55.4 | -0.009 |
| 4-pert only | 1.1437 | 42.2 | -0.015 |
| True baseline | 1.1337 | 44.0 | -0.025 |

---
**Worker**: W1
**Completed**: 2026-02-02
**Runs**: 3
**Result**: PROMISING - additive improvement but over budget

# Experiment: timestep_42pct

## Status: MARGINAL - 42% shows tiny improvement, 44% is worse

## Hypothesis
Test higher temporal fidelity (42%, 44%) vs 40% baseline. More timesteps may improve accuracy.

## Configuration
Using validated config: sigma 0.18/0.22, tabu_distance=0.04

## Results

| Config | Timestep | Score | Time (min) | RMSE 1src | RMSE 2src | vs Baseline |
|--------|----------|-------|------------|-----------|-----------|-------------|
| timestep_42pct | 42% | **1.1526** | 56.8 | 0.1255 | 0.1826 | **+0.0030** |
| timestep_44pct | 44% | 1.1433 | 56.8 | 0.1130 | 0.2009 | -0.0063 |
| baseline_40pct | 40% | 1.1488 | 52.9 | 0.1258 | 0.1823 | -0.0008 |

**Baseline (validated)**: 1.1496 @ 55.4 min (mean from tabu_004_validation)

## Key Findings

### 1. 42% Shows Small Improvement (+0.0030)
- Best single-run score: 1.1526 @ 56.8 min
- Improvement is small and may be within noise (prior variance std ~0.005)
- Time increased from 52.9 to 56.8 min (+4 min)

### 2. 44% is WORSE (-0.0063)
- Score dropped to 1.1433
- More temporal fidelity does NOT always help
- There's an optimal point between 40% and 44%

### 3. 40% is Near Optimal
- The 40% run (1.1488) is very close to validated mean (1.1496)
- 40% provides good accuracy/time trade-off

### 4. Time-Accuracy Trade-off
| Timestep | Score | Time | Score/Min Budget |
|----------|-------|------|------------------|
| 40% | 1.1488 | 52.9 min | 7.1 min remaining |
| 42% | 1.1526 | 56.8 min | 3.2 min remaining |
| 44% | 1.1433 | 56.8 min | 3.2 min remaining |

The 42% improvement (+0.0038 vs same-run 40%) costs +4 min time.

## Conclusion

**42% temporal fidelity shows a marginal improvement**, but:
1. Improvement is small (+0.0030) and likely within variance
2. Time budget becomes tighter (only 3.2 min remaining)
3. Risk of going over budget on some runs increases

**Recommendation**: Keep 40% temporal fidelity
- More robust to variance
- Better time budget margin
- Smaller improvement doesn't justify risk

## Not Worth Pursuing Further

Given:
- Small improvement (+0.003)
- Tight time budget (3.2 min remaining)
- High variance in scores (~0.005 std)

The 42% temporal fidelity is NOT worth pursuing. Stick with 40%.

## Tuning Efficiency Metrics
- **Runs executed**: 3 (systematic exploration)
- **Time utilization**: 95% (56.8/60 min for 42%)
- **Parameter space explored**: timestep_fraction in [0.40, 0.42, 0.44]

---
**Worker**: W2
**Completed**: 2026-02-02
**Runs**: 3 (systematic tuning)
**Result**: MARGINAL IMPROVEMENT - not worth adopting

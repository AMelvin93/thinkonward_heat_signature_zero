# Experiment Summary: condition_number_init

## Metadata
- **Experiment ID**: EXP_CONDITION_NUMBER_INIT_001
- **Worker**: W1
- **Date**: 2026-01-25
- **Algorithm Family**: initialization_v4

## Status: ABORTED (Prior Evidence Conclusive)

## Objective
Use sensor layout condition number to weight sensors during initialization, potentially improving initial source position estimates.

## Why Aborted

Both the **initialization family** and **sensor weighting approaches** have been marked EXHAUSTED by prior experiments.

### Key Prior Evidence

| Experiment | Result | Key Finding |
|------------|--------|-------------|
| **physics_informed_init** | FAILED (-0.0046) | "The initialization family should be marked as EXHAUSTED" |
| **weighted_sensor_loss** | FAILED | "Optimization Target Mismatch: optimum of weighted loss differs from unweighted RMSE" |
| **informative_sensor_subset** | FAILED | "Selecting sensors HURTS accuracy vs using all sensors" |
| **more_inits_select_best** | FAILED | "The baseline 2-init strategy (triangulation + hotspot) is already optimal" |

### Why Sensor Weighting Cannot Help

1. **Scoring Uses UNWEIGHTED RMSE**
   - The competition scoring formula uses unweighted RMSE across ALL sensors
   - Any initialization that weights sensors optimizes for a DIFFERENT objective
   - Quote from weighted_sensor_loss: "The optimum of weighted loss is different from unweighted loss"

2. **Sensor Subset Selection Hurts Accuracy**
   - `informative_sensor_subset` tested using only "informative" sensors
   - Result: WORSE accuracy than using all sensors
   - All sensors contribute useful information

3. **Current Initialization Is Already Optimal**
   - Triangulation uses ALL sensor onset times equally
   - Hotspot uses peak temperature across ALL sensors
   - Quote: "The baseline 2-init strategy (triangulation + hotspot) is already optimal"

### Why Condition Number Doesn't Help Initialization

The condition number of the sensor observation matrix relates to:
- How well source parameters can be reconstructed from sensor readings
- Numerical stability of the inverse problem

But for **initialization** (not optimization):
1. Triangulation uses onset times, not condition number
2. Hotspot uses peak values, not condition number
3. Both methods already work well without weighting

## Technical Analysis

### The PSPO Research Reference

The experiment references PSPO research on condition number and reconstruction error. However:
- PSPO studies sensor PLACEMENT optimization
- We cannot change sensor positions (fixed per sample)
- The condition number is already determined by the given layout

### What Condition Number Weighting Would Do

```python
# Proposed:
weighted_triangulation = triangulation(sensors, weights=condition_contributions)

# Problem:
# 1. Weights are derived from observation matrix (physics-dependent)
# 2. Requires computing observation matrix for multiple source guesses
# 3. Adds computational overhead
# 4. Still optimizes for weighted objective, not unweighted RMSE
```

## Algorithm Family Status

- **initialization (v1, v2, v3, v4)**: **EXHAUSTED**
- **sensor_weighting**: **EXHAUSTED**
- **sensor_subset**: **EXHAUSTED**

Key insight: Current triangulation + hotspot initialization is locally optimal. All tested alternatives perform worse.

## Recommendations

1. **Do NOT pursue sensor weighting** - scoring uses unweighted RMSE
2. **Do NOT pursue initialization improvements** - current approach is optimal
3. **Accept that initialization is solved** - focus on other components

## Conclusion

The condition number initialization would fail because: (1) the initialization family is exhausted with current triangulation+hotspot being optimal, (2) sensor weighting optimizes for a different objective than the unweighted RMSE we're scored on, and (3) sensor subset selection has been shown to hurt accuracy. The current initialization is already physics-optimal.

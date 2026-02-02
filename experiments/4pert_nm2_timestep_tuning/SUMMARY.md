# Experiment: 4pert_nm2_timestep_tuning

## Status: COMPLETED - No improvement (0.40 is optimal)

## Purpose
Test if different timestep_fraction values can improve the 4pert_nm2 config.

## Results

| Config | timestep_fraction | Score | Time (min) | vs Baseline |
|--------|-------------------|-------|------------|-------------|
| timestep_0.35 | 0.35 | 1.1469 | 49.0 | -0.0013 |
| **timestep_0.40** | **0.40** | **1.1483** | **53.1** | **+0.0001** |
| timestep_0.45 | 0.45 | 1.1414 | 55.6 | -0.0068 |

**Baseline**: 1.1482 @ 51.7 min (timestep_fraction=0.40)

## Key Finding

**timestep_fraction=0.40 is already optimal.**

- Lower (0.35): Loses information, slight score drop (-0.0013)
- Higher (0.45): Slower without benefit, significant score drop (-0.0068)

## Conclusion

**No improvement found. timestep_fraction=0.40 is optimal.**

This confirms the current 4pert_nm2 config is well-tuned.

## Parameter Space Status

With this experiment, ALL major tuning dimensions have been exhausted:

| Parameter | Status | Optimal Value |
|-----------|--------|---------------|
| sigma_1src | EXHAUSTED | 0.18 |
| sigma_2src | EXHAUSTED | 0.22 |
| max_fevals_1src | EXHAUSTED | 20 |
| max_fevals_2src | EXHAUSTED | 44 |
| n_perturbations | EXHAUSTED | 4 |
| perturb_nm_iters | EXHAUSTED | 2 |
| perturbation_scale | EXHAUSTED | 0.05 |
| tabu_distance | EXHAUSTED | 0.04 |
| refine_maxiter | EXHAUSTED | 8 |
| **timestep_fraction** | **EXHAUSTED** | **0.40** |

---
**Worker**: W1
**Completed**: 2026-02-02
**Runs**: 3
**Result**: No improvement - 0.40 is optimal

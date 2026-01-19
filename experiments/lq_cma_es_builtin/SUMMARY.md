# Experiment Summary: lq_cma_es_builtin

## Metadata
- **Experiment ID**: EXP_LQ_CMAES_001
- **Worker**: W1
- **Date**: 2026-01-18 to 2026-01-19
- **Algorithm Family**: surrogate_lq

## Objective
Test pycma's built-in lq-CMA-ES (linear-quadratic surrogate model) as a drop-in replacement for standard CMA-ES to reduce function evaluations and runtime.

## Hypothesis
The global quadratic surrogate in lq-CMA-ES can reduce function evaluations by 2-6x while maintaining accuracy, based on GECCO 2019 paper findings.

## Results Summary
- **Best In-Budget Score**: NONE (all runs over budget)
- **Best Overall Score**: 0.253 RMSE @ 64.1 min
- **Baseline Comparison**: **FAILED** - RMSE 0.18/0.30 vs baseline 0.14/0.21 (29-71% worse)
- **Status**: **FAILED**

## Tuning History

| Run | Config Changes | RMSE | Time (min) | In Budget | Notes |
|-----|---------------|------|------------|-----------|-------|
| 1 | Initial: baseline params | 0.268 | 66.7 | No | Scoring format errors |
| 2 | Fixed predictions format | 0.253 | 64.1 | No | Best RMSE achieved |
| 3 | Fixed scoring API | 0.291 | 61.8 | No | Higher RMSE, more variance |

## Key Findings

### What Didn't Work
1. **fmin_lq_surr API mismatch**: The function only returns ONE final solution (xfavorite), not intermediate population members. Our scoring requires multiple diverse candidates.

2. **No speedup achieved**: lq-CMA-ES was 7-17% SLOWER than baseline, not faster. The surrogate model overhead didn't pay off with our small evaluation budget (10-18 fevals per init).

3. **Worse accuracy**: RMSE was 29% worse for 1-source and 43-71% worse for 2-source problems.

4. **Diversity penalty**: Only 1 valid candidate per sample vs 3 for baseline. The diversity bonus (0.3 * N_valid/3) is severely impacted.

### Critical Insights
1. **API design mismatch**: lq-CMA-ES's `fmin_lq_surr` is designed for single-objective optimization returning ONE best solution. Our problem needs MULTIPLE diverse candidates for the scoring formula.

2. **Surrogate data requirement**: The quadratic surrogate needs sufficient data points to build an accurate model. With only 10-18 fevals per initialization, there's not enough data.

3. **RMSE landscape complexity**: The thermal inverse problem may have a complex landscape that a simple quadratic model cannot capture well.

4. **Covariance matters more than surrogate**: CMA-ES's power comes from covariance adaptation, which is preserved in standard CMA-ES. The surrogate doesn't provide additional benefit.

## Parameter Sensitivity
- **max_fevals**: Limited budget (20/36) insufficient for surrogate to build accurate model
- **sigma0**: Default values from baseline (0.15/0.20) used

## Recommendations for Future Experiments

1. **ABANDON lq-CMA-ES for this problem**: The API doesn't match our multi-candidate requirement. Would need to use SurrogatePopulation class with ask/tell to access intermediate solutions.

2. **Focus on multi-candidate approaches**: Any optimization method needs to provide MULTIPLE diverse candidates, not just one best solution.

3. **Try different surrogate approaches**:
   - Use pycma's SurrogatePopulation with ask/tell interface (more complex)
   - POD (Proper Orthogonal Decomposition) might work better for physics-based surrogates

4. **Keep standard CMA-ES**: The covariance adaptation is essential and working well.

5. **What W0 should try next**:
   - **EXP_SEQUENTIAL_HANDOFF_001** (CMA-ES + NM) - this preserves multi-candidate generation
   - **EXP_BAYESIAN_OPT_001** - different optimization paradigm
   - **EXP_MULTIFID_OPT_001** - multi-fidelity with proper grid ratios

## Raw Data
- MLflow run IDs:
  - 1c2d93d8edaf418f9a52908a554f3337
  - 4372b4befb964bfd82c72a73eb344e55
  - b44884fc53354e9b97aba0140c9aca31
- Best config: Standard baseline parameters (no tuning helped)
- Failure mode: API design mismatch - single solution vs multi-candidate requirement

## Conclusion

**lq-CMA-ES is NOT suitable for this problem.**

The fundamental issue is that `cma.fmin_lq_surr` is a high-level API designed to return ONE best solution, while our scoring formula rewards MULTIPLE diverse candidates. Using lq-CMA-ES would require reimplementing with the lower-level SurrogatePopulation API to access intermediate solutions, which negates the "built-in" advantage we hoped to leverage.

Additionally, the surrogate provided no speedup and reduced accuracy, suggesting the quadratic model doesn't fit our RMSE landscape well.

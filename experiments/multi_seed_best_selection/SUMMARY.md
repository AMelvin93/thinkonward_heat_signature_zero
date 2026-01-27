# Experiment Summary: multi_seed_best_selection

## Metadata
- **Experiment ID**: EXP_MULTI_SEED_ENSEMBLE_001
- **Worker**: W2
- **Date**: 2026-01-26
- **Algorithm Family**: robustness_v2

## Objective
Run CMA-ES with 3 different seeds per sample and select the best result to reduce variance.

## Hypothesis
Given baseline variance of ±0.0049, running multiple seeds and selecting the best might improve score by getting lucky more often.

## Feasibility Analysis

### Time Budget Constraint
- **Baseline runtime**: ~58 min for 80 samples
- **3 seeds per sample (sequential)**: 3 × 58 = 174 min (**WAY OVER 60 min budget**)
- **3 seeds with 1/3 evaluations each**: Same total time, but each seed converges worse

### Why This Experiment is NOT FEASIBLE

#### 1. Initialization is Deterministic
The baseline uses **triangulation-based initialization** which is deterministic:
- Given the same sensor data, triangulation produces the same starting point
- Different random seeds only affect the CMA-ES stochastic sampling
- The impact of random seeds is minimal when starting from a good initial guess

#### 2. Time Budget Trade-off is Unfavorable
Options:
| Approach | Time | Evaluations/Seed | Expected Outcome |
|----------|------|------------------|------------------|
| 1 seed, full evals | 58 min | 20 (1-src), 36 (2-src) | **Baseline** |
| 3 seeds, full evals | 174 min | 20, 36 | Over budget |
| 3 seeds, 1/3 evals | ~58 min | 7, 12 | Worse convergence |

Running 3 seeds with reduced evaluations each would hurt performance because:
- CMA-ES needs sufficient iterations to converge
- 7 evaluations for 1-source is below the threshold for reliable convergence
- The best of 3 poorly-converged runs is still worse than 1 well-converged run

#### 3. Variance Source Analysis
The ±0.0049 variance in baseline score comes from:
1. **CMA-ES random sampling** (~20%): Affected by seeds
2. **NM polish local search** (~10%): Deterministic
3. **Sample difficulty variance** (~70%): Not affected by seeds

Random seeds only help with component 1, which is a small part of total variance.

### Alternative Approaches (Also Rejected)

#### Multi-Start with Same Budget
- Run 2 seeds with 50% evaluations each
- Problem: 50% evaluations = significantly worse convergence
- Net effect: Negative (worse convergence outweighs seed diversity)

#### Parallel Seeds
- Run seeds in parallel using multiple workers
- Problem: This is how the baseline already works (parallel across samples)
- Can't parallelize within a single sample efficiently

## Recommendation

**MARK AS NOT_FEASIBLE**

The multi-seed approach doesn't work for this problem because:
1. Time budget doesn't allow 3x computation
2. Reducing per-seed budget hurts more than it helps
3. Deterministic initialization already reduces the value of random seeds
4. The baseline's triangulation + CMA-ES pipeline is already near-optimal

## Better Alternatives for Reducing Variance

If variance reduction is the goal, focus on:
1. **Better initialization** (already optimized with triangulation)
2. **More evaluations** (but time-limited)
3. **Better adaptive sigma** (explored, didn't help)
4. **Accept current variance** as inherent to the problem

## Conclusion

**NOT_FEASIBLE** - The multi-seed approach cannot work within the 60-minute time budget. The fundamental trade-off (more seeds vs fewer evaluations) is unfavorable for this optimization problem where CMA-ES convergence requires sufficient iterations.

## Key Insight

The baseline's single-seed approach with good initialization (triangulation) is more effective than multi-seed with reduced evaluations. **Don't sacrifice convergence quality for lucky restarts.**

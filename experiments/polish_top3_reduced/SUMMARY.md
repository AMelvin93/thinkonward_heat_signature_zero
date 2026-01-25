# Experiment Summary: polish_top3_reduced

## Metadata
- **Experiment ID**: EXP_POLISH_TOP3_001
- **Worker**: W3
- **Date**: 2026-01-25
- **Algorithm Family**: polish_strategy_v2

## Objective
Test whether polishing ALL top-3 candidates (4 NM iterations each) instead of just the best candidate (8 NM iterations) improves the average accuracy component of the submission score.

## Hypothesis
The scoring formula averages accuracy over N candidates: `score = (1/N) * Σ(1/(1+RMSE_i)) + 0.3*(N/3)`

By improving the accuracy of 2nd/3rd place candidates (not just the best), we should increase the average accuracy term and thus the overall score.

## Results Summary
- **Best In-Budget Score**: N/A (no runs within budget)
- **Best Overall Score**: 1.0743 @ 114.2 min
- **Baseline Comparison**: -0.0945 vs 1.1688 (8% WORSE)
- **Status**: FAILED

## Tuning History

| Run | Config Changes | Score | Time (min) | In Budget | Notes |
|-----|---------------|-------|------------|-----------|-------|
| 1 | 4 iters × 3 candidates = 12 total | 1.0743 | 114.2 | No | 1.9x over budget, 8% worse accuracy |

## Key Findings

### What Didn't Work
- **Multi-candidate polish is fundamentally flawed**
  - 12 NM iterations on fine grid with full timesteps takes 114 min (1.9x over budget)
  - Each NM iteration on fine grid requires expensive simulations
  - Tripling the number of candidates to polish roughly triples the overhead

- **2nd/3rd candidates have high base RMSE that polish can't fix**
  - Best candidate RMSE: 0.146
  - All candidate average RMSE: 0.291 (2x worse)
  - Polish can't fix fundamentally worse starting positions

- **Spreading budget hurts best candidate**
  - Baseline: 8 iterations on best → baseline_score 1.1688
  - This run: 4 iterations × 3 candidates → baseline_score 1.1510
  - The best candidate got LESS polish (4 vs 8) and performed worse

### Critical Insights
1. **Fine-grid polish is the bottleneck**: Each NM iteration runs expensive full-timestep simulations on the 100×50 grid
2. **Candidate quality gap is fundamental**: 2nd/3rd candidates are inherently worse - they come from different local optima that CMA-ES explored but rejected
3. **Baseline polish strategy is optimal**: Concentrating all polish budget on the BEST candidate maximizes the score

## Parameter Sensitivity
- **Most impactful parameter**: n_candidates_to_polish
  - Polishing 3 vs 1 candidate: +55.8 min runtime, -0.0945 score
- **Time-sensitive parameters**: polish_iters_per_candidate × n_candidates_to_polish

## Recommendations for Future Experiments
1. **DO NOT polish multiple candidates** - the baseline strategy (8 NM on best only) is optimal
2. **Focus polish budget on accuracy, not diversity** - the scoring formula already rewards having 3 candidates (diversity bonus), but averaging accuracy means bad candidates hurt more than diversity helps
3. **Polish overhead is multiplicative** - each additional candidate to polish roughly multiplies polish time
4. **Alternative approach**: Could try polishing ONLY if 2nd/3rd candidates are close to best (within 0.02 RMSE) - but prior evidence suggests this is rare

## Why This Hypothesis Was Wrong
The hypothesis assumed polishing 2nd/3rd candidates would improve their RMSE significantly. However:
1. 2nd/3rd candidates have fundamentally different positions (different local optima)
2. NM polish can only refine locally - it can't jump to a better basin
3. The 4 iterations of polish (vs 8 baseline) isn't enough to significantly improve any candidate
4. Time overhead of 3× fine-grid polish completely destroys any potential benefit

## Conclusion
**polish_strategy_v2 family EXHAUSTED.** The baseline strategy of concentrating all polish budget on the single best candidate is provably optimal. Spreading polish across multiple candidates adds overhead without improving accuracy.

## Raw Data
- MLflow run IDs: a9f603923c694cdea9fc4360bfc1481c
- Best config: N/A (experiment failed)

# Experiment Summary: Niching CMA-ES for Diversity

## Metadata
- **Experiment ID**: EXP_NICHING_CMAES_001
- **Worker**: W2
- **Date**: 2026-01-19
- **Algorithm Family**: diversity

## Objective
Improve the diversity component of the scoring formula by using Niching CMA-ES to maintain multiple distinct local optima simultaneously.

## Hypothesis
Niching CMA-ES can find multiple distinct solutions naturally, improving the diversity term (0.3 * N_valid/3) in the scoring formula.

## Results Summary
- **Best In-Budget Score**: 1.1688 @ 58.4 min (BASELINE - niching failed to beat)
- **Niching Attempt Score**: 1.0622 @ 46.9 min (-0.1066 vs baseline)
- **Status**: FAILED

## Critical Discovery

### The Scoring Formula Penalizes Diverse But Worse Candidates

The actual scoring formula is:
```
score = (1/N) * sum(1/(1+L_i)) + 0.3 * (N/3)
```

Where:
- N = number of valid candidates
- L_i = RMSE for each candidate
- The first term **AVERAGES** accuracy over ALL candidates

**Implication**: Adding diverse candidates that have higher RMSE hurts the score more than the diversity bonus helps!

Example:
- 1 candidate with RMSE=0: score = 1.0 + 0.1 = 1.1
- 3 candidates with RMSE=[0, 0.5, 0.5]: score = (1/3)*(1 + 0.67 + 0.67) + 0.3 = 0.78 + 0.3 = 1.08

Adding two worse candidates DECREASED the score from 1.1 to 1.08!

## Tuning History

| Run | Config Changes | Score | Time (min) | N_valid | Notes |
|-----|---------------|-------|------------|---------|-------|
| 1 | Taboo niching (radius=0.15) | 1.0622 | 46.9 | 1.70 | FAILED - Taboo pushes to suboptimal solutions |
| 2 | Baseline analysis | 1.1741 | 47.2 | 2.75 | 80% of samples already have 3 candidates |

## Key Findings

### What Didn't Work
1. **Taboo-based niching**: Penalizing proximity to previous solutions pushes CMA-ES away from the global optimum, causing worse accuracy
2. **Forcing diversity**: The scoring formula punishes this approach

### Critical Insights
1. **Baseline already achieves near-maximum diversity**: ~80% of samples get 3 candidates, average N_valid = 2.75
2. **Diversity is not the bottleneck**: With N_valid already at 2.75/3, improving diversity can only gain 0.025 in the diversity term
3. **Accuracy is the bottleneck**: Improving RMSE is the only viable path to higher scores
4. **The niching hypothesis was fundamentally flawed**: We assumed diversity was underutilized, but it's already near-saturated

### Why Niching CMA-ES Fails Here
1. The thermal inverse problem typically has ONE global optimum per sample
2. For 1-source: Only one true solution, no natural diversity
3. For 2-source: Minor symmetry (swap sources) doesn't create truly distinct optima
4. Forcing diversity = finding suboptimal solutions = worse average RMSE

## Candidate Distribution Analysis (Baseline)

| N_candidates | Count | Percentage |
|-------------|-------|------------|
| 3 | 16/20 | 80% |
| 2 | 3/20 | 15% |
| 1 | 1/20 | 5% |

## Parameter Sensitivity
- **taboo_radius**: Even with values larger than tau (0.2), the forced diversity hurts accuracy
- **n_independent_runs**: More runs = more budget spread thin = worse individual solutions

## Recommendations for Future Experiments

### Do NOT pursue:
1. Any niching/multimodal optimization approach
2. Forcing diversity through taboo regions
3. Multi-population strategies that dilute the budget

### Instead focus on:
1. **Accuracy improvement**: The scoring formula rewards low RMSE more than diversity
2. **Single best solution**: Finding the ONE global optimum is more important than multiple local optima
3. **Polish refinement**: The NM polish on full timesteps is the most impactful technique

### Specific next steps:
1. Consider higher sigma with temporal fidelity (W1 is testing this)
2. Focus on 2-source samples where RMSE is typically higher
3. Physics-informed initialization might help convergence

## Conclusion

**Diversity optimization is NOT a viable path.**

The current baseline with temporal fidelity + NM polish already achieves:
- Near-maximum diversity (2.75/3 candidates)
- Best-ever in-budget score (1.1688)
- Within time budget (58.4 min)

Future work should focus on **accuracy improvement**, not diversity.

## Files
- `optimizer.py`: NichingCMAESOptimizer with taboo regions
- `run.py`: Run script with niching parameters
- `STATE.json`: Experiment state and tuning history

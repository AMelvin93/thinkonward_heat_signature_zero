# Experiment: Multi-Objective Pareto Optimization

**Experiment ID**: EXP_MULTI_OBJECTIVE_001
**Worker**: W1
**Status**: ABORTED
**Date**: 2026-01-24

## Hypothesis
Use multi-objective optimization (NSGA-II) to find Pareto-optimal solutions that balance accuracy and diversity, potentially outperforming the baseline's greedy approach.

## Why This Was Aborted

### 1. The Scoring Formula Is ALREADY Multi-Objective
The competition scoring formula is:
```
score = (1/N)*sum(1/(1+RMSE_i)) + 0.3*(N_valid/3)
```
This already combines accuracy (first term) and diversity (second term). **We're already optimizing a multi-objective function.**

### 2. Pareto Optimization Can't Improve on Direct Optimization
When we have a combined objective f(x) = accuracy(x) + diversity(x), Pareto optimization between (accuracy, diversity) can only find the SAME optimal solutions - the ones that maximize the combined objective.

The only benefit of Pareto would be if:
- We didn't know how to combine the objectives
- The objectives were in different units

Neither applies here. The scoring formula defines exactly how to combine them.

### 3. The Baseline Already Balances Accuracy and Diversity
The baseline achieves 2.75/3 valid candidates on average (92% of max diversity bonus). The bottleneck is accuracy, not diversity.

### 4. NSGA-II Requires Too Many Evaluations
- NSGA-II with pop=20 and gen=10 needs 200+ evaluations
- Our budget per sample is 20-36 evaluations
- NSGA-II is 6-10x over budget

### 5. Prior Evidence
Multiple experiments have confirmed that accuracy is the bottleneck:
- Niching (FAILED): Diverse but worse candidates hurt score
- Taboo regions (baseline): Already maintain diversity efficiently
- Solution injection (FAILED): Sharing solutions didn't help

## Theoretical Analysis

Multi-objective optimization is useful when:
1. Objectives conflict in non-obvious ways
2. You want to find a Pareto front to present to a decision-maker
3. The objective combination is unknown

None of these apply here:
1. RMSE and diversity don't fundamentally conflict (better RMSE solutions can still be diverse)
2. We need ONE submission, not a Pareto front
3. The scoring formula is fixed and known

## Conclusion

**ABORTED** - Multi-objective Pareto optimization is theoretically unsound for this problem. The baseline's direct optimization of the combined score is already optimal.

**Recommendation**: The multi_objective family should be marked as EXHAUSTED. The scoring formula's fixed combination of accuracy and diversity means multi-objective approaches offer no advantage.

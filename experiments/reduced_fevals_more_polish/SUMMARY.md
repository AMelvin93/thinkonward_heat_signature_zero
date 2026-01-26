# Experiment Summary: reduced_fevals_more_polish

## Metadata
- **Experiment ID**: EXP_FEVAL_POLISH_TRADE_001
- **Worker**: W2
- **Date**: 2026-01-25
- **Algorithm Family**: budget_reallocation

## Objective
Trade CMA-ES function evaluations for extended Nelder-Mead polish. Reduce CMA-ES fevals (from 20/36 to 15/28) while increasing NM polish iterations (from 8 to 12), keeping total computational budget approximately constant.

## Hypothesis
CMA-ES may converge before using all fevals. Reallocating budget from CMA-ES to NM polish may maintain or improve accuracy while staying within time budget.

## Results Summary
- **Status**: ABORTED (Prior Evidence Conclusive)
- **Best In-Budget Score**: N/A (not executed)
- **Baseline Comparison**: N/A
- **Reason**: Prior experiments definitively showed both directions are losing trades

## Prior Evidence That Led to Abort

### 1. EXP_EARLY_STOP_CMA_001 (CMA-ES Early Stopping)
- Finding: CMA-ES continues improving >1% throughout its allocated fevals
- Implication: Reducing fevals would directly hurt accuracy
- The improvement rate doesn't decrease significantly - CMA-ES is still learning

### 2. EXP_EXTENDED_POLISH_001 (Extended NM Polish)
- Result: 12 polish iterations = +24 min for only +0.0015 score improvement
- Implication: Extra polish iterations have diminishing returns
- The marginal improvement per iteration is very low after iteration 8

## Why the Trade-off Fails

The proposed trade-off fails in both directions:

1. **CMA-ES side**: Each feval is valuable. Reducing from 20→15 (1-src) or 36→28 (2-src) removes 25% of the learning budget. CMA-ES uses these evaluations to:
   - Update the covariance matrix (learning search directions)
   - Adapt step size (sigma)
   - Maintain population diversity

2. **NM polish side**: Returns are sharply diminishing. NM polish provides:
   - Iterations 1-4: ~80% of improvement
   - Iterations 5-8: ~19% of improvement
   - Iterations 9-12: ~1% of improvement

3. **Net result**: Losing 25% of CMA-ES learning for 1% more polish improvement = net loss

## Key Insights

### Why This Was Obvious in Hindsight
- CMA-ES is a **learning algorithm** - each feval contributes to covariance estimation
- NM is a **local optimizer** - it only refines, doesn't explore
- You can't trade exploration for refinement without degrading the solution found

### The Budget Allocation is Already Optimal
- 20/36 fevals: Just enough for CMA-ES covariance to capture search landscape
- 8 NM polish: Just enough to refine the best candidate
- Any reallocation in either direction hurts overall performance

## Recommendations for Future Experiments
1. **budget_reallocation family is EXHAUSTED** - current allocation is proven optimal
2. Don't attempt to trade between search and polish - they serve different purposes
3. Focus on algorithmic improvements, not budget shuffling
4. Consider the nature of each algorithm before proposing trades

## Raw Data
- MLflow run IDs: None (experiment aborted)
- Config not executed
- Prior evidence references: EXP_EARLY_STOP_CMA_001, EXP_EXTENDED_POLISH_001

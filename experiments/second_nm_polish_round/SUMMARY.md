# Experiment: second_nm_polish_round

## Objective
Test if a second focused NM polish round on the final best candidate can improve accuracy while staying within 60 min budget.

## Hypothesis
nm4_perturb1 achieves 1.1585 @ 54.9 min with 5 min budget remaining. A second NM polish round on the winner only should add minimal overhead (1-2 min) while improving accuracy.

## Baselines
| Config | Score | Time (min) | In Budget |
|--------|-------|------------|-----------|
| nm4_perturb1 | 1.1585 | 54.9 | YES |
| nm_8iter + perturbation | 1.1622 | 64.1 | NO |

## Results

| Config | Score | Time (min) | In Budget | RMSE 1-src | RMSE 2-src | Overhead |
|--------|-------|------------|-----------|------------|------------|----------|
| second_polish_2iter | 1.1594 | 64.5 | NO | 0.1244 | 0.2029 | +9.5 min |
| second_polish_4iter | 1.1552 | 63.7 | NO | 0.1264 | 0.2123 | +8.7 min |
| second_polish_6iter | **1.1642** | 64.3 | NO | 0.1215 | 0.1928 | +9.3 min |

## Analysis

### Why All Configs Exceeded Budget
The second polish round adds **~9 minutes overhead**, NOT the expected 1-2 minutes. This is because:

1. **Fine grid verification**: After NM polish, the optimizer verifies the result on the fine grid with full timesteps
2. **Additional simulations**: Each NM step requires new PDE simulations
3. **Sequential overhead**: The second polish adds to the existing processing chain

### Score vs Time Tradeoff
| Config | Score Delta vs baseline | Time Delta | Efficiency |
|--------|------------------------|------------|------------|
| second_polish_6iter | +0.0057 | +9.3 min | 0.00061/min |
| second_polish_2iter | +0.0009 | +9.5 min | 0.00009/min |

The best result (second_polish_6iter @ 1.1642) represents a modest improvement but at too high a time cost.

### Variance Observation
Interestingly, second_polish_4iter (1.1552) scored LOWER than second_polish_2iter (1.1594), confirming high run-to-run variance observed in previous experiments.

## Key Findings

1. **Second polish is too expensive** - Adds ~9 min overhead, exceeding budget
2. **Overhead is consistent** - All configs took 63-65 min regardless of iteration count
3. **Fine grid verification dominates** - The extra NM iterations add minimal time; overhead comes from verification
4. **Best score is with 6 iterations** - 1.1642, but 4 min over budget

## Why This Failed

The implementation runs NM polish on the **coarse grid with reduced timesteps**, then verifies on the **fine grid with full timesteps**. This verification step is expensive and negates the time savings from reduced NM iterations.

To make second polish work within budget, we would need:
- Skip fine grid verification (but this may hurt accuracy)
- Use even coarser grid or fewer timesteps
- Accept lower accuracy on the second polish

## Recommendations

1. **Keep nm4_perturb1 as optimal in-budget config** - 1.1585 @ 54.9 min
2. **Mark polish_v4 family as EXHAUSTED** - Second polish doesn't fit budget
3. **Focus on CMA-ES exploration improvements** - Better initial convergence is more time-efficient

## Conclusion
**FAILED** - Second NM polish round adds too much overhead to fit in 60 min budget. The 5 min buffer in nm4_perturb1 cannot accommodate this enhancement.

## Family Status
`polish_v4` - **EXHAUSTED** - Second polish round adds 9 min overhead, exceeding budget.

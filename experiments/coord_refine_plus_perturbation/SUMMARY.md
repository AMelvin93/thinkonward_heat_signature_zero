# Experiment: coord_refine_plus_perturbation

## Objective
Test if combining coordinate refinement AND perturbation achieves higher accuracy than either technique alone.

## Hypothesis
Coordinate refinement and perturbation are orthogonal improvements:
- Coord refinement: Fine-tunes within current basin
- Perturbation: Explores nearby basins

If both techniques help different samples, combining them could improve overall score.

## Baselines
| Technique | Score | Time (min) | In Budget |
|-----------|-------|------------|-----------|
| nm6_coord_refine | 1.1602 | 57.55 | YES |
| hopping_no_tabu | 1.1689 | 58.18 | YES |

## Results

| Config | Score | Time (min) | In Budget | RMSE 1-src | RMSE 2-src |
|--------|-------|------------|-----------|------------|------------|
| nm5_coord_perturb2 | 1.1561 | 74.2 | NO | 0.1282 | 0.2080 |
| nm6_coord_perturb1 | 1.1658 | 69.2 | NO | 0.1197 | 0.1902 |
| nm6_coord_perturb2 | 1.1657 | 76.3 | NO | 0.1251 | 0.1851 |

## Analysis

### Time Analysis
- Best combined (nm6_coord_perturb1): 69.2 min - **9 min over budget**
- The combination adds 10-16 min overhead compared to individual techniques
- Cannot fit within 60 min budget even with reduced parameters

### Score Analysis
- Best combined score: 1.1658 (nm6_coord_perturb1)
- This is **WORSE** than hopping_no_tabu alone (1.1689)
- Only marginally better than nm6_coord_refine alone (1.1602)

### Key Finding
**The techniques do NOT stack beneficially.** Combining them adds overhead without improving accuracy. This suggests:
1. Both techniques target similar improvement opportunities
2. Perturbation after coord refinement doesn't find new basins
3. Coord refinement after perturbation doesn't add value

## Tuning Efficiency Metrics
- **Runs executed**: 3
- **Time utilization**: 0% (none within budget)
- **Parameter space explored**: NM iterations (5-6), perturbations (1-2)
- **Pivot points**: None - all runs over budget

## Budget Analysis
| Run | Score | Time | Budget Remaining | Decision |
|-----|-------|------|------------------|----------|
| 1   | 1.1561 | 74.2 | -14.2 min | CONTINUE |
| 2   | 1.1658 | 69.2 | -9.2 min | CONTINUE |
| 3   | 1.1657 | 76.3 | -16.3 min | CONCLUDE |

## What Would Have Been Tried With More Time
- If budget were 70 min: nm6_coord_perturb1 would be viable (1.1658)
- If budget were 80 min: Could test higher NM (7-8) + both techniques

## Conclusion
**FAILED** - Combining coord refinement + perturbation does not improve over either technique alone and pushes timing over budget.

## Recommendations
1. **Do NOT combine coord refinement + perturbation**
2. Use hopping_no_tabu (1.1689 @ 58 min) for best accuracy
3. Use nm6_coord_refine (1.1602 @ 57.6 min) as alternative
4. Mark `combined_approach` family as **EXHAUSTED**

## Family Status
`combined_approach` - **EXHAUSTED**

The techniques are not orthogonal; they target similar improvement opportunities. Focus on hopping_no_tabu as the production optimizer.

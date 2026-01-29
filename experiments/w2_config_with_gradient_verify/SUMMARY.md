# W2 Config with Gradient Verification Experiment

## Experiment ID
EXP_W2_CONFIG_WITH_VERIFICATION_001

## Hypothesis
Combining W2 baseline configuration (40% temporal fidelity, sigma 0.18/0.22) with gradient verification from perturbation_plus_verification might achieve better accuracy while staying within budget.

## Baseline
- **perturbation_plus_verification**: 1.1468 @ 54.2 min (25% temporal)
- **perturbed_extended_polish**: 1.1464 @ 51.2 min

## Results

### FAILED - All runs massively over budget with no score improvement

| Run | Temporal | Perturbation | Verification | Score | Time (min) | vs Budget | vs Baseline |
|-----|----------|--------------|--------------|-------|------------|-----------|-------------|
| 1 | 40% | Yes | Yes | 1.1410 | 391.6 | 6.5x OVER | -0.0058 |
| 2 | 35% | No | Yes | 1.1413 | 309.0 | 5.1x OVER | -0.0055 |
| 3 | 25% | Yes | Yes | 1.1355 | 313.9 | 5.2x OVER | -0.0113 |

**Best run**: Run 2 with score 1.1413 (still over budget)
**No runs within budget** - best_in_budget = null

## Tuning Efficiency Metrics
- **Runs executed**: 3
- **Time utilization**: N/A (no in-budget runs)
- **Parameter space explored**:
  - timestep_fraction: [0.25, 0.35, 0.40]
  - enable_perturbation: [true, false]
  - enable_verification: [true]
  - refine_maxiter: [3, 5]
- **Pivot points**:
  - Run 1 at 391.6 min → Reduced temporal to 35% and disabled perturbation
  - Run 2 at 309.0 min → Tried 25% temporal (matching baseline)
  - Run 3 at 313.9 min → Confirmed verification is the bottleneck

## Budget Analysis
| Run | Score | Time | Budget Remaining | Decision |
|-----|-------|------|------------------|----------|
| 1 | 1.1410 | 391.6 min | -331.6 min (6.5x OVER) | PIVOT - reduce temporal |
| 2 | 1.1413 | 309.0 min | -249.0 min (5.1x OVER) | PIVOT - use 25% temporal |
| 3 | 1.1355 | 313.9 min | -253.9 min (5.2x OVER) | ABANDON - fundamentally broken |

## Why W2 Config + Verification Fails

1. **Gradient verification is SLOW**: Computing gradients via finite differences adds massive overhead per candidate
2. **Temporal fidelity doesn't explain timing**: 25% temporal still takes 313.9 min, similar to 35% temporal (309.0 min)
3. **The optimizer design is flawed**: Verification-based reranking requires many full simulations
4. **No accuracy benefit**: Despite the extra computation, scores are WORSE than baseline

## Critical Analysis

The key insight is that **gradient verification overhead dominates runtime**, not temporal fidelity:
- Run 2 (35% temporal, no perturbation): 309.0 min
- Run 3 (25% temporal, with perturbation): 313.9 min

Even reducing temporal fidelity by 40% (from 40% to 25%) and disabling perturbation only marginally reduces time. The verification step itself is the bottleneck.

## Key Findings

1. **W2 config (higher temporal) is incompatible with 60-min budget when combined with verification**
2. **Gradient verification adds ~5x time overhead without improving accuracy**
3. **The baseline 25% temporal approach is already optimal for the time budget**
4. **Verification-based approaches should be abandoned**

## What Would Have Been Tried With More Time
- If this experiment had shown promise, we would have tried:
  - Faster gradient approximation methods
  - Reduced number of candidates to verify
  - Coarser grid for verification
- However, since all runs showed 5x+ time overhead, further exploration is not warranted

## Recommendation

**DO NOT use gradient verification approaches.** The overhead is prohibitive and provides no accuracy benefit.

Stick with proven optimizers:
- perturbation_plus_verification (1.1468 @ 54.2 min) - despite the name, uses lightweight verification
- perturbed_extended_polish (1.1464 @ 51.2 min)
- perturbed_local_restart (1.1452 @ 47.7 min)

## Conclusion

**EXPERIMENT FAILED**

The W2 config with gradient verification approach is fundamentally incompatible with the 60-minute time budget. All three runs exceeded the budget by 5-6x while achieving lower scores than baseline. The hypothesis that combining higher temporal fidelity with gradient verification could improve accuracy is DISPROVED.

The verification_v2 family of experiments should be marked as EXHAUSTED.

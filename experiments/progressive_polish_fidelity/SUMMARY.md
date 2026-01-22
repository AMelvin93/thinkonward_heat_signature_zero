# Experiment Summary: progressive_polish_fidelity

## Metadata
- **Experiment ID**: EXP_PROGRESSIVE_POLISH_FIDELITY_001
- **Worker**: W2
- **Date**: 2026-01-19
- **Algorithm Family**: temporal_fidelity_extended

## Objective
Test if using progressive timestep fidelity during NM polish (60% early, 100% final) could save time while maintaining accuracy.

## Result
**ABORTED - Prior experiment showed truncated polish HURTS accuracy significantly.**

## Prior Evidence (from early_timestep_filtering)

From the early_timestep_filtering experiment SUMMARY.md:

| Configuration | Score | Time |
|--------------|-------|------|
| 40% timesteps + 8 NM polish (truncated) | 1.1342 | 45.8 min |
| 40% timesteps + 8 NM polish (full) | 1.1688 | 58.4 min |

**Key quote from SUMMARY.md:**
> "NM polish on **truncated** timesteps HURTS (Run 8: 1.1342)
> NM polish on **full** timesteps HELPS significantly (Run 10: 1.1688)
> The truncated signal is a proxy; polishing the proxy overfits to noise"

## Analysis

The prior experiment showed that using truncated (40%) timesteps for NM polish:
- Decreased score by 0.0346 (2.9% worse)
- Saved 12.6 min

This is a poor tradeoff - the accuracy loss is significant.

Progressive polish fidelity (60% early, 100% late) would:
- Use 60% timesteps for iterations 1-4 (still a proxy signal)
- Use 100% timesteps for iterations 5-8 (accurate signal)

The early iterations with 60% would still "polish the proxy" - the same fundamental problem that caused truncated polish to fail.

## Why Abort Was Correct

1. **Prior experiment already answered the question**: Truncated polish hurts accuracy significantly
2. **Fundamental issue**: NM polish needs accurate fitness landscape, not a proxy
3. **60% is still a proxy**: Even 60% timesteps won't fully represent the thermal diffusion
4. **Implementation complexity**: Would require substantial optimizer modifications
5. **Risk/reward unfavorable**: Likely to lose accuracy for modest time savings

## Recommendation

Full timestep polish (100%) is optimal for accuracy. Do not use reduced timesteps during polish phase.

The 12.6 min spent on full polish is well worth the 0.0346 score improvement.

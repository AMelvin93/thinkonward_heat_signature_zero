# Population Warmup Experiment

## Status: ABORTED

## Reason
This experiment is fundamentally the same as `sigma_ladder` which already FAILED.

### Prior Evidence (sigma_ladder)
> "Sigma ladder (0.30/0.35 -> 0.15/0.20) adds overhead and interferes with CMA-ES covariance learning. sigma_scheduling EXHAUSTED."

### Why It Doesn't Work
1. **CMA-ES adapts sigma internally** - Manually changing sigma mid-run conflicts with the algorithm's covariance adaptation mechanism
2. **Covariance matrix depends on consistent sigma** - Switching sigma invalidates the learned covariance structure
3. **Exploration/exploitation trade-off is already handled** - CMA-ES naturally starts broad (high sigma) and tightens (low sigma) as it converges

### Similar Failed Experiments
- sigma_ladder: FAILED - "sigma_scheduling EXHAUSTED"
- adaptive_sigma_schedule: FAILED - Same issue
- learning_rate_adapted_cmaes: FAILED - Parameter adaptation interference

## Conclusion
ABORTED - Prior evidence conclusively shows sigma scheduling interferes with CMA-ES. The `cmaes_efficiency` family is theoretically exhausted.

## Recommendation
Do not pursue sigma schedule modifications. CMA-ES's internal sigma adaptation is optimal. Focus on other approaches like perturbation which has shown promise.

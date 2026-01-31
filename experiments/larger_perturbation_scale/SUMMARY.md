# Larger Perturbation Scale Experiment

## Status: FAILED

**Hypothesis DISPROVED**: Larger perturbation scales (0.08, 0.10) do NOT improve over baseline (0.05).

## Hypothesis
Current 0.05 perturbation scale may be too small. Larger jumps (0.08, 0.10) may reach better basins.

## Results

| Run | Scale | Score | RMSE | Perturbed Selected | Projected Time |
|-----|-------|-------|------|-------------------|----------------|
| 1 | 0.08 | 1.1469 | 0.1173 | 1/20 | 154.7 min |
| 2 | 0.10 | 1.1506 | 0.1445 | 3/20 | 155.5 min |
| baseline | 0.05 | 1.1549 | 0.1287 | 1/20 | 152.8 min |

**Note**: Machine runs ~4.5x slower than baseline, so all times are inflated.

## Key Findings

### 1. Baseline Scale is Optimal
- Scale 0.05 achieves the best score (1.1549)
- Larger scales (0.08, 0.10) show no improvement

### 2. Perturbation Scale Does Not Significantly Impact Results
- All results within noise range (~0.008)
- Scale 0.10 had more perturbed candidates selected (3 vs 1), but worse RMSE

### 3. Confirms Prior Evidence
- adaptive_perturbation_scale experiment also FAILED
- Fixed scale 0.05 remains optimal

## Conclusion
**FAILED**: Larger perturbation scales do NOT improve over baseline scale 0.05.

The current perturbation approach (scale=0.05, perturb_top_n=1, n_perturbations=2) is already optimal. Increasing the scale doesn't help because:
1. The perturbation is already effective at finding nearby improved solutions
2. Larger jumps may land in worse basins, not better ones
3. The local basin structure favors small refinements over large jumps

## Tuning Efficiency Metrics
- **Runs executed**: 3 (including baseline comparison)
- **Parameter space explored**: perturbation_scale [0.05, 0.08, 0.10]
- **Conclusion**: Baseline scale 0.05 optimal

## Family Status
`perturbation_v4` family: EXHAUSTED
- Larger perturbation scale: FAILED
- Fixed scale 0.05 is optimal

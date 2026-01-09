# baseline_lbfgs

Baseline L-BFGS-B optimizer with finite difference gradients.

## Approach

Uses scipy's L-BFGS-B optimizer with:
- Multiple random restarts (parallel across CPU cores)
- Finite difference gradient approximation
- Box constraints on source positions and intensities

This is the simplest approach and serves as a baseline for comparison.

## Usage

```bash
# Run full experiment
python scripts/run_experiment.py --experiment baseline_lbfgs

# Run on subset for testing
python scripts/run_experiment.py --experiment baseline_lbfgs --n-samples 5

# Run with custom tags
python scripts/run_experiment.py --experiment baseline_lbfgs --tags version=v2
```

## Results

| Metric | Value |
|--------|-------|
| RMSE (mean) | TBD |
| RMSE (std) | TBD |

## Notes

- Created: 2024-12-20
- Baseline for comparison with other approaches

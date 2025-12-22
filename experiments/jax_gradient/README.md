# jax_gradient

JAX-based optimizer with automatic differentiation.

## Approach

Uses JAX for:
- **GPU acceleration** - Run PDE solver on GPU for 10-50x speedup
- **Automatic differentiation** - True gradients instead of finite differences
- **JIT compilation** - Compiled solver for maximum speed

Key advantages over baseline:
- Much faster per iteration (GPU)
- More accurate gradients (autodiff vs finite diff)
- Better convergence in fewer iterations

## Usage

```bash
# Run full experiment
python scripts/run_experiment.py --experiment jax_gradient

# Run on subset for testing
python scripts/run_experiment.py --experiment jax_gradient --n-samples 5

# Tune learning rate
# Edit experiments/jax_gradient/config.yaml to change jax.learning_rate
```

## Results

| Metric | Value |
|--------|-------|
| RMSE (mean) | TBD |
| RMSE (std) | TBD |
| GPU Available | TBD |

## Notes

- Created: 2024-12-20
- Requires JAX installation (already in pyproject.toml)
- For GPU: install jax[cuda12] instead of jax

# CMA-ES Optimizer Experiment

## Overview

CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer for heat source identification. Particularly effective for 2-source problems where L-BFGS-B gets stuck in local minima.

## Why CMA-ES?

2-source problems have **permutation symmetry**: sources (A, B) = sources (B, A). This creates multiple equivalent optima that confuse gradient-based optimizers like L-BFGS-B.

CMA-ES is:
- Gradient-free (no finite difference overhead)
- Learns covariance structure of the landscape
- Better at escaping local minima
- Handles multi-modal problems naturally

## Preliminary Results

| Sample Type | HybridOptimizer RMSE | CMA-ES+Polish RMSE | Improvement |
|-------------|---------------------|--------------------| ------------|
| 1-source | ~0.44 | ~0.26 | 1.7x better |
| 2-source | ~1.16 | ~0.27 | **4.3x better** |

## Usage

```bash
# Prototype mode (all CPUs, no MLflow)
uv run python experiments/cmaes/run.py --workers -1

# G4dn simulation mode (7 workers, MLflow logging)
cd /mnt/c/Users/amelv/Repo/thinkonward_heat_signature_zero  # WSL
uv run python experiments/cmaes/run.py --workers 7
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-fevals-1src` | 15 | Max CMA-ES evaluations for 1-source |
| `--max-fevals-2src` | 25 | Max CMA-ES evaluations for 2-source |
| `--sigma0-1src` | 0.10 | Initial step size for 1-source |
| `--sigma0-2src` | 0.20 | Initial step size for 2-source |
| `--polish` | True | Add L-BFGS-B polish step |
| `--polish-iter` | 5 | Max iterations for polish |

## Time Budget

- Estimated: ~64 min for 400 samples (slightly over 60 min budget)
- Trade-off: Much better quality (~4x RMSE improvement) at slight time cost

## Status

**TESTING** - Awaiting full WSL validation run before promotion to `src/`.

## Promotion Criteria

- [ ] Full 80-sample WSL run
- [ ] MLflow logging with platform=wsl
- [ ] Projected time <60 min for 400 samples
- [ ] Score improvement over HybridOptimizer

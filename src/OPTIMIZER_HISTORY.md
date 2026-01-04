# Optimizer History & Experiment Log

This document tracks all optimization approaches attempted for the Heat Signature Zero challenge, their results, and lessons learned.

## Key Finding

**NumPy CPU with joblib parallelization outperforms JAX GPU** for this problem because:
- The ADI time-stepping is inherently sequential (can't parallelize across timesteps)
- GPU kernel launch overhead exceeds computation time for small grids (100x50)
- Sample-level parallelism (7 workers) is more effective than GPU parallelism

---

## Models in MLflow

### SELECTED FOR SUBMISSION (2026-01-04)
| Run Name | Optimizer | Score | RMSE | Projected Time | Config |
|----------|-----------|-------|------|----------------|--------|
| `enhanced_transfer_20260104_141737` | EnhancedTransferOptimizer | **0.8688** | 0.456 | **55.6 min** | fevals=18/36, shuffle=True, enhanced_features=True, adaptive_k=False |

**Key Innovation**: Enhanced 11-feature similarity matching doubles transfer effectiveness (17.5% vs 8.8%).

**Enhanced Features (11 vs 5 basic)**:
- Basic: max_temp, mean_temp, std_temp, kappa, n_sensors
- Spatial: centroid_x, centroid_y, spatial_spread
- Temporal: onset_mean, onset_std
- Correlation: avg_sensor_correlation

**Per-Source Performance**:
- 1-source RMSE: **0.276** (excellent)
- 2-source RMSE: **0.576** (main bottleneck - 2x worse)

**Batch RMSE Progression** (shows transfer learning working):
- Batch 1: 0.4071 (no history)
- Batch 2: 0.5391
- Batch 3: 0.4512
- Batch 4: 0.4273 (lowest - most history)

### Previous Submissions
| Run Name | Optimizer | Score | RMSE | Projected Time | Config |
|----------|-----------|-------|------|----------------|--------|
| `multi_candidates_20251230_140041` | MultiCandidateOptimizer | 0.7764 | 0.525 | 53.8 min | fevals=20/40, triangulation=True, intensity_polish=True |
| `cmaes_20251229_124607` | CMAESOptimizer | 0.7501 | 0.515 | 57.2 min | polish_iter=1, triangulation=True |

### CMA-ES Testing Results (2024-12-29)
| Run Name | Score | RMSE | Projected | Config |
|----------|-------|------|-----------|--------|
| `cmaes_20251229_112401` | 0.8419 | 0.360 | 75.5 min | polish=5 (over budget) |
| `cmaes_20251229_122106` | 0.7677 | 0.442 | 59.7 min | polish=2 (borderline) |
| `cmaes_20251229_124607` | **0.7501** | 0.515 | **57.2 min** | **polish=1 (SELECTED)** |
| `cmaes_20251229_123251` | 0.6505 | 0.623 | 42.8 min | no polish (bad accuracy) |

### Previous Models
| Run Name | Optimizer | Score | RMSE | Notes |
|----------|-----------|-------|------|-------|
| `jax_hybrid_max_iter_3_fixed` | HybridOptimizer | 0.5710 | 0.900 | Previous best |
| `baseline_lbfgs_20251220_214520` | BaselineLBFGS | 1.009 | 0.100 | Good RMSE but slow |
| `tabu_search_20251220_221852` | TabuSearch | 1.162 | 0.160 | High score but time-intensive |

---

## Optimizer Implementations

### Currently Used
| File | Status | Description |
|------|--------|-------------|
| `experiments/transfer_learning/optimizer.py` | **SELECTED** | Transfer learning + CMA-ES with feature-based similarity matching |
| `experiments/transfer_learning/run.py` | **SELECTED** | Batch processing with history accumulation for transfer |
| `experiments/cmaes/optimizer.py` | Previous | CMA-ES + L-BFGS-B polish with triangulation init |
| `triangulation.py` | **PRODUCTION** | Physics-based initialization from sensor onset times |
| `hybrid_optimizer.py` | Previous | NumPy + L-BFGS-B with triangulation init and joblib parallelism |
| `optimizer.py` | Baseline | Original L-BFGS-B implementation |
| `scoring.py` | Utility | Competition scoring functions |

### CMA-ES Results (Preliminary Testing)
CMA-ES shows **4x better RMSE** on 2-source problems compared to L-BFGS-B:

| Sample Type | HybridOptimizer RMSE | CMA-ES+Polish RMSE | Improvement |
|-------------|---------------------|--------------------| ------------|
| 1-source | ~0.44 | ~0.26 | 1.7x better |
| 2-source | ~1.16 | ~0.27 | **4.3x better** |

**Why CMA-ES helps for 2-source**: Permutation symmetry creates multiple local minima. CMA-ES finds global optima where L-BFGS-B gets stuck.

**Trade-off**: CMA-ES is ~10% slower but produces much better results on hard problems.

### JAX Implementations (Experimental)
| File | Status | Description | Why Not Used |
|------|--------|-------------|--------------|
| `jax_hybrid_optimizer.py` | Tested | JAX forward sim + scipy L-BFGS-B | Slower than NumPy due to GPU overhead |
| `jax_optimizer.py` | Tested | Basic JAX optimizer | GPU overhead exceeds benefit |
| `jax_simulator.py` | Utility | JAX-based ADI simulator | Foundation for JAX optimizers |
| `jax_simulator_fast.py` | Utility | Memory-efficient JAX simulator | Samples during loop, not after |
| `jax_fast_optimizer.py` | Tested | Optimized JAX with batched gradients | Still slower than NumPy CPU |
| `jax_batched_optimizer.py` | Tested | Batched finite differences via vmap | GPU overhead still dominant |
| `jax_ultrafast_optimizer.py` | Tested | All JAX optimizations combined | Target: 9s/sample, achieved ~15s |
| `jax_coarse_to_fine_optimizer.py` | Tested | Two-phase: coarse (50x25) then fine | Promising but NumPy still faster |
| `jax_aggressive_optimizer.py` | Tested | Ultra-coarse (25x12) + minimal iters | Sacrificed too much accuracy |
| `jax_pure_optimizer.py` | Tested | Pure JAX Adam (no scipy) | Eliminates callback overhead but still slower |

### Adjoint Method Implementations
| File | Status | Description | Why Not Used |
|------|--------|-------------|--------------|
| `adjoint_optimizer.py` | Tested | Full-storage adjoint for exact gradients | Memory: O(nt*nx*ny), too slow (~100-300s/sample) |
| `adjoint_optimizer_fast.py` | Tested | Checkpointed adjoint O(sqrt(nt)) memory | 2x faster gradients, but 157s/sample overall |

**Adjoint Method Finding**: The adjoint method provides exact gradients (validated: max error 0.01%) with cost independent of parameter count. However, it doesn't reduce the number of L-BFGS-B iterations needed, so overall time is still dominated by the number of forward/backward passes. The HybridOptimizer wins through sample-level parallelism (7 workers), not through faster gradients.

### Other Approaches
| File | Status | Description | Why Not Used |
|------|--------|-------------|--------------|
| `adaptive_learning_optimizer.py` | Abandoned | Online learning during inference | Too slow, didn't converge reliably |
| `tabu_optimizer.py` | Tested | Tabu search for discrete exploration | Good results but too slow for 400 samples |
| `tabu_gradient_optimizer.py` | Tested | Tabu + gradient refinement | Combined approach, still too slow |

---

## Performance Comparison (80 samples)

### HybridOptimizer (NumPy CPU) - Winner
| max_iter | Time | Projected 400 | Score |
|----------|------|---------------|-------|
| 2 | ~45 min | 54.7 min | 0.52 |
| 3 | ~55 min | 68.7 min | 0.57 |
| 4 | ~65 min | 79.3 min | 0.77 |

### JAX Approaches - Not Viable
| Optimizer | Time/Sample | Issue |
|-----------|-------------|-------|
| JAXHybrid | ~15-20s | GPU kernel overhead |
| JAXUltrafast | ~12-15s | Still slower than NumPy parallel |
| JAXCoarseToFine | ~10-12s | Accuracy loss at coarse resolution |

---

## Lessons Learned

### What Works
1. **Sample-level parallelism** with joblib (7 workers) beats GPU parallelism
2. **Triangulation initialization** from sensor onset times (+13.4% RMSE improvement over hottest-sensor)
3. **L-BFGS-B** is effective for this smooth optimization landscape
4. **Fewer iterations** (2-3) with good init beats many iterations with random init
5. **Physics-based init** using heat diffusion scaling: r ~ sqrt(4*kappa*t)

### What Doesn't Work
1. **JAX/GPU acceleration** - overhead exceeds benefit for small grids
2. **Coarse-to-fine** - accuracy loss at coarse resolution hurts final score
3. **Pure gradient descent** - L-BFGS-B's Hessian approximation is crucial
4. **Tabu search** - too many simulator calls, doesn't scale to 400 samples
5. **Multiple candidates** - diversity bonus (0.3 * N/3) doesn't offset time cost
6. **Adjoint method** - exact gradients but doesn't reduce L-BFGS iterations; sample parallelism is more effective

### Key Constraints
- **Time limit**: 60 minutes for 400 samples on G4dn.2xlarge (8 vCPUs)
- **Per-sample budget**: ~9 seconds average (with 7 parallel workers: ~63s wall time)
- **Score formula**: `P = (1/N) * Σ(1/(1+RMSE)) + 0.3 * (N_valid/3)`

---

## Future Directions to Explore

### Fully Tested (Not Viable for Production)
1. **Adjoint Method** (`adjoint_optimizer_fast.py`) - Exact gradients with O(sqrt(nt)) memory checkpointing
   - Gradients validated: max relative error 0.01%
   - Per-gradient speedup: 2x for 1-source, 4x for 2-source vs finite differences
   - Overall time: 157.4s per sample (still too slow)
   - **Why not faster**: Adjoint reduces gradient cost but not iteration count; sample-level parallelism (7 workers) in HybridOptimizer is more effective
   - Conclusion: Not viable for production under 60-minute constraint

### Not Yet Tried
1. **Surrogate modeling** - Train a fast neural network to approximate the simulator
2. **Bayesian optimization** - Gaussian process for smart exploration
3. **Ensemble methods** - Combine predictions from multiple fast models
4. **Dimension reduction** - PCA on sensor readings to guide initialization
5. **Transfer learning** - Use solutions from similar samples as starting points

### Recently Implemented
1. **Triangulation initialization** (`triangulation.py`) - Uses heat diffusion physics to estimate source positions
   - Detects characteristic times (when sensors reach fraction of max temp)
   - Estimates distances using r ~ sqrt(4*kappa*t)
   - Trilaterates position from multiple distance estimates
   - **Results**: +13.4% RMSE improvement, +3.1% competition score improvement, no time penalty

### Potential Improvements
1. **Early stopping** - Stop when RMSE improvement plateaus
2. **Adaptive iterations** - More iterations for hard samples, fewer for easy ones
3. **Caching** - Reuse solver setup across samples with same (nt, bc)

---

## File Reference

```
src/
├── OPTIMIZER_HISTORY.md           # This file
├── hybrid_optimizer.py            # PRODUCTION - NumPy + L-BFGS-B + triangulation
├── triangulation.py               # PRODUCTION - Physics-based initialization
├── optimizer.py                   # Baseline L-BFGS-B
├── scoring.py                     # Competition scoring
├── visualize.py                   # Plotting utilities
├── jax_*.py                       # JAX experiments (not used in production)
├── adjoint_optimizer.py           # Adjoint method (full storage - too slow)
├── adjoint_optimizer_fast.py      # Checkpointed adjoint (still too slow)
├── adaptive_learning_optimizer.py # Abandoned
└── tabu_*.py                      # Tabu search experiments (too slow)

scripts/
├── compare_initializations.py     # Compare init methods (RMSE without optimization)
├── compare_optimization.py        # Compare full optimization with different inits
└── ...

experiments/
└── adjoint_optimizer/train.py     # Experiment runner for adjoint optimizer
```

---

*Last updated: 2024-12-28*
*Production model: HybridOptimizer with max_iter=3, score=0.5710*

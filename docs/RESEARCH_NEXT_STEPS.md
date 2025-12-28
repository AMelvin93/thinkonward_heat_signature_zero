# Research: Next Steps for Heat Signature Zero

*Research conducted: 2024-12-28*
*Current best score: 0.5710 (HybridOptimizer, max_iter=3)*

---

## Executive Summary

Based on literature review and analysis of our current approach, the most promising directions are:

| Priority | Approach | Potential Gain | Effort | Risk |
|----------|----------|---------------|--------|------|
| **1** | Adjoint Method for Gradients | High | Medium | Low |
| **2** | Better Initialization (Triangulation) | Medium | Low | Low |
| **3** | Multi-Fidelity Optimization | Medium | Medium | Medium |
| **4** | PINN Surrogate + Simulator Refinement | High | High | Medium |
| **5** | CMA-ES for Multi-Modal Search | Medium | Low | Low |

---

## 1. Adjoint Method for Efficient Gradients

### Why This Could Work
Currently using numerical finite differences for gradients (L-BFGS-B). The [adjoint method](https://aalexan3.math.ncsu.edu/articles/adjoint_based_gradient_and_hessian.pdf) can compute exact gradients with **cost independent of parameter count**.

### How It Works
Instead of perturbing each parameter, solve a single "adjoint problem" backwards in time:
```
Forward: solve heat equation for T(x,y,t)
Adjoint: solve adjoint equation backwards for λ(x,y,t)
Gradient: ∇J = ∫ λ · ∂S/∂θ dt
```

### Implementation Plan
1. Derive adjoint equations for ADI scheme
2. Implement backward-in-time solver
3. Compute exact gradients in O(1) forward + O(1) backward passes
4. Use with L-BFGS-B or gradient descent

### Expected Impact
- **Speed**: 2-3x faster gradient computation
- **Accuracy**: Exact gradients (vs numerical approximation)
- **Score**: Could enable more iterations → better RMSE

### References
- [Computing gradients and Hessians using the adjoint method (NC State)](https://aalexan3.math.ncsu.edu/articles/adjoint_based_gradient_and_hessian.pdf)
- [Notes on Adjoint Methods (MIT)](https://math.mit.edu/~stevenj/18.336/adjoint.pdf)
- [Heat Source Estimation with Conjugate Gradient (SciELO)](https://www.scielo.br/j/jbsms/a/pyB9sdrFkZw6JszHp3jYbKx/?lang=en)

---

## 2. Better Initialization via Triangulation

### Why This Could Work
Current "smart init" places sources at hottest sensors. But with multiple sensors, we can **triangulate** the actual source position more accurately.

### Approach: Time-of-Arrival / Heat Diffusion Inversion
Heat diffuses at known rate (κ). By analyzing when each sensor "sees" temperature rise:
1. Fit exponential rise to each sensor's time series
2. Estimate distance from source to each sensor
3. Triangulate position using multiple distance estimates

### Implementation Plan
```python
def triangulate_source(sensors_xy, Y_noisy, kappa, dt):
    # 1. Detect onset time for each sensor (when T > threshold)
    onset_times = detect_onset(Y_noisy, threshold=0.1)

    # 2. Estimate distance: d ∝ sqrt(κ * t_onset)
    distances = np.sqrt(4 * kappa * onset_times * dt)

    # 3. Triangulate using least squares
    # minimize: Σ (||p - sensor_i|| - distance_i)²
    x0 = trilaterate(sensors_xy, distances)
    return x0
```

### Expected Impact
- **Speed**: Nearly free (analysis only, no simulation)
- **Accuracy**: Better starting point → faster convergence
- **Score**: +0.05-0.10 improvement possible

### For 2-Source Problems
Use clustering on sensor responses to identify which sensors are dominated by which source, then triangulate separately.

---

## 3. Multi-Fidelity Optimization

### Why This Could Work
[Multi-fidelity optimization](https://royalsocietypublishing.org/doi/10.1098/rspa.2007.1900) uses cheap low-fidelity simulations to explore, then expensive high-fidelity to refine.

### Current Coarse-to-Fine Issue
Our JAX coarse-to-fine approach lost too much accuracy at coarse resolution. Better approach: **Co-Kriging**.

### Co-Kriging Approach
1. Build Gaussian Process surrogate from coarse simulations (fast)
2. Correct GP using sparse high-fidelity evaluations
3. Use GP to guide optimization, only call full simulator at promising points

### Implementation Plan
```python
from sklearn.gaussian_process import GaussianProcessRegressor

def multi_fidelity_optimize(sample, meta):
    # Phase 1: Coarse exploration (50x25 grid, 1/4 timesteps)
    coarse_samples = latin_hypercube_sample(n=50)
    coarse_losses = [coarse_simulate(params) for params in coarse_samples]

    # Phase 2: Build surrogate
    gp = GaussianProcessRegressor()
    gp.fit(coarse_samples, coarse_losses)

    # Phase 3: Acquisition-guided fine evaluation
    for _ in range(5):
        next_point = maximize_expected_improvement(gp, bounds)
        fine_loss = fine_simulate(next_point)
        gp.fit(...)  # Update with new point

    return best_point
```

### Expected Impact
- **Speed**: Fewer full-resolution simulations needed
- **Exploration**: Better global search
- **Score**: Depends on GP accuracy

### References
- [Multi-fidelity optimization via surrogate modelling (Royal Society)](https://royalsocietypublishing.org/doi/10.1098/rspa.2007.1900)
- [Multi-fidelity RBF surrogate (Springer)](https://link.springer.com/article/10.1007/s00158-020-02575-7)

---

## 4. Physics-Informed Neural Network (PINN) Surrogate

### Why This Could Work
[PINNs](https://www.sciencedirect.com/science/article/abs/pii/S0735193323000519) can learn to approximate the simulator while respecting physics constraints. Once trained, evaluation is ~1000x faster.

### Competition-Compliant Approach
**Key**: Must use simulator at inference. Solution: PINN for initial guess, simulator for refinement.

```
PINN(sensor_data) → initial_guess → L-BFGS-B with simulator → final_answer
```

### Implementation Plan
1. **Pre-train** PINN on synthetic data (many samples, varying sources)
2. **At inference**:
   - PINN predicts initial (x, y, q) from sensor readings
   - Run 2-3 L-BFGS-B iterations with actual simulator
   - Return refined solution

### Architecture
```python
class HeatSourcePINN(nn.Module):
    def __init__(self):
        # Input: flattened sensor readings + metadata
        # Output: (x, y, q) for each source
        self.encoder = MLP([n_sensors * n_timesteps, 256, 128, 64])
        self.decoder = MLP([64, 32, 3 * n_sources])
```

### Expected Impact
- **Speed**: PINN inference ~1ms, allows more simulator iterations
- **Accuracy**: Pre-trained on distribution, generalizes well
- **Score**: Potentially +0.1-0.2 if PINN is accurate

### Challenges
- Requires training data generation (can parallelize)
- May not generalize to edge cases
- Competition rules require simulator use (satisfied by refinement step)

### References
- [Enhanced surrogate modelling of heat conduction (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0735193323000519)
- [PINN-based virtual thermal sensor (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0017931023005380)
- [ThermoNet for heat source localization (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0924424718314110) - 99% accuracy!

---

## 5. CMA-ES for Multi-Modal Exploration

### Why This Could Work
[CMA-ES](https://cma-es.github.io/) is state-of-the-art for non-convex, multi-modal optimization. Our 2-source problems may have multiple local minima.

### Key Advantages
- **Gradient-free**: No finite differences needed
- **Covariance adaptation**: Learns parameter correlations
- **Multi-modal**: Can escape local minima
- **Parallel**: Population-based (good for joblib)

### Implementation Plan
```python
import cma

def cma_es_optimize(sample, meta, max_evals=100):
    def objective(params):
        return simulate_and_compute_rmse(params, sample, meta)

    x0 = smart_init(sample)
    sigma0 = 0.3  # Initial step size

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'maxfevals': max_evals,
        'popsize': 8,  # Parallel evaluations
        'bounds': [[x_min, y_min, q_min], [x_max, y_max, q_max]]
    })

    while not es.stop():
        solutions = es.ask()
        # Parallel evaluation with joblib
        fitness = Parallel(n_jobs=8)(delayed(objective)(s) for s in solutions)
        es.tell(solutions, fitness)

    return es.result.xbest
```

### Expected Impact
- **Speed**: Population parallelism fits our joblib approach
- **Exploration**: Better for 2-source problems with multiple minima
- **Score**: May find better global optima

### References
- [CMA-ES Official Site](https://cma-es.github.io/)
- [CMA-ES Tutorial (arXiv)](https://arxiv.org/abs/1604.00772)
- [pymoo CMA-ES Documentation](https://pymoo.org/algorithms/soo/cmaes.html)

---

## 6. Differentiable Simulation with JAX (Revisited)

### Why Reconsider
Our JAX attempts failed due to GPU overhead, but [JAX-FEM](https://www.sciencedirect.com/science/article/abs/pii/S0010465523001479) shows automatic differentiation can work for PDE optimization.

### New Approach: CPU-Only JAX with Adjoint
Instead of GPU, use JAX on CPU for:
1. **Automatic differentiation** through the solver
2. **JIT compilation** for speed
3. **vmap** for batching across workers

### Key Insight from JAX-Fluids
> "End-to-end optimization. ML models can be optimized with gradients that are backpropagated through the entire CFD algorithm."

### Implementation
```python
@jax.jit
def differentiable_loss(params, sample_data):
    Y_pred = jax_simulate(params, sample_data)
    return jnp.mean((Y_pred - Y_obs) ** 2)

grad_fn = jax.grad(differentiable_loss)

# Use JAX gradients with scipy L-BFGS-B
def scipy_objective(params_np):
    params = jnp.array(params_np)
    loss = differentiable_loss(params, data)
    grad = grad_fn(params, data)
    return float(loss), np.array(grad)
```

### Expected Impact
- **Speed**: Exact gradients faster than finite differences
- **Accuracy**: No gradient approximation error
- **Compatibility**: Works with scipy optimizers

### References
- [JAX-FEM (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0010465523001479)
- [JAX, M.D. Differentiable Physics (NeurIPS)](https://papers.nips.cc/paper/2020/file/83d3d4b6c9579515e1679aca8cbc8033-Paper.pdf)
- [Solving PDEs with JAX (Medium)](https://medium.com/@jassem.abbasi/solving-partial-differential-equations-pde-with-jax-a-differentiable-approach-by-minimizing-d0dc5c366e5f)

---

## 7. Hybrid Approaches (Combining Methods)

### Recommended Strategy
Combine multiple approaches for maximum effect:

```
┌─────────────────────────────────────────────────────────────┐
│  1. Triangulation Init (from sensor onset times)            │
│     ↓                                                       │
│  2. PINN Refinement (if trained, ~1ms)                      │
│     ↓                                                       │
│  3. CMA-ES Exploration (8 parallel evals, 2-3 generations)  │
│     ↓                                                       │
│  4. L-BFGS-B Polish (2-3 iterations with adjoint gradients) │
│     ↓                                                       │
│  5. Return best candidate                                   │
└─────────────────────────────────────────────────────────────┘
```

### Time Budget (per sample, 7 workers)
| Step | Time | Cumulative |
|------|------|------------|
| Triangulation | 0.01s | 0.01s |
| PINN inference | 0.01s | 0.02s |
| CMA-ES (16 evals) | 4.0s | 4.02s |
| L-BFGS-B (3 iters) | 3.0s | 7.02s |
| **Total** | **~7s** | - |

With 7 workers: 400 samples × 7s / 7 workers = **67 minutes** (within budget with margin)

---

## Recommended Implementation Order

### Phase 1: Quick Wins (1-2 days)
1. **Triangulation initialization** - Low effort, immediate improvement
2. **CMA-ES experiment** - Replace L-BFGS-B, test on 80 samples

### Phase 2: Medium Effort (3-5 days)
3. **Adjoint method** - Implement for ADI solver, exact gradients
4. **Multi-fidelity GP** - Coarse exploration + fine refinement

### Phase 3: High Effort (1-2 weeks)
5. **PINN surrogate** - Train on synthetic data, use as initializer
6. **Full hybrid pipeline** - Combine all methods

---

## Score Projection

| Approach | Current | Projected | Confidence |
|----------|---------|-----------|------------|
| Baseline (max_iter=3) | 0.57 | - | - |
| + Triangulation init | - | 0.62 | High |
| + Adjoint gradients | - | 0.68 | Medium |
| + CMA-ES exploration | - | 0.72 | Medium |
| + PINN surrogate | - | 0.80+ | Low |

---

## References Summary

### Adjoint Methods
- [NC State Tutorial](https://aalexan3.math.ncsu.edu/articles/adjoint_based_gradient_and_hessian.pdf)
- [MIT Notes](https://math.mit.edu/~stevenj/18.336/adjoint.pdf)
- [Stanford Tutorial](https://cs.stanford.edu/~ambrad/adjoint_tutorial.pdf)

### Physics-Informed Neural Networks
- [Enhanced Surrogate Modelling (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0735193323000519)
- [Virtual Thermal Sensor PINN (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0017931023005380)
- [ThermoNet Heat Source Localization](https://www.sciencedirect.com/science/article/abs/pii/S0924424718314110)

### Multi-Fidelity Optimization
- [Royal Society Foundational Paper](https://royalsocietypublishing.org/doi/10.1098/rspa.2007.1900)
- [Multi-fidelity RBF Surrogate](https://link.springer.com/article/10.1007/s00158-020-02575-7)

### Differentiable Simulation
- [JAX-FEM](https://www.sciencedirect.com/science/article/abs/pii/S0010465523001479)
- [JAX, M.D.](https://papers.nips.cc/paper/2020/file/83d3d4b6c9579515e1679aca8cbc8033-Paper.pdf)
- [JAX-Fluids](https://www.sciencedirect.com/science/article/abs/pii/S0010465522002466)

### Evolutionary Optimization
- [CMA-ES Official](https://cma-es.github.io/)
- [CMA-ES Tutorial](https://arxiv.org/abs/1604.00772)

### Bayesian Approaches
- [Incremental Bayesian Inversion](https://www.sciencedirect.com/science/article/abs/pii/S0017931024014455)
- [Dynamic Bayesian Networks for IHCP](https://www.sciencedirect.com/science/article/abs/pii/S1290072922003659)

---

*Document created for Heat Signature Zero competition optimization research*

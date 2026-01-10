# Research: Next Steps for Heat Signature Zero

*Last updated: 2026-01-10 (Session 10)*
*CURRENT BEST: Early Timestep + 15/24 fevals → 1.0957 @ 58.3 min*

## LEADERBOARD GAP ANALYSIS (CRITICAL)

```
LEADERBOARD:                    OUR CURRENT:
#1  Jonas M      1.2268         1.0957 (84% of max)
#2  kjc          1.2265         Gap to top 5: -0.054
#3  MGöksu       1.1585         Gap to top 2: -0.124
#4  Matt Motoki  1.1581
#5  StarAtNyte   1.1261         TARGET: 1.15+ (top 5)
--- WE ARE HERE --- 1.0957      STRETCH: 1.20+ (top 2)
```

**Progress: Started at 0.77 → Now at 1.0957 (+42% improvement)**

---

## CRITICAL INSIGHT FROM SESSION 10

### A24 Position Refinement Proved Headroom Exists!

| Metric | Current (CMA-ES only) | With L-BFGS-B Refinement | Improvement |
|--------|----------------------|--------------------------|-------------|
| 2-src RMSE | 0.27 | **0.15** | **-44%** |
| 1-src RMSE | 0.18 | **0.13** | **-28%** |
| Score | 1.0957 | **1.1627** | **+6%** |
| Time | 58.3 min | 202 min | **3.4x over budget** |

**The accuracy is achievable - we just need FASTER gradient computation!**

L-BFGS-B uses finite differences: 2n+1 = 9 simulations per gradient for 4D (2-source).
This is the bottleneck. We need alternatives.

---

## PRIORITY QUEUE: FAST LOCAL REFINEMENT (NEW)

Based on web research, here are approaches to get L-BFGS-B-level accuracy without the cost:

| Priority | Approach | Expected Speedup | Effort | Research Basis |
|----------|----------|------------------|--------|----------------|
| **B1** | Nelder-Mead Refinement | 3-5x faster | Low | [Nelder-Mead](https://en.wikipedia.org/wiki/Nelder–Mead_method) - n+1 evals/iter vs 2n+1 |
| **B2** | Coordinate Descent | 4-8x faster | Low | [Fast Coordinate Descent](https://arxiv.org/abs/1803.01454) - 1D line searches |
| **B3** | Adjoint Gradient | 9x faster | Medium | [Adjoint Tutorial](https://www.sciencedirect.com/science/article/abs/pii/S0045782521001468) - O(1) gradients |
| **B4** | JAX Differentiable Sim | 9x faster | High | [PhiFlow](https://proceedings.mlr.press/v235/holl24a.html), [JAX-FEM](https://github.com/deepmodeling/jax-fem) |
| **B5** | LR-CMA-ES (Local Refinement) | 2-3x faster | Medium | [MDPI LR-CMA-ES](https://www.mdpi.com/2076-3417/14/19/9133) |
| **B6** | Model-and-Search (MAS) | 2-4x faster | Medium | [MAS 2025](https://link.springer.com/article/10.1007/s10589-025-00686-9) - new DFO for local |

---

## B1: Nelder-Mead Refinement (HIGHEST PRIORITY)

**Why this first**: Lowest effort, well-suited for our 2-4D problem.

**Research findings**:
- Nelder-Mead uses **n+1 evals per iteration** (vs L-BFGS-B's 2n+1 for finite diff)
- For 4D (2-source): 5 evals/iter vs 9 evals/iter = **1.8x fewer evals**
- "Works wonderfully up to a dozen or so dimensions" - perfect for our 2-4D problem
- "Extremely simple and often quite robust" - low implementation risk
- "Typically requires only one or two function evaluations per iteration, except in shrink"

**Implementation**:
```python
from scipy.optimize import minimize

def nelder_mead_refinement(best_params, objective, bounds):
    """Fast local refinement using Nelder-Mead."""
    result = minimize(
        objective,
        x0=best_params,
        method='Nelder-Mead',
        options={
            'maxiter': 10,      # Few iterations (we're already close)
            'xatol': 0.01,      # Position tolerance
            'fatol': 0.001,     # Function tolerance
            'adaptive': True,   # Dimension-adaptive parameters
        }
    )
    return result.x
```

**Expected impact**:
- If L-BFGS-B refinement took 150 min extra, Nelder-Mead might take ~50-80 min
- Could fit within budget with score ~1.10-1.12

**Test plan**:
```bash
# Create experiments/nelder_mead_refinement/
# Add Nelder-Mead after CMA-ES, before returning candidates
uv run python experiments/nelder_mead_refinement/run.py --workers 7 --shuffle --max-iter 10
```

---

## B2: Coordinate Descent Refinement

**Why this works**: For smooth functions in low dimensions, optimizing one coordinate at a time is efficient.

**Research findings**:
- "Coordinate descent algorithms are popular with practitioners owing to their simplicity"
- Each iteration optimizes ONE dimension via line search
- For 4D: 4 line searches per full iteration (vs 9 finite-diff evals for gradient)
- "Fast Best Subset Selection" paper shows it scales to millions of features

**Implementation**:
```python
from scipy.optimize import minimize_scalar

def coordinate_descent_refinement(params, objective, bounds, max_cycles=3):
    """Optimize each coordinate in sequence."""
    params = params.copy()
    for cycle in range(max_cycles):
        for i in range(len(params)):
            def line_objective(val):
                test_params = params.copy()
                test_params[i] = val
                return objective(test_params)

            result = minimize_scalar(
                line_objective,
                bounds=(bounds[i][0], bounds[i][1]),
                method='bounded'
            )
            params[i] = result.x
    return params
```

**Expected impact**:
- 3 cycles × 4 coords × ~3 evals/line = ~36 evals (vs L-BFGS-B's ~90 evals for 10 iters)
- **2.5x fewer simulations**

---

## B3: Adjoint Gradient Method

**Why revisit this**: We tested adjoint before but concluded "too slow". However, that was for FULL optimization. For refinement of already-good solutions, adjoint could work.

**Research findings**:
- [Nature 2024 Tutorial](https://www.nature.com/articles/s42005-024-01606-9): "Efficient handling of large systems with many unknown parameters"
- [ScienceDirect Tutorial](https://www.sciencedirect.com/science/article/abs/pii/S0045782521001468): "Gradient of functional computed efficiently"
- Key property: **"Number of computations is independent of number of parameters"**
- For 4D: 1 forward + 1 adjoint = 2 simulations per gradient (vs 9 for finite diff)

**Why it might work now**:
- Previous test: Full optimization from scratch = many iterations needed
- New approach: Refinement of CMA-ES solution = few iterations needed (1-3)
- With 1-3 L-BFGS-B iters using adjoint gradients: 6-18 sims vs 27-81 for finite diff

**Implementation approach**:
```python
def adjoint_gradient(params, Y_observed, solver):
    """Compute gradient via adjoint method - O(1) simulations."""
    # Forward solve
    T_forward = solver.solve(params)

    # Adjoint solve (backward in time with residual as forcing)
    residual = T_forward - Y_observed
    lambda_adjoint = solver.solve_adjoint(residual)  # NEW: implement adjoint solver

    # Gradient from inner product
    grad = compute_gradient_from_adjoint(lambda_adjoint, params)
    return grad
```

**Effort**: Medium - need to implement adjoint solver for Heat2D

---

## B4: JAX Differentiable Simulator

**Research findings**:
- [PhiFlow](https://proceedings.mlr.press/v235/holl24a.html): "Seamlessly integrates with PyTorch, TensorFlow, JAX"
- [JAX-FEM](https://github.com/deepmodeling/jax-fem): "Supports heat equation, differentiable programming"
- [Diffrax](https://github.com/patrick-kidger/diffrax): "Numerical differential equation solvers in JAX"

**Key benefit**: Automatic differentiation gives EXACT gradients for free
- No finite differences needed
- Gradient computation is ~same cost as forward pass

**Implementation approach**:
```python
import jax
import jax.numpy as jnp

@jax.jit
def differentiable_objective(params, Y_observed, solver_state):
    """Objective that JAX can differentiate through."""
    T_predicted = jax_heat_solve(params, solver_state)
    return jnp.mean((T_predicted - Y_observed)**2)

# Get gradient function for free!
grad_fn = jax.grad(differentiable_objective)

# Use with L-BFGS-B
from jax.scipy.optimize import minimize
result = minimize(differentiable_objective, x0=params, method='BFGS')
```

**Challenge**: Need to rewrite Heat2D solver in JAX
**Effort**: High (but potentially biggest payoff)

---

## B5: LR-CMA-ES (CMA-ES with Local Refinement)

**Research findings**:
- [MDPI 2024](https://www.mdpi.com/2076-3417/14/19/9133): "Knowledge-driven perturbation-based strategy significantly improves convergence speed and solution accuracy"
- Combines evolutionary learning (CMA-ES) with self-learning (local heuristics)
- "Enhances the algorithm's exploitability"

**Implementation**:
```python
import cma

def lr_cmaes_optimize(init, objective, bounds, max_fevals):
    """CMA-ES with local refinement steps."""
    es = cma.CMAEvolutionStrategy(init, sigma0=0.2)

    while not es.stop() and es.result.evaluations < max_fevals:
        solutions = es.ask()
        fitnesses = [objective(x) for x in solutions]
        es.tell(solutions, fitnesses)

        # Local refinement on best solution every N generations
        if es.countiter % 5 == 0:
            best = es.result.xbest
            refined = nelder_mead_refinement(best, objective, bounds, max_iter=3)
            refined_fit = objective(refined)
            if refined_fit < es.result.fbest:
                es.inject([refined], [refined_fit])

    return es.result.xbest
```

---

## B6: Model-and-Search (MAS) - New 2025 Algorithm

**Research findings**:
- [Springer 2025](https://link.springer.com/article/10.1007/s10589-025-00686-9): "Local-search DFO algorithm capable of finding (near-) local minima within a limited number of function evaluations"
- "Rapid convergence is desirable and would make this algorithm attractive for refining solutions"
- "Before evaluating new points, MAS prioritizes reusing information from previously evaluated points"

**Key insight**: MAS is designed EXACTLY for our use case - fast local refinement after global search

**Implementation**: Would need to implement from paper or find library

---

## EXPERIMENT QUEUE FOR RALPH LOOP

When the Ralph loop resumes, execute these experiments IN ORDER:

### Immediate (B1 - Nelder-Mead)
```
1. Create experiments/nelder_mead_refinement/
2. Copy early_timestep_opt as base
3. Add Nelder-Mead after CMA-ES, before candidate selection
4. Test configs:
   - 15/24 + NM(maxiter=5)
   - 15/24 + NM(maxiter=10)
   - 15/22 + NM(maxiter=10)
5. Target: Score 1.10+ within 60 min
```

### If B1 works, try B1+B2 combo
```
1. Add coordinate descent before Nelder-Mead
2. CD(2 cycles) + NM(5 iters) might be optimal
3. Target: Score 1.12+ within 60 min
```

### If B1-B2 don't help enough, try B5 (LR-CMA-ES)
```
1. Modify CMA-ES loop to inject refined solutions
2. Every 3-5 generations, run NM on best
3. This may require fewer total fevals
```

### Parallel track: B3 (Adjoint)
```
1. Implement adjoint solver for Heat2D
2. Test adjoint gradients vs finite diff (accuracy check)
3. If accurate, use for 1-3 refinement iterations
4. Could be game-changer if it works
```

---

## WHAT NOT TO TRY (Already Tested, Not Effective)

| Approach | Result | Why It Failed |
|----------|--------|---------------|
| Sequential 2-source (A23) | -2% score | High variance, not stable |
| Higher fevals (15/26+) | Over budget | Timing variance too high |
| Sensor subset diversity | Same solutions | TAU filter rejects similar candidates |
| Gradient triangulation | Hurts score | RBF interpolation gives noisy gradients |
| Bayesian optimization | -2.7% score | GP overhead doesn't pay off |
| Multi-start CMA-ES | Dilutes fevals | Each restart gets too few evals |
| ICA decomposition | Over budget | Best score but 27 min over |

---

## CURRENT PRODUCTION CONFIGURATION

```bash
# Best verified config within budget
uv run python experiments/early_timestep_opt/run.py \
    --workers 7 \
    --shuffle \
    --early-fraction 0.3 \
    --max-fevals-2src 24

# Score: 1.0957 @ 58.3 min (1.7 min buffer)
```

---

## SCORE JOURNEY

```
Session 1-3:   0.77 → 1.02  (baseline improvements)
Session 8:     1.02 → 0.995 (verified baseline)
Session 9:     0.995 → 1.08 (early timestep breakthrough +8.5%)
Session 10:    1.08 → 1.096 (extended fevals +1.5%)

Total: +42% improvement from start

Gap remaining:
- To top 5 (1.15): need +5% more
- To top 2 (1.22): need +11% more
```

---

## KEY PHYSICS INSIGHTS

1. **Heat equation is LINEAR in intensity q**: T(x,t) = q × T_unit(x,t)
   - Optimal q has closed-form solution (already exploited)

2. **Early timesteps are more discriminative**:
   - Contain onset timing → distance from source
   - Breaks 2-source symmetry/degeneracy
   - 30% early_fraction is optimal

3. **2-source RMSE is the bottleneck**:
   - Current: ~0.27
   - Achievable: ~0.15 (proven by A24)
   - Need 1.8x improvement in 2-source accuracy

4. **Position refinement works, just slow**:
   - L-BFGS-B can reach excellent accuracy
   - But finite differences cost 9 sims/gradient for 4D
   - Need faster gradient or derivative-free local search

---

## REFERENCES

### Fast Local Optimization
- [Nelder-Mead Wikipedia](https://en.wikipedia.org/wiki/Nelder–Mead_method)
- [Coordinate Descent Fast Subset Selection](https://arxiv.org/abs/1803.01454)
- [Model-and-Search (MAS) 2025](https://link.springer.com/article/10.1007/s10589-025-00686-9)
- [LR-CMA-ES with Local Refinement](https://www.mdpi.com/2076-3417/14/19/9133)

### Adjoint Methods
- [Adjoint Tutorial (Nature 2024)](https://www.nature.com/articles/s42005-024-01606-9)
- [Adjoint for PDE Optimization](https://www.sciencedirect.com/science/article/abs/pii/S0045782521001468)
- [NC State Adjoint Tutorial](https://aalexan3.math.ncsu.edu/articles/adjoint_based_gradient_and_hessian.pdf)

### Differentiable Simulators
- [PhiFlow (JAX/PyTorch/TensorFlow)](https://proceedings.mlr.press/v235/holl24a.html)
- [JAX-FEM (Differentiable FEM)](https://github.com/deepmodeling/jax-fem)
- [Diffrax (JAX ODE/PDE solvers)](https://github.com/patrick-kidger/diffrax)

### Heat Source Estimation
- [Bayesian Heat Source Estimation](https://arxiv.org/html/2405.02319)
- [Recursive Least Squares for Heat Source](https://www.osti.gov/biblio/418044)
- [2D Inverse Heat Source](https://www.sciencedirect.com/science/article/abs/pii/S0017931097001257)

---

*Document updated 2026-01-10 after Session 10*
*Current BEST: 1.0957 @ 58.3 min (Early Timestep + 15/24 fevals)*
*Next priority: B1 Nelder-Mead Refinement*

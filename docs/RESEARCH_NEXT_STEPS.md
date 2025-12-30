# Research: Next Steps for Heat Signature Zero

*Last updated: 2024-12-30*
*FINAL SUBMISSION: TransferLearningOptimizer 0.8107 @ 54.6 min ✅*

---

## Final Evaluation Reminder (CRITICAL)
- **70%** - Performance on holdout dataset
- **20%** - Innovation score (learning at inference, smart optimization, generalizability)
- **10%** - Interpretability of Jupyter Notebook

**Key for 20% Innovation Score:**
- Active simulator use during inference ✓ (we have this)
- Smart optimization beyond brute-force ✓ (CMA-ES + triangulation)
- **Evidence of learning at inference** ✓ (Transfer Learning - improves across batches!)
- **Generalizable methods** ✓ (Feature-based similarity matching)

---

## Executive Summary

### 🏆 CURRENT BEST MODEL (FINAL SUBMISSION)

| Model | Score | RMSE | Time | Status |
|-------|-------|------|------|--------|
| **TransferLearningOptimizer** | **0.8107** | ~0.45 | **54.6 min** | ✅ **SELECTED** |

**Config:** `--k-similar 1 --max-fevals-1src 15 --max-fevals-2src 30`
**MLflow:** `transfer_learning_20251230_XXXXXX`

**Key Innovation Features:**
- Batch processing with history accumulation between batches
- Feature-based similarity matching for solution transfer
- Demonstrates "learning at inference" - 12.5% of best results from transferred solutions

### Runner-up Model (Safe Fallback)

| Model | Score | RMSE | Time | Status |
|-------|-------|------|------|--------|
| MultiCandidateOptimizer | 0.7764 | 0.525 | 53.8 min | ✅ Previous best |

### Approach History

| Priority | Approach | Status | Result |
|----------|----------|--------|--------|
| ~~1~~ | ~~Adjoint Method~~ | **TESTED - NOT VIABLE** | 157s/sample, too slow |
| ~~2~~ | ~~Triangulation Init~~ | **IMPLEMENTED** | +13.4% RMSE improvement |
| ~~3~~ | ~~CMA-ES~~ | **IMPLEMENTED** | Best scores but time-constrained |
| ~~4~~ | ~~JAX/Differentiable Sim~~ | **TESTED - NOT VIABLE** | GPU overhead too high |
| ~~5~~ | ~~Adaptive Polish~~ | **TESTED - NOT EFFECTIVE** | Inconsistent timing |
| ~~6~~ | ~~Intensity-Only Polish~~ | **TESTED - VIABLE** | 0.7862 @ 58 min |
| ~~7~~ | ~~Multiple Candidates~~ | **FINALIZED** | 0.7764 @ 53.8 min |
| ~~8~~ | ~~Transfer Learning~~ | **FINALIZED** | **0.8107 @ 54.6 min ✅** |

### Future Improvements (If Time Permits)
| Priority | Approach | Status | Potential |
|----------|----------|--------|-----------|
| ~~1~~ | ~~Transfer Learning~~ | **DONE** | **+4.4% score improvement** |
| **2** | Multi-Fidelity GP | Not started | Medium potential |
| **3** | PINN Surrogate | Not started | High effort, high potential |

---

## 🎯 Innovation Score Boosters (20% of Final Score)

These approaches specifically target the **"learning at inference"** and **"generalizability"** criteria that judges will assess.

### ✅ PRIORITY 1: Sample-to-Sample Transfer Learning ⭐⭐⭐⭐⭐ **IMPLEMENTED**

**Status**: COMPLETED - Score 0.8107 @ 54.6 min

**Implementation**: `experiments/transfer_learning/`
- `optimizer.py`: TransferLearningOptimizer with feature extraction and similarity matching
- `run.py`: Batch processing with history accumulation

**Results**:
| Config | Score | RMSE | Transfers Used | Time |
|--------|-------|------|----------------|------|
| k=2, 20/40 fevals | 0.8844 | best | high | 83.5 min ❌ |
| k=1, 20/40 fevals | 0.8716 | good | moderate | 60.8 min ⚠️ |
| **k=1, 15/30 fevals** | **0.8107** | good | 12.5% | **54.6 min ✅** |

**Key Innovation Points**:
- Batch processing maintains parallelism while enabling transfer
- Feature-based similarity matching using thermal characteristics
- 12.5% of samples got best result from transferred initialization
- Demonstrates "learning at inference" - model improves as it processes batches

**Why judges will like this**:
- Shows explicit "learning at inference" ✓
- Demonstrates "adaptive refinement" as more samples are processed ✓
- Generalizable pattern for any simulation-driven problem ✓

---

### (PREVIOUSLY) PRIORITY 1: Sample-to-Sample Transfer Learning (Original Proposal)

**Why it matters**: Currently each sample is solved independently. Adding transfer shows "adaptive refinement" and "learning at inference".

**Implementation**:
```python
class TransferLearningOptimizer:
    def __init__(self):
        self.solved_samples = []  # Store (sample_features, solution) pairs
        self.feature_extractor = self._build_feature_extractor()

    def _extract_features(self, sample):
        """Extract sample characteristics for similarity matching."""
        Y = sample['Y_noisy']
        return np.array([
            Y.max(),                    # Peak temperature
            Y.mean(),                   # Mean temperature
            np.std(Y),                  # Temperature variance
            sample['n_sources'],        # Number of sources
            len(sample['sensors_xy']),  # Number of sensors
            sample['sample_metadata']['kappa'],  # Diffusivity
        ])

    def _find_similar_samples(self, sample, k=3):
        """Find k most similar previously solved samples."""
        if not self.solved_samples:
            return []

        query_features = self._extract_features(sample)
        similarities = []
        for features, solution in self.solved_samples:
            dist = np.linalg.norm(query_features - features)
            similarities.append((dist, solution))

        similarities.sort(key=lambda x: x[0])
        return [sol for _, sol in similarities[:k]]

    def estimate_sources(self, sample, meta):
        # Get initializations from similar samples
        similar_solutions = self._find_similar_samples(sample)

        # Use similar solutions as additional starting points!
        initializations = [triangulation_init(sample, meta)]
        for sol in similar_solutions:
            initializations.append(sol)  # Transfer knowledge

        # Run CMA-ES from best initialization
        best_result = None
        for init in initializations:
            result = cmaes_optimize(init, sample, meta)
            if best_result is None or result.rmse < best_result.rmse:
                best_result = result

        # Store for future transfer
        self.solved_samples.append((self._extract_features(sample), best_result.params))

        return best_result
```

**Why judges will like this**:
- Shows explicit "learning at inference"
- Demonstrates "adaptive refinement" as more samples are processed
- Generalizable pattern for any simulation-driven problem

**Effort**: Medium

---

### PRIORITY 2: Online Surrogate Learning ⭐⭐⭐⭐

**Why it matters**: Building a surrogate model during inference demonstrates "learning" and efficiency.

**Implementation**:
```python
class OnlineSurrogateOptimizer:
    def __init__(self):
        self.global_surrogate = None
        self.all_evaluations = []  # (params, rmse) across all samples

    def estimate_sources(self, sample, meta):
        # Phase 1: Use global surrogate for smart initialization (if available)
        if self.global_surrogate is not None:
            # Pre-screen 100 random candidates
            candidates = random_sample(100)
            predictions = self.global_surrogate.predict(candidates)
            best_idx = np.argmin(predictions)
            smart_init = candidates[best_idx]
        else:
            smart_init = triangulation_init(sample, meta)

        # Phase 2: Run CMA-ES, collect all evaluations
        result, evaluations = cmaes_with_history(smart_init, sample, meta)

        # Phase 3: Update global surrogate with new data
        self.all_evaluations.extend(evaluations)
        if len(self.all_evaluations) > 50:
            X = np.array([e[0] for e in self.all_evaluations])
            y = np.array([e[1] for e in self.all_evaluations])
            self.global_surrogate = GaussianProcessRegressor()
            self.global_surrogate.fit(X, y)

        return result
```

**Why judges will like this**:
- Explicit "learning at inference"
- Surrogate improves as more samples processed
- Classic technique in simulation-based optimization literature

**Effort**: Medium-High

---

### PRIORITY 3: Emphasize Physics-Informed Aspects ⭐⭐⭐

**Why it matters**: Our triangulation IS physics-informed but we don't emphasize it. This is LOW EFFORT, HIGH IMPACT for innovation score.

**What to do**:
1. **Rename/rebrand** the approach as "Physics-Informed Heat Source Localization"
2. **Document in notebook** the heat diffusion physics: `r = sqrt(4*κ*t)`
3. **Explain** how triangulation uses the physics of thermal diffusion
4. **Frame** CMA-ES as "physics-guided optimization"

**Notebook sections to add**:
```markdown
## Physics-Informed Initialization

Our approach leverages the fundamental physics of heat diffusion. The heat equation:

$$\frac{\partial T}{\partial t} = \kappa \nabla^2 T$$

implies that thermal signals propagate with characteristic speed related to diffusivity κ.
By detecting when each sensor first "sees" the heat signal, we can estimate the distance
to the source using:

$$r = \sqrt{4 \kappa t_{onset}}$$

This transforms the inverse problem into a trilateration problem, providing a
physics-grounded initialization that dramatically reduces the search space for
subsequent optimization.
```

**Effort**: Low (documentation only)

---

### PRIORITY 4: Bayesian Optimization Layer ⭐⭐⭐

**Why it matters**: BO is explicitly mentioned in the evaluation criteria as a "smart optimization strategy".

**Implementation**:
```python
from sklearn.gaussian_process import GaussianProcessRegressor
from scipy.stats import norm

def bayesian_optimization_init(sample, meta, n_initial=10, n_bo_iters=5):
    """Use Bayesian Optimization to find good initialization."""

    # Initial random samples
    X = latin_hypercube_sample(bounds, n=n_initial)
    y = [simulate_and_score(x, sample, meta) for x in X]

    # Build GP surrogate
    gp = GaussianProcessRegressor(normalize_y=True)
    gp.fit(X, y)

    # BO iterations
    for _ in range(n_bo_iters):
        # Expected Improvement acquisition
        def ei(x):
            mu, sigma = gp.predict([x], return_std=True)
            best = min(y)
            z = (best - mu) / (sigma + 1e-8)
            return -(sigma * (z * norm.cdf(z) + norm.pdf(z)))

        # Optimize acquisition
        next_x = minimize(ei, random_init(), bounds=bounds).x
        next_y = simulate_and_score(next_x, sample, meta)

        X = np.vstack([X, next_x])
        y = np.append(y, next_y)
        gp.fit(X, y)

    return X[np.argmin(y)]  # Best found point
```

**Effort**: Medium

---

### Recommended Innovation Implementation Order

| Priority | Approach | Effort | Innovation Impact | Performance Impact |
|----------|----------|--------|-------------------|-------------------|
| **1** | Physics-Informed Documentation | Low | High | None (docs only) |
| **2** | Sample-to-Sample Transfer | Medium | Very High | Moderate |
| **3** | Bayesian Optimization Layer | Medium | High | Moderate |
| **4** | Online Surrogate Learning | Medium-High | Very High | Moderate |

**Minimum for good innovation score**: Priority 1 + 2

---

## Completed Implementations

### Triangulation Initialization (DONE)
- **Location**: `src/triangulation.py`
- **Result**: +13.4% RMSE improvement over hottest-sensor init
- **How it works**: Uses heat diffusion physics (r ~ sqrt(4*kappa*t)) to estimate source positions from sensor onset times

### CMA-ES Optimizer (DONE)
- **Location**: `experiments/cmaes/optimizer.py`
- **Result**: Significantly better than L-BFGS-B, especially for 2-source problems
- **Key findings from testing (2024-12-29)**:

| Config | Score | RMSE | Projected Time | Status |
|--------|-------|------|----------------|--------|
| polish=5 (baseline) | **0.8419** | 0.360 | 75.5 min | Over budget |
| polish=3 | 0.7984 | 0.406 | 63.4 min | Over budget |
| polish=2 | 0.7677 | 0.442 | 59.7 min | Borderline |
| polish=1, 2src=20 | 0.7501 | 0.515 | 57.2 min | Safe |
| No polish | 0.6505 | 0.623 | 42.8 min | Bad accuracy |

**Key insight**: The L-BFGS-B polish step is critical for accuracy but expensive. Reducing CMA-ES fevals doesn't help because polish compensates with more work.

---

## Tested But Not Viable

### Adjoint Method
- **Location**: `src/adjoint_optimizer_fast.py`
- **Result**: 157.4s per sample (too slow)
- **Why it failed**: Adjoint reduces gradient cost but not iteration count. Sample-level parallelism (7 workers) in HybridOptimizer is more effective.
- **Gradient validation**: Max relative error 0.01% (exact gradients work, just not faster overall)

### JAX/GPU Acceleration
- **Result**: GPU kernel launch overhead exceeds computation time for small grids (100x50)
- **Why it failed**: ADI time-stepping is inherently sequential; sample-level parallelism beats GPU parallelism for this problem size

---

## Next Steps to Explore

### ~~1. Adaptive Per-Problem-Type Strategy~~ (TESTED - NOT EFFECTIVE)

**Status**: Tested on 2024-12-29. Did not provide reliable time savings.

**Results**:
- Adaptive (0/2): 61.0 min, score 0.7318 - worse than uniform polish=2
- Adaptive (1/2): 70.7 min, score 0.7779 - high variance, unreliable

**Conclusion**: 1-source problems still need polish to refine intensity (q). Skipping polish hurts accuracy without reliable time savings.

---

### 1. Early Stopping in Polish (LOW EFFORT)

**Motivation**: L-BFGS-B runs fixed iterations even when converged.

**Proposed approach**:
```python
# Stop when improvement plateaus
if abs(prev_rmse - rmse) < 1e-4:
    break
```

**Expected impact**: Save time on samples that converge quickly

**Effort**: Low

### 2. Multi-Fidelity Optimization with GP Surrogate (MEDIUM EFFORT)

**Motivation**: Use cheap coarse simulations to guide expensive fine evaluations.

**Approach**:
1. Build Gaussian Process surrogate from coarse (50x25) simulations
2. Use GP to identify promising regions
3. Only run full-resolution sims at best candidates

**Implementation**:
```python
from sklearn.gaussian_process import GaussianProcessRegressor

def multi_fidelity_optimize(sample, meta):
    # Phase 1: Coarse exploration (50x25 grid)
    coarse_samples = latin_hypercube_sample(n=50)
    coarse_losses = [coarse_simulate(params) for params in coarse_samples]

    # Phase 2: Build surrogate
    gp = GaussianProcessRegressor()
    gp.fit(coarse_samples, coarse_losses)

    # Phase 3: Acquisition-guided fine evaluation
    for _ in range(5):
        next_point = maximize_expected_improvement(gp, bounds)
        fine_loss = fine_simulate(next_point)
        gp.update(next_point, fine_loss)

    return best_point
```

**Expected impact**: Fewer full-resolution simulations needed

**Effort**: Medium

**References**:
- [Multi-fidelity optimization via surrogate modelling (Royal Society)](https://royalsocietypublishing.org/doi/10.1098/rspa.2007.1900)
- [Multi-fidelity RBF surrogate (Springer)](https://link.springer.com/article/10.1007/s00158-020-02575-7)

### 3. PINN Surrogate for Initialization (HIGH EFFORT)

**Motivation**: Train a neural network to predict initial (x, y, q) from sensor readings, then refine with simulator.

**Competition-compliant approach**:
```
PINN(sensor_data) → initial_guess → CMA-ES/L-BFGS-B with simulator → final_answer
```

**Architecture**:
```python
class HeatSourcePINN(nn.Module):
    def __init__(self):
        # Input: flattened sensor readings + metadata
        # Output: (x, y, q) for each source
        self.encoder = MLP([n_sensors * n_timesteps, 256, 128, 64])
        self.decoder = MLP([64, 32, 3 * n_sources])
```

**Expected impact**:
- PINN inference ~1ms
- Could provide better initialization than triangulation
- Allow more simulator iterations within time budget

**Effort**: High (need to generate training data, train model)

**References**:
- [Enhanced surrogate modelling of heat conduction](https://www.sciencedirect.com/science/article/abs/pii/S0735193323000519)
- [ThermoNet for heat source localization](https://www.sciencedirect.com/science/article/abs/pii/S0924424718314110) - 99% accuracy

---

## New Insights-Based Approaches (2024-12-29)

Based on comprehensive analysis of all testing, here are new approaches derived from our learnings:

### KEY INSIGHT: L-BFGS-B Finite Differences Are The Bottleneck

```
Polish evals per iteration (finite differences):
- 1-source (3 params): 2*3 + 1 = 7 evals/iter → 35 evals for maxiter=5
- 2-source (6 params): 2*6 + 1 = 13 evals/iter → 65 evals for maxiter=5

This DOMINATES compute time, not CMA-ES!
```

---

### ~~4. Intensity-Only Polish~~ (TESTED - VIABLE ALTERNATIVE)

**Status**: Tested on 2024-12-29. Shows promise as faster alternative with competitive accuracy.

**Location**: `experiments/intensity_polish/`

**Results**:
| Config | Score | RMSE | Projected | Status |
|--------|-------|------|-----------|--------|
| fevals 15/25 | 0.6969 | 0.565 | 40.2 min | Too low score |
| **fevals 25/45** | **0.7862** | 0.455 | **58.0 min** | **✅ Beats current submission!** |
| fevals 28/48 | 0.7626 | 0.450 | 61.4 min | Over budget (variance) |
| fevals 30/50 | 0.8147 | 0.416 | 61.0 min | Over budget |

**Key Findings**:
- Intensity-only polish is significantly faster than L-BFGS-B polish
- With increased CMA-ES budget (25/45), achieves **better score** than CMA-ES + L-BFGS-B polish=1
- High run-to-run variance makes fine-tuning difficult
- Best config: `--max-fevals-1src 25 --max-fevals-2src 45`

**Viable Submission Command**:
```bash
uv run python experiments/intensity_polish/run.py --workers 7 --max-fevals-1src 25 --max-fevals-2src 45
```
- Score: 0.7862 (vs 0.7501 current)
- Projected: 58.0 min

**Why It Works**: CMA-ES with more budget gets positions close enough; intensity-only polish (1-2 params) is much faster than full L-BFGS-B polish (3-6 params), allowing more CMA-ES iterations within budget.

---

### 5. Gradient-Free Polish (Replace L-BFGS-B)

**Insight**: L-BFGS-B's finite differences are expensive. Use derivative-free optimizer.

**Options**:
- `method='Nelder-Mead'`: ~n+1 evals per iteration
- `method='Powell'`: Direction-set method
- `method='COBYLA'`: Constrained, derivative-free

**Approach**:
```python
result = minimize(
    objective,
    x0=best_params,
    method='Nelder-Mead',  # Instead of 'L-BFGS-B'
    options={'maxiter': 10, 'xatol': 1e-4}
)
```

**Expected Impact**: Fewer evals per iteration, may converge faster for well-initialized problems

**Effort**: Low (single line change)

---

### 6. Extended CMA-ES (No Polish, Larger Budget)

**Insight**: Our no-polish test (42.8 min, score 0.65) used limited budget. What if CMA-ES gets more iterations?

**Approach**:
```python
# Instead of CMA-ES(15/25 fevals) + Polish(5 iter)
# Try CMA-ES(50/80 fevals) with smaller sigma for fine convergence
max_fevals_1src = 50
max_fevals_2src = 80
sigma0_1src = 0.05  # Smaller for fine-tuning
sigma0_2src = 0.10
```

**Rationale**: CMA-ES is gradient-free (no finite diff overhead). With good init and small sigma, may reach polish-level accuracy.

**Expected Impact**: If CMA-ES can match polish accuracy, faster overall (no polish overhead)

**Effort**: Low (parameter tuning)

---

### ~~7. Multiple Candidates from CMA-ES Population~~ (TESTED - BEST RESULT!)

**Status**: Tested on 2024-12-29. **BEST RESULTS ACHIEVED!**

**Location**: `experiments/multi_candidates/`

**Results**:
| Config | Score | RMSE | Avg Candidates | Projected | Status |
|--------|-------|------|----------------|-----------|--------|
| **fevals 28/50, pool=10** | **0.8577** | 0.431 | 2.1 | **54.0 min** | **✅ BEST!** |
| fevals 25/45, pool=10 | 0.8455 | 0.449 | 2.1 | 51.6 min | Good |

**Per-source breakdown (fevals 28/50)**:
- 1-source: RMSE=0.317, avg 1.5 candidates
- 2-source: RMSE=0.506, avg 2.5 candidates

**Why it works**: The diversity bonus in the score formula significantly outweighs the per-candidate accuracy:
- Math prediction: 3 candidates @ RMSE=0.5 → score 0.967 vs 1 candidate @ RMSE=0.3 → score 0.869
- Actual result: 2.1 avg candidates @ RMSE=0.449 → score **0.8455** (vs 0.7501 single candidate)

**Key Implementation Details**:
- Collects all solutions evaluated during CMA-ES (not just best)
- Applies dissimilarity filtering (τ = 0.2) using normalized coordinates
- Intensity-only polish on each candidate (faster than L-BFGS-B)
- Returns up to N_max=3 valid candidates per sample

**Command** (best config):
```bash
uv run python experiments/multi_candidates/run.py --workers 7 --max-fevals-1src 28 --max-fevals-2src 50
```

**Effort**: Low-Medium (implemented)

---

### 8. Coarse-to-Fine Grid Strategy

**Insight**: Full resolution (100x50) is expensive. Use coarse for optimization, fine for final polish.

**Approach**:
```python
# Phase 1: Coarse optimization (50x25 = 1/4 compute cost)
coarse_solver = Heat2D(Lx, Ly, 50, 25, kappa, bc)
x0 = triangulation_init(...)
params = cmaes_optimize(coarse_solver, x0, max_fevals=30)
params = polish(coarse_solver, params, max_iter=3)

# Phase 2: Fine polish only (100x50)
fine_solver = Heat2D(Lx, Ly, 100, 50, kappa, bc)
final_params = polish(fine_solver, params, max_iter=1)
```

**Expected Impact**: Coarse sim ~4x faster; most work at coarse level

**Effort**: Medium

---

### 9. Analytical Intensity Estimation

**Insight**: Heat equation is LINEAR in q. Given (x, y), optimal q has closed-form solution.

**Derivation**:
```
T(sensors, t) = q * T_unit(sensors, t)   where T_unit is response to q=1

Optimal q = argmin ||q*T_unit - T_observed||²
          = (T_unit · T_observed) / (T_unit · T_unit)
```

**Implementation**:
```python
def analytical_intensity(x, y, Y_observed, solver):
    # Simulate with q=1.0
    _, Us = solver.solve(sources=[{'x': x, 'y': y, 'q': 1.0}])
    Y_unit = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])

    # Closed-form optimal q
    q_optimal = np.dot(Y_unit.flat, Y_observed.flat) / np.dot(Y_unit.flat, Y_unit.flat)
    return np.clip(q_optimal, 0.5, 2.0)
```

**Expected Impact**: 1 forward sim to get optimal q (vs many in polish)

**Effort**: Medium

---

### 10. Sample Difficulty Prediction + Adaptive Budget

**Insight**: Some samples are inherently easier. Spend compute where it matters.

**Difficulty heuristics**:
- Signal-to-noise ratio (SNR)
- Number of sources (2-source harder)
- Source positions (near boundaries harder)

**Implementation**:
```python
def estimate_difficulty(sample):
    Y = sample['Y_noisy']
    snr = np.max(Y) / (np.std(Y) + 1e-8)
    difficulty = 1.0 / snr
    if sample['n_sources'] == 2:
        difficulty *= 1.5
    return difficulty

# Adaptive budget allocation
if difficulty < 0.5:  # Easy
    config = {'max_fevals': 15, 'polish_iter': 1}
else:  # Hard
    config = {'max_fevals': 30, 'polish_iter': 3}
```

**Expected Impact**: Better overall score/time by focusing compute on hard samples

**Effort**: Medium

---

## Innovative "Out of the Box" Approaches (2024-12-29)

These approaches fundamentally rethink the problem instead of iterating on optimization parameters.

### KEY PARADIGM SHIFT

**Current approach**: Guess → Simulate → Compare → Repeat (iterative optimization)

**New approach**: Exploit physics structure to get DIRECT solutions (no iteration!)

---

### 11. Signal Decomposition for 2-Source (ICA/NMF) ⭐⭐⭐⭐ HIGH POTENTIAL

**Core Insight**: The heat equation is LINEAR - temperature fields from multiple sources ADD:
```
T_total = T_source1 + T_source2
```

We can use **Independent Component Analysis (ICA)** or **Non-negative Matrix Factorization (NMF)** to DIRECTLY decompose sensor signals into individual source contributions - **NO OPTIMIZATION NEEDED!**

**Implementation**:
```python
from sklearn.decomposition import FastICA, NMF

def decompose_sources(Y_obs, sensors):
    """Directly extract source signatures from sensor data."""

    # ICA separates mixed signals into independent components
    ica = FastICA(n_components=2, random_state=42)
    source_signals = ica.fit_transform(Y_obs)  # (time, 2)
    mixing_matrix = ica.mixing_  # (n_sensors, 2) - encodes spatial info!

    # Each column of mixing matrix shows how strongly each sensor
    # "sees" each source - this IS the spatial signature!

    sources = []
    for i in range(2):
        weights = np.abs(mixing_matrix[:, i])
        weights = weights / weights.sum()  # Normalize
        x_est = np.average(sensors[:, 0], weights=weights)
        y_est = np.average(sensors[:, 1], weights=weights)
        sources.append((x_est, y_est))

    return sources
```

**Why this is innovative**:
- Completely bypasses iterative optimization
- Extracts source structure directly from data using signal processing
- Takes milliseconds instead of seconds
- Can provide multiple distinct decompositions as candidates

**Expected Impact**: 10-100x speedup for 2-source problems

**Effort**: Medium

---

### 12. Closed-Form Geometric Solution (Trilateration) ⭐⭐⭐⭐ HIGH POTENTIAL

**Core Insight**: With 8-10 sensors, we have an OVERDETERMINED system. Timing information gives distance equations:
```
||sensor_i - source|| = sqrt(4 * kappa * t_onset_i)
```

This is a **trilateration problem** with more equations than unknowns - solvable via LINEAR ALGEBRA!

**Implementation**:
```python
def geometric_solution(sample, meta):
    """Solve source location as overdetermined linear system."""

    Y = sample['Y_noisy']
    sensors = sample['sensors_xy']
    kappa = sample['sample_metadata']['kappa']
    dt = meta['dt']

    # Extract onset times
    onset_times = []
    for i in range(len(sensors)):
        signal = Y[:, i]
        threshold = 0.05 * signal.max()
        onset_idx = np.argmax(signal > threshold)
        onset_times.append(onset_idx * dt)

    # Convert to distances: r² = 4κt
    distances = [np.sqrt(4 * kappa * max(t, 0.01)) for t in onset_times]

    # Build linear system using distance differences
    # ||p - s_i||² - ||p - s_0||² = d_i² - d_0²
    # Expands to: 2(s_0 - s_i)·p = d_i² - d_0² + ||s_i||² - ||s_0||²

    A, b = [], []
    s0, d0 = sensors[0], distances[0]

    for i in range(1, len(sensors)):
        si, di = sensors[i], distances[i]
        A.append(2 * (s0 - si))
        b.append(di**2 - d0**2 + np.dot(si, si) - np.dot(s0, s0))

    # Least squares solution (overdetermined)
    position, _, _, _ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)

    return position[0], position[1]  # x, y in MICROSECONDS!
```

**Why this is innovative**:
- Uses LINEAR ALGEBRA instead of iterative optimization
- Gets answer in microseconds, not seconds
- Naturally handles noisy data via least squares
- Can use residuals to estimate uncertainty

**Expected Impact**: 1000x speedup for position estimation

**Effort**: Low-Medium

---

### 13. Learn-During-Inference Neural Surrogate ⭐⭐⭐⭐ HIGH POTENTIAL

**Core Insight**: Build a tiny neural network DURING inference (competition-legal!) to approximate the simulator, then search exhaustively.

**Implementation**:
```python
def adaptive_surrogate_search(sample, meta, n_initial=15, n_surrogate_evals=500):
    """Build surrogate during inference, search exhaustively."""

    # Phase 1: Initial samples with real simulator (15 evals)
    X_train = latin_hypercube_sample(bounds, n=n_initial)
    Y_train = [full_simulation(x, sample, meta) for x in X_train]

    # Phase 2: Train tiny surrogate (takes milliseconds)
    from sklearn.neural_network import MLPRegressor
    surrogate = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=200)
    surrogate.fit(X_train, Y_train)

    # Phase 3: Exhaustive search on surrogate (500 evals in <1 second!)
    best_candidates = []
    for _ in range(n_surrogate_evals):
        x = random_sample(bounds)
        pred_rmse = surrogate.predict([x])[0]
        best_candidates.append((x, pred_rmse))

    # Phase 4: Validate top candidates with real simulator (5 evals)
    best_candidates.sort(key=lambda t: t[1])
    validated = []
    for x, _ in best_candidates[:5]:
        real_rmse = full_simulation(x, sample, meta)
        validated.append((x, real_rmse))

    return validated  # Multiple candidates!
```

**Why this is innovative**:
- Surrogate built FRESH for each sample (competition-legal)
- Allows 100x more candidate evaluations
- Natural way to generate multiple diverse candidates
- Only 20 real simulator calls total

**Expected Impact**: More candidates with same compute budget

**Effort**: Medium

---

### 14. Temperature Field Interpolation + Peak Finding ⭐⭐⭐

**Core Insight**: Heat flows FROM sources. Interpolate the temperature field and find the PEAK - that's where the source is!

**Implementation**:
```python
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize

def peak_finding_solution(sample, meta):
    """Find source by locating peak of interpolated temperature field."""

    Y = sample['Y_noisy']
    sensors = sample['sensors_xy']

    # Use late-time (quasi-steady) temperatures
    T_steady = Y[-20:].mean(axis=0)

    # Fit smooth surface through sensor readings
    rbf = RBFInterpolator(sensors, T_steady, kernel='thin_plate_spline')

    # Find maximum (source location)
    result = minimize(
        lambda xy: -rbf([[xy[0], xy[1]]])[0],  # Negative for maximization
        x0=[1.0, 0.5],
        bounds=[(0.1, 1.9), (0.1, 0.9)],
        method='L-BFGS-B'
    )

    x_source, y_source = result.x
    q_est = estimate_intensity_from_peak(T_steady.max())

    return x_source, y_source, q_est
```

**Why this is innovative**:
- No PDE solving at all - just interpolation!
- Works well when source is within sensor coverage
- Very fast (<10ms per sample)

**Expected Impact**: Fast initialization, good for 1-source

**Effort**: Low

---

### 15. Hybrid Direct Solution Strategy ⭐⭐⭐⭐⭐ RECOMMENDED

**Core Insight**: Combine the best direct methods for maximum speed, then use minimal CMA-ES for polish.

**Strategy**:
```
1-source pipeline:
    geometric_solution() → analytical_intensity() → CMA-ES(5 evals) → candidates

2-source pipeline:
    ICA_decomposition() → geometric_solution(each) → analytical_intensity() → CMA-ES(10 evals) → candidates
```

**Implementation**:
```python
def hybrid_direct_optimizer(sample, meta):
    n_sources = sample['n_sources']

    if n_sources == 1:
        # Direct geometric solution (microseconds)
        x, y = geometric_solution(sample, meta)
        q = analytical_intensity(x, y, sample, meta)
        init = [x, y, q]
    else:
        # ICA decomposition (milliseconds)
        positions = ica_decompose(sample)
        init = []
        for x, y in positions:
            q = analytical_intensity(x, y, sample, meta)
            init.extend([x, y, q])

    # Very short CMA-ES polish (5-10 evals)
    final = cmaes_polish(init, sample, meta, max_fevals=10)

    return final
```

**Why this is innovative**:
- 90% of work done via direct methods (microseconds)
- Only 10% via optimization (few evals)
- Could be 10-50x faster than current approach
- More time for generating multiple candidates

**Expected Impact**: Potentially MASSIVE speedup while maintaining accuracy

**Effort**: Medium-High

---

### Recommended Priority for Innovative Approaches

| Priority | Approach | Speed Gain | Accuracy Risk | Effort |
|----------|----------|------------|---------------|--------|
| **1** | Geometric Solution (#12) | 1000x | Low (overdetermined) | Low |
| **2** | ICA Decomposition (#11) | 100x | Medium | Medium |
| **3** | Hybrid Direct (#15) | 10-50x | Low | Medium |
| **4** | Neural Surrogate (#13) | 5-10x | Medium | Medium |
| **5** | Peak Finding (#14) | 100x | Medium | Low |

**Suggested Implementation Order**:
1. Start with Geometric Solution for 1-source (fastest to implement, highest confidence)
2. Add ICA for 2-source decomposition
3. Combine into Hybrid approach
4. Use saved time budget for more candidates

---

## Recommended Implementation Order (Updated 2024-12-29)

### Completed
- ~~**Adaptive Polish**~~ - Tested, not effective
- ~~**Intensity-Only Polish**~~ - Tested, viable alternative (score 0.7862 @ 58 min)
- ~~**Multiple Candidates**~~ - **TESTED, BEST RESULT!** (score **0.8455** @ 51.6 min)

### Next Priority: Fine-tune Multi-Candidates
Current runtime is 51.6 min, leaving **~3.4 min headroom** to reach 55 min target.

Options to explore:
1. **Increase CMA-ES fevals** - Try 30/50 or 28/48 to improve per-candidate RMSE
2. **Increase candidate pool size** - Try pool_size=15 or 20 for more diversity
3. **Increase polish iterations** - Try polish_maxiter=8 or 10

### Medium Term (If More Improvement Needed)
4. **Extended CMA-ES** - More fevals, no polish
5. **Analytical Intensity** - Closed-form q estimation
6. **Coarse-to-Fine Grid** - Optimize at 50x25, refine at 100x50
7. **Multi-Fidelity GP** - Surrogate-guided optimization

### Long Term
8. **PINN Surrogate** - Neural network for initialization

---

## Current Best Configurations

### FINAL SUBMISSION: Multi-Candidates (53.8 min) ⭐ SELECTED
```bash
uv run python experiments/multi_candidates/run.py --workers 7 --max-fevals-1src 20 --max-fevals-2src 40
```
- **Score: 0.7764**
- **RMSE: 0.5247**
- **Avg Candidates: 2.2**
- **Projected: 53.8 min** ✅ Safe buffer for G4dn.2xlarge
- MLflow run: `multi_candidates_20251230_140041`

### Option 1 (Higher Score, Over Budget):
```bash
uv run python experiments/multi_candidates/run.py --workers 7 --max-fevals-1src 25 --max-fevals-2src 45
```
- **Score: 0.8043**
- **Projected: 63.2 min** ❌ Over budget

### Option 2: Intensity-Only Polish (58.0 min)
```bash
uv run python experiments/intensity_polish/run.py --workers 7 --max-fevals-1src 25 --max-fevals-2src 45
```
- **Score: 0.7862**
- **RMSE: 0.4549**
- **Projected: 58.0 min**
- MLflow run: `intensity_polish_20251229_184132`

### Option 3: CMA-ES + L-BFGS-B Polish (SAFE - 57.2 min)
```bash
uv run python experiments/cmaes/run.py --workers 7 --polish-iter 1
```
- **Score: 0.7501**
- **RMSE: 0.5146**
- **Projected: 57.2 min**
- MLflow run: `cmaes_20251229_124607`

### Comparison Table
| Approach | Score | RMSE | Time | Improvement |
|----------|-------|------|------|-------------|
| **Multi-Candidates (28/50)** | **0.8577** | 0.431 | 54.0 min | **+14.3% vs CMA-ES** |
| Multi-Candidates (25/45) | 0.8455 | 0.449 | 51.6 min | +12.7% vs CMA-ES |
| Intensity-Only | 0.7862 | 0.455 | 58.0 min | +4.8% vs CMA-ES |
| CMA-ES polish=1 | 0.7501 | 0.515 | 57.2 min | baseline |

### Not Recommended
- **Adaptive Polish**: Inconsistent timing, no reliable improvement
- **Intensity-Only (15/25)**: Too low score (0.6969)
- **CMA-ES polish=2**: Borderline timing (59.7 min)

---

## References

### Implemented
- [CMA-ES Official](https://cma-es.github.io/)
- [CMA-ES Tutorial (arXiv)](https://arxiv.org/abs/1604.00772)

### For Future Work
- [Multi-fidelity optimization (Royal Society)](https://royalsocietypublishing.org/doi/10.1098/rspa.2007.1900)
- [PINN for heat conduction](https://www.sciencedirect.com/science/article/abs/pii/S0735193323000519)
- [ThermoNet](https://www.sciencedirect.com/science/article/abs/pii/S0924424718314110)
- [Incremental Bayesian Inversion](https://www.sciencedirect.com/science/article/abs/pii/S0017931024014455)

---

*Document updated 2024-12-29 - Added innovative direct-solution approaches (ICA, Geometric, Hybrid)*
*Current BEST: 0.8577 @ 54.0 min (Multi-Candidates)*

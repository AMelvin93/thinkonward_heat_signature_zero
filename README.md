# Heat Signature Zero

**ThinkOnward Competition Entry** | Thermal Source Identification via Simulation-Driven Optimization

**Best Validated Score: 1.152** | Projected Runtime: ~58 min | Score Journey: 0.57 &rarr; 1.155 (+103%)

---

## Competition Overview

[Heat Signature Zero](https://thinkonward.com/app/c/challenges/heat-signature-zero) is a ThinkOnward challenge that tasks participants with identifying hidden heat sources from noisy thermal sensor readings using inverse optimization. The key constraint: **solutions must actively use the thermal simulator during inference** &mdash; no pre-computed lookup tables or brute-force grid searches allowed.

### Problem Statement

Given a 2D thermal domain (L<sub>x</sub>=2.0, L<sub>y</sub>=1.0) discretized on a 100&times;50 grid, sensors record temperature readings over time as unknown heat sources warm the domain. The task is to recover the positions (x, y) and intensities (q &isin; [0.5, 2.0]) of 1&ndash;2 hidden point sources.

- **Test set**: 80 samples (40% single-source, 60% dual-source)
- **Competition set**: 400 samples, must complete in <60 minutes on G4dn.2xlarge (8 vCPUs)
- **Solver**: ADI (Alternating Direction Implicit) method for the 2D heat equation
- **Sensors**: Sparse spatial measurements sampled at discrete timesteps

### Scoring Formula

Each sample produces up to N<sub>max</sub> = 3 candidate solutions, scored by:

```
P = (1/N_valid) * sum(1/(1 + RMSE_i)) + 0.3 * (N_valid / 3)
```

| Component | Max Value | Description |
|-----------|-----------|-------------|
| Accuracy  | 1.0       | Rewards low forward-model RMSE |
| Diversity | 0.3       | Rewards multiple distinct candidates (tau=0.2 dissimilarity threshold) |
| **Total** | **1.3**   | Theoretical maximum score |

The final score is the average P across all samples.

### Evaluation Criteria

| Weight | Criterion | Description |
|--------|-----------|-------------|
| 70%    | Performance | Score on unseen holdout dataset |
| 20%    | Innovation  | Active simulator use, smart optimization, adaptive refinement |
| 10%    | Interpretability | Clear Jupyter notebook explaining the pipeline |

---

## Final Results

### Score Progression

```
Start (Dec 2025):   0.57   HybridOptimizer (JAX + L-BFGS-B)
CMA-ES baseline:    0.75   CMA-ES + L-BFGS-B polish
Triangulation:      0.78   Physics-based initialization
Multi-candidate:    0.87   Multiple candidate generation
Analytical q:       1.00   Analytical intensity (closed-form)
Early timesteps:    1.08   30% early fraction weighting
Multi-fidelity:     1.12   Coarse (50x25) exploration + fine eval
Tabu hopping:       1.13   Basin hopping with tabu memory
Parameter tuning:   1.15   Exhaustive sigma/fevals/perturbation sweep
```

**Total improvement: +103% from initial baseline to final score.**

### Final Production Performance

| Metric | 1-Source | 2-Source | Overall |
|--------|---------|---------|---------|
| RMSE   | 0.117   | 0.184   | 0.157   |
| Score component | ~0.90 | ~0.84 | 1.152 |

- **Validated mean score**: 1.152 &plusmn; 0.001
- **Best single run**: 1.158 (matches leaderboard Top 10)
- **Projected runtime**: 55&ndash;59 min for 400 samples

### Leaderboard Context

```
#1   Jonas M       1.2268
#2   kjc           1.2265
...
#10  MGoksu        1.1585
---- Our Best Run  1.1585  (matches #10!)
---- Our Mean      1.1520
#11  bobatea       1.1295
```

---

## Solution Architecture

The optimizer follows the competition-mandated pattern of **guess &rarr; simulate &rarr; compare &rarr; refine**:

```
For each sample:
  1. INITIALIZE    Physics-based triangulation + hottest-sensor + transfer learning
  2. EXPLORE       CMA-ES on coarse grid (50x25) with position-only search
  3. EVALUATE      Re-simulate top candidates on fine grid (100x50)
  4. REFINE        Nelder-Mead polish on fine grid
  5. PERTURB       Tabu basin hopping: perturb + re-optimize for diversity
  6. FILTER        Dissimilarity filtering to maximize diversity bonus
  7. SCORE         Analytical intensity + RMSE computation
```

### Key Innovations

#### 1. Analytical Intensity Computation
The heat equation is **linear in source intensity q**: `T(x,t) = q * T_unit(x,t)`. This means for any candidate position (x, y), the optimal intensity has a closed-form least-squares solution:

```
q_optimal = (Y_unit . Y_observed) / (Y_unit . Y_unit)
```

For 2-source problems, this becomes a 2&times;2 linear system. This reduces the search space from 3D per source to **2D (position only)**, dramatically improving CMA-ES convergence.

#### 2. Physics-Based Triangulation Initialization
Instead of random initialization, the optimizer uses heat diffusion physics to estimate initial source positions:

- **Onset detection**: Identifies when each sensor first detects the heat front
- **Distance estimation**: Uses the diffusion scaling law `r ~ sqrt(4 * kappa * t)` to estimate source-sensor distances
- **Trilateration**: Solves for source position from multiple distance estimates

This provides a warm start that is typically within 0.2 units of the true source.

#### 3. Multi-Fidelity Optimization
CMA-ES exploration runs on a **coarse 50&times;25 grid** (~4x faster per simulation), while final candidate evaluation uses the full **100&times;50 grid**. This enables more function evaluations within the time budget without sacrificing final accuracy.

#### 4. Early Timestep Weighting
The optimizer uses only the first 40% of timesteps for the position objective function. Early timesteps contain **onset timing information** that is more discriminative for source localization, particularly for breaking 2-source symmetry.

#### 5. Tabu Basin Hopping
After finding initial candidates, the optimizer applies perturbation-based local search with **tabu memory** to avoid revisiting previously explored regions. Each perturbation is refined with Nelder-Mead, and the tabu list prevents wasting evaluations on already-discovered basins.

#### 6. Robust Fallback Strategy
If CMA-ES converges to a poor solution (RMSE above threshold), the optimizer automatically **restarts with a different initialization** strategy. Thresholds are tuned separately for 1-source (0.35) and 2-source (0.45) problems.

#### 7. Transfer Learning
For each new sample, the optimizer checks previously solved samples for **feature-based similarity** (using 11 features including peak temperatures, spatial moments, and temporal statistics). If a similar sample is found, its solution serves as an additional initialization, bootstrapping the search.

---

## Physics & Mathematical Foundation

### Heat Equation

The 2D heat equation with point sources:

```
dT/dt = kappa * (d2T/dx2 + d2T/dy2) + sum_i(q_i * delta(x - x_i, y - y_i))
```

Solved numerically using the **ADI (Alternating Direction Implicit)** method, which splits each timestep into two half-steps (one implicit in x, one implicit in y), maintaining unconditional stability.

### Linearity Exploitation

Since the heat equation is linear in q:
- **1-source**: `q = argmin ||q * T_unit(x,y) - T_obs||^2` has closed-form solution
- **2-source**: Joint intensity optimization solves a 2&times;2 linear system:
  ```
  [T1.T1  T1.T2] [q1]   [T1.Tobs]
  [T2.T1  T2.T2] [q2] = [T2.Tobs]
  ```

### Diffusion Scaling

The fundamental solution to the heat equation in 2D: `T(r,t) ~ exp(-r^2 / (4*kappa*t)) / t`

This gives the key relationship for triangulation:
```
r ~ sqrt(4 * kappa * t_onset)
```

where `t_onset` is the time a sensor first detects the heat signal above noise.

---

## Optimization Pipeline (Detailed)

### Step 1: Initialization (3 strategies)

| Strategy | How it Works | Usage |
|----------|-------------|-------|
| **Triangulation** | Onset times &rarr; distances &rarr; trilateration | 35% of best inits |
| **Hottest Sensor** | Peak temperature location + random offset | 51% of best inits |
| **Transfer** | Solution from most similar past sample | 14% of best inits |

### Step 2: CMA-ES on Coarse Grid

- **Population size**: Auto-scaled by CMA-ES (typically 5&ndash;7 for 2D, 9&ndash;11 for 4D)
- **Function evaluations**: 20 (1-source), 44 (2-source)
- **Sigma**: 0.18 (1-source), 0.22 (2-source)
- **Grid**: 50&times;25 coarse grid (~4x speedup)
- **Search space**: Position only (intensity computed analytically)

### Step 3: Fine-Grid Evaluation

Top 2 candidates are re-simulated on the full 100&times;50 grid to get accurate RMSE values.

### Step 4: Nelder-Mead Refinement

Up to 8 iterations of derivative-free Nelder-Mead simplex optimization on the fine grid. For 2D (1-source) or 4D (2-source), Nelder-Mead requires only n+1 evaluations per iteration vs. 2n+1 for finite-difference gradients.

### Step 5: Tabu Basin Hopping

- **Perturbations**: 2&ndash;4 per sample (budget-dependent)
- **Scale**: 0.05 (in normalized coordinates)
- **Tabu distance**: 0.04 (minimum distance from known solutions)
- **Each perturbation**: Generates a new candidate, refined with 2&ndash;3 Nelder-Mead iterations

### Step 6: Candidate Filtering

Dissimilarity filter ensures candidates are at least `tau = 0.2` apart (in normalized parameter space with scale factors [2.0, 1.0, 2.0] for [x, y, q]). Up to 3 candidates are kept, maximizing the diversity bonus.

---

## Project Structure

```
heat-signature-zero/
├── README.md                  # This file
├── CLAUDE.md                  # AI assistant instructions
├── pyproject.toml             # Dependencies (uv)
├── main.py                    # Entry point
│
├── src/                       # Production code
│   ├── OPTIMIZER_HISTORY.md   # Complete optimizer evolution log
│   ├── triangulation.py       # Physics-based initialization
│   ├── scoring.py             # Competition scoring functions
│   ├── seed_manager.py        # Reproducibility utilities
│   ├── hybrid_optimizer.py    # NumPy + L-BFGS-B optimizer
│   ├── optimizer.py           # Baseline L-BFGS-B
│   ├── seeded_cmaes.py        # CMA-ES with seed support
│   ├── visualize.py           # Plotting utilities
│   ├── jax_*.py               # JAX experiments (not used in production)
│   ├── adjoint_optimizer*.py  # Adjoint method experiments
│   └── tabu_*.py              # Tabu search experiments
│
├── experiments/               # 250+ experimental configurations
│   ├── robust_fallback/       # Final production optimizer
│   ├── multi_fidelity/        # Multi-fidelity breakthrough
│   ├── analytical_intensity/  # Analytical intensity innovation
│   ├── ica_decomposition/     # ICA signal decomposition (tested)
│   ├── transfer_learning/     # Transfer learning + CMA-ES
│   ├── cmaes/                 # CMA-ES + L-BFGS-B polish
│   ├── ... (250+ experiments) # See Experiment History below
│   └── SESSION_SUMMARY_*.md   # Session reports
│
├── scripts/                   # Utility scripts
│   ├── run_experiment.py      # Run any experiment
│   ├── run_final_submission.py # Final submission runner
│   ├── evaluate.py            # Evaluate predictions
│   ├── make_submission.py     # Create submission file
│   ├── create_submission.py   # Submission formatting
│   ├── calculate_score*.py    # Score calculation
│   └── test_*.py              # Benchmarks and tests
│
├── configs/                   # Configuration files
│   └── final_submission.yaml  # Production configuration
│
├── notebooks/                 # Jupyter notebooks
│   ├── 01_optimizer_demo.ipynb
│   ├── 02_jax_benchmark.ipynb
│   └── 03_interactive_visualization.ipynb
│
├── data/                      # Competition data
│   └── heat-signature-zero-test-data.pkl
│
├── docs/                      # Documentation
│   ├── RESEARCH_NEXT_STEPS.md # Research notes & leaderboard analysis
│   ├── SEEDING.md             # Reproducibility guide
│   └── future_optimizations.md
│
├── submissions/               # Generated submissions
│   └── final5/               # Top 5 submission candidates
│
├── mlruns/                    # MLflow experiment tracking
├── model/                     # Model artifacts
├── outputs/                   # Run outputs
├── results/                   # Result files
└── orchestration/             # Multi-worker experiment orchestration
```

---

## Experiment History & Evolution

This project explored **250+ experimental configurations** across dozens of algorithmic families over 6 weeks. Below are the major breakthroughs and categories.

### Major Breakthroughs (Chronological)

| Session | Innovation | Score Impact | Key Insight |
|---------|-----------|-------------|-------------|
| 1&ndash;3 | CMA-ES replaces L-BFGS-B | 0.57 &rarr; 0.75 | CMA-ES handles 2-source permutation symmetry |
| 4 | Triangulation initialization | +3.1% | Physics-based warm start from onset times |
| 5 | Multi-candidate generation | +11.7% | Diversity bonus from filtered candidate sets |
| 6 | Analytical intensity | +14.8% | Closed-form q reduces search to position-only |
| 9 | Early timestep weighting | +8.5% | Onset timing is most discriminative for localization |
| 12 | Multi-fidelity grids | +1.2% | Coarse exploration + fine evaluation |
| 14 | Coarse refinement (NM) | +1.3% | Nelder-Mead polish on coarse grid |
| 15&ndash;17 | Robust fallback | +2.1% | Auto-restart on poor CMA-ES convergence |
| 18+ | Tabu basin hopping | +1.4% | Perturbation search with tabu memory |
| Final | Exhaustive parameter tuning | +1.4% | Sigma, fevals, perturbation count optimization |

### Approaches Explored

#### Evolutionary Strategies
- **CMA-ES** (production) &mdash; Core optimizer, handles multimodal landscape
- **IPOP-CMA-ES** &mdash; Restart with increasing population; marginal improvement, slower
- **BIPOP-CMA-ES** &mdash; Bidirectional population restart; no improvement
- **Separable CMA-ES** &mdash; Diagonal covariance; worse for correlated parameters
- **Differential Evolution** &mdash; Tested; CMA-ES converges faster

#### Gradient-Based Optimization
- **L-BFGS-B** (early production) &mdash; Good accuracy but finite-diff gradients expensive for 4D
- **Adjoint Method** &mdash; Exact gradients validated (0.01% error), but overall time dominated by iteration count
- **Conjugate Gradient** &mdash; Similar to L-BFGS-B performance
- **SLSQP** &mdash; Sequential least squares; no benefit over L-BFGS-B

#### Derivative-Free Local Search
- **Nelder-Mead** (production polish) &mdash; Efficient for 2&ndash;4D; n+1 evals/iteration
- **Powell's method** &mdash; Coordinate-wise line search; similar to Nelder-Mead
- **COBYLA** &mdash; Constrained optimization; no benefit for this problem
- **Coordinate Descent** &mdash; Sequential 1D optimization; marginal improvement
- **BOBYQA** &mdash; Quadratic model-based; comparable to Nelder-Mead

#### Meta-Heuristics
- **Basin Hopping** (production) &mdash; Global perturbation + local refinement
- **Tabu Search** (production memory) &mdash; Avoids revisiting explored regions
- **Simulated Annealing** &mdash; Tested; CMA-ES more sample-efficient
- **Particle Swarm (PSO)** &mdash; Population-based; slower convergence than CMA-ES
- **Genetic Algorithm** &mdash; Tested; CMA-ES dominates for continuous optimization

#### Signal Processing
- **ICA Decomposition** &mdash; Separate 2-source signals via FastICA; **best score ever (1.042)** but 27 min over budget
- **Frequency Domain** &mdash; FFT-based features; no benefit for heat equation
- **PCA/POD** &mdash; Dimensionality reduction; marginal initialization help

#### Machine Learning & Surrogate Models
- **Bayesian Optimization (GP)** &mdash; Gaussian process surrogate; GP overhead doesn't pay off (-2.7%)
- **Neural Network Surrogate** &mdash; Trained approximation; accuracy insufficient
- **Kriging Infill** &mdash; RBF-based local surrogate; similar to BO results
- **PINN** &mdash; Physics-informed neural net; training too slow for inference-time use

#### Multi-Fidelity Approaches
- **Coarse-to-Fine** (production) &mdash; 50&times;25 exploration + 100&times;50 evaluation
- **Ultra-Coarse 40&times;20** &mdash; Too much accuracy loss at this resolution
- **Temporal Subsampling** &mdash; Every-other-timestep; marginal speedup, accuracy loss

#### Initialization Strategies
- **Triangulation** (production) &mdash; Physics-based onset time analysis
- **Hottest Sensor** (production) &mdash; Simple but effective baseline
- **Transfer Learning** (production) &mdash; Feature-similarity matching
- **Gradient-based** &mdash; RBF interpolation of temperature gradients; noisy
- **K-means Clustering** &mdash; Cluster sensor readings; marginal benefit
- **Center-of-Mass** &mdash; Weighted centroid; simple baseline

#### JAX/GPU Acceleration
- **JAX Forward Simulator** &mdash; GPU port of ADI solver
- **JAX Batched Optimizer** &mdash; Vectorized via vmap
- **JAX Pure Adam** &mdash; All-JAX optimization loop
- **Finding**: GPU kernel launch overhead exceeds computation benefit for 100&times;50 grids. CPU parallelism across samples (7 workers) is more effective.

### What Didn't Work (Lessons Learned)

| Approach | Why It Failed |
|----------|---------------|
| JAX/GPU acceleration | Kernel launch overhead > computation for small grids |
| Adjoint gradients | Exact but doesn't reduce iteration count; sample parallelism wins |
| Sequential 2-source decomposition | Compounding errors from independent optimization |
| Bayesian optimization | GP surrogate overhead doesn't pay off in low dimensions |
| ICA decomposition | Best accuracy ever but 27 min over time budget |
| More CMA-ES restarts | Dilutes function evaluations per restart |
| 5+ perturbations | Improves score but adds irreducible ~24 min overhead |
| Sensor subset diversity | Dissimilarity filter rejects similar candidates anyway |

---

## Getting Started

### Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended package manager)
- Linux/WSL recommended for accurate timing (Windows is ~35% slower)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd thinkonward_heat_signature_zero

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

### Quick Start

```bash
# Run the production optimizer on test data (80 samples)
uv run python scripts/run_final_submission.py

# Run a specific experiment
uv run python scripts/run_experiment.py robust_fallback

# Evaluate predictions against ground truth
uv run python scripts/evaluate.py

# Create a submission file
uv run python scripts/make_submission.py

# View MLflow experiment tracking
uv run mlflow ui
```

### Running on WSL (Recommended for Timing)

```bash
# All timing-critical runs should use WSL for accurate projection
cd /mnt/c/Users/amelv/Repo/thinkonward_heat_signature_zero
uv run python scripts/run_final_submission.py
```

---

## Key Commands Reference

| Command | Description |
|---------|-------------|
| `uv run python scripts/run_final_submission.py` | Run production optimizer on full test set |
| `uv run python scripts/run_experiment.py <name>` | Run a specific experiment |
| `uv run python scripts/evaluate.py` | Score predictions vs ground truth |
| `uv run python scripts/make_submission.py` | Generate submission `.npz` file |
| `uv run python scripts/calculate_score.py` | Quick score calculation |
| `uv run python scripts/test_reproducibility.py` | Verify seed determinism |
| `uv run mlflow ui` | Launch MLflow tracking UI |

---

## Reproducibility

All experiments use deterministic seeding via `src/seed_manager.py`:

```python
from src.seed_manager import SeedManager

seed_manager = SeedManager(master_seed=42)

# Per-sample seeds for parallel workers
sample_seed = seed_manager.get_sample_seed(sample_idx)

# Per-CMA-ES-run seeds
cmaes_seed = seed_manager.get_cmaes_seed(sample_idx, init_idx)
```

### Seeding Protocol

1. **Master seed** (default: 42) passed via `--seed` argument
2. **Per-sample seeds** derived deterministically from master seed
3. **Worker seeding**: Each parallel worker seeds `np.random.seed(sample_seed)` at start
4. **CMA-ES seeding**: Passed to `cma.CMAOptions()['seed']`

### Platform Notes

- **WSL/Linux**: Reference platform for timing benchmarks
- **Windows**: ~35% slower due to process creation overhead and NumPy differences
- **G4dn.2xlarge**: Competition target (8 vCPUs, 32GB RAM, Linux)
- Use `n_workers=7` for submission validation (8 vCPUs - 1 for system)

---

## Production Configuration

The final submission uses `configs/final_submission.yaml`:

```yaml
optimizer:
  type: robust_fallback
  sigma0_1src: 0.18           # CMA-ES initial step size (1-source)
  sigma0_2src: 0.22           # CMA-ES initial step size (2-source)
  max_fevals_1src: 20         # Function evaluations (1-source)
  max_fevals_2src: 44         # Function evaluations (2-source)
  threshold_1src: 0.35        # Fallback RMSE threshold (1-source)
  threshold_2src: 0.45        # Fallback RMSE threshold (2-source)
  nx_coarse: 50               # Coarse grid for CMA-ES
  ny_coarse: 25
  refine_maxiter: 8           # Nelder-Mead iterations
  refine_top_n: 2             # Refine top N candidates
  early_fraction: 0.40        # Early timestep fraction
  use_triangulation: true     # Physics-based initialization
  enable_tabu_hopping: true   # Basin hopping with tabu memory
  n_perturbations: 4          # Perturbations per sample
  perturb_nm_iters: 2         # NM iterations per perturbation
  perturbation_scale: 0.05    # Perturbation magnitude
  tabu_distance: 0.04         # Minimum distance from known solutions

scoring:
  lambda_: 0.3                # Diversity weight
  tau: 0.2                    # Dissimilarity threshold
  n_max: 3                    # Maximum candidates per sample
```

### Parameter Tuning Summary

Every parameter above was systematically tuned through multi-run validation:

| Parameter | Range Tested | Optimal | Method |
|-----------|-------------|---------|--------|
| sigma0_1src | 0.12&ndash;0.20 | 0.18 | Grid sweep + 3-run validation |
| sigma0_2src | 0.18&ndash;0.25 | 0.22 | Grid sweep + 3-run validation |
| max_fevals_1src | 14&ndash;24 | 20 | Time-accuracy tradeoff |
| max_fevals_2src | 32&ndash;48 | 44 | Time-accuracy tradeoff |
| n_perturbations | 1&ndash;5 | 4 | Score vs budget constraint |
| perturbation_scale | 0.04&ndash;0.07 | 0.05 | Grid sweep |
| timestep_fraction | 0.30&ndash;0.45 | 0.40 | Grid sweep |
| refine_maxiter | 4&ndash;10 | 8 | Diminishing returns analysis |

---

## Tech Stack

### Core Dependencies

| Package | Role |
|---------|------|
| **NumPy** | Numerical computation, array operations |
| **SciPy** | Nelder-Mead, L-BFGS-B, optimization utilities |
| **CMA** (cma) | CMA-ES evolutionary strategy |
| **joblib** | Sample-level parallelization |
| **MLflow** | Experiment tracking and comparison |
| **PyYAML** | Configuration management |
| **Matplotlib/Plotly** | Visualization |

### Experimental (tested, not in production)

| Package | Role | Finding |
|---------|------|---------|
| **JAX** | GPU-accelerated simulation | Kernel overhead > benefit for small grids |
| **scikit-optimize** | Bayesian optimization | GP surrogate too expensive for low-D |

### Infrastructure

- **uv**: Fast Python package manager
- **Docker**: Containerized submission environment
- **WSL**: Linux environment for accurate timing benchmarks

---

## Competition Information

- **Competition**: [ThinkOnward Heat Signature Zero](https://thinkonward.com/app/c/challenges/heat-signature-zero)
- **Deadline**: 23:00 UTC, 4 February 2026
- **Winners announced**: 19 February 2026
- **Prizes**: 1st: $10,000 | 2nd: $6,000 | 3rd: $3,500 | Honorable mention: $500
- **Target hardware**: G4dn.2xlarge (8 vCPUs, 32GB RAM, Linux)
- **Time limit**: 400 samples in <60 minutes

### Finalist Process

1. Top scorers submit reproducible Python code
2. 5 days to set up in remote Linux workspace
3. Code runs on unseen holdout data
4. Must complete in <1 hour on G4dn.2xlarge

---

## License

Open-source software license (as required by competition rules).

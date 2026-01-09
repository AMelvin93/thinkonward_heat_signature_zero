# Adaptive Learning Tabu Search

This experiment implements a Tabu Search optimizer that combines multiple **learning mechanisms** to adapt during inference. Unlike grid search or pre-computed strategies, this optimizer learns from each simulator call.

## Philosophy

The competition requires solutions that **learn by inference** rather than brute-force approaches. This optimizer embodies that philosophy through five learning mechanisms:

## Learning Mechanisms

### 1. Observational Learning (Smart Initialization)

**What it learns**: Likely source locations from sensor temperature patterns.

**How**: Analyzes observed temperatures at sensor locations:
- Identifies hottest sensors as likely source proximity indicators
- For multi-source problems, selects well-separated hot regions
- Initializes search near these inferred locations

**Why aligned**: Uses the observed data (not pre-computed) to inform starting points.

### 2. Gradient Learning (Descent Direction)

**What it learns**: Local descent direction from recent objective evaluations.

**How**:
- Maintains a buffer of recent (params, cost) evaluations
- Estimates gradient via weighted least squares
- Biases neighbor generation toward descent direction
- Balances gradient-informed (exploitation) with random (exploration) moves

**Why aligned**: Learns the objective landscape shape from its own simulator calls.

### 3. Landscape Learning (Adaptive Step Size)

**What it learns**: Appropriate step size from objective function roughness.

**How**:
- Monitors variance of recent costs (flat vs rough landscape)
- Tracks improvement trends (converging vs stuck)
- Adapts step size:
  - Flat region (low variance) → larger steps to escape plateau
  - Improving → smaller steps for precision
  - Stuck → larger steps for exploration

**Why aligned**: The step size adapts to what the search discovers about the landscape.

### 4. Curvature Learning (L-BFGS-B Polish)

**What it learns**: Local curvature (Hessian approximation) for efficient refinement.

**How**:
- After Tabu search finds a promising basin, applies L-BFGS-B
- L-BFGS-B approximates the Hessian from function evaluations
- Uses this curvature information for efficient local convergence

**Why aligned**: L-BFGS-B learns the curvature from evaluations, not pre-computed.

### 5. Search Dynamics Learning (Adaptive Tabu Tenure)

**What it learns**: Appropriate memory length from search progress.

**How**:
- Monitors iterations of improvement vs being stuck
- Adapts tabu tenure:
  - Improving → shorter tenure (exploit current region)
  - Stuck → longer tenure (force exploration of new regions)can 

**Why aligned**: The memory structure adapts to the search dynamics.

## Running the Experiment

```bash
# Full run with MLflow tracking
python scripts/run_experiment.py --experiment adaptive_learning

# Quick test with fewer samples
python scripts/run_experiment.py --experiment adaptive_learning --n-samples 10

# Direct test (no MLflow)
python experiments/adaptive_learning/train.py
```

## Configuration

Key parameters in `config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tabu.max_iterations` | 25 | Iterations per restart |
| `tabu.n_restarts` | 3 | Independent search restarts |
| `gradient.buffer_size` | 20 | Evaluations for gradient estimation |
| `gradient.exploitation_ratio` | 0.5 | Gradient vs exploratory moves |
| `adaptive.enable_polish` | true | Enable L-BFGS-B refinement |

## Expected Improvements

Compared to baseline Tabu Search:

| Metric | Baseline Tabu | Expected Adaptive |
|--------|---------------|-------------------|
| RMSE (1-source) | ~0.12 | ~0.08 |
| RMSE (2-source) | ~0.18 | ~0.12 |
| Efficiency | ~540 calls | ~400 calls |

## MLflow Metrics

The experiment logs these learning-specific metrics:

- `avg_gradient_usage`: Fraction of moves using gradient info
- `avg_polish_improvement`: RMSE reduction from L-BFGS-B polish
- `total_tenure_adaptations`: Count of tabu tenure adjustments
- `rmse_{n}src_mean/std`: Performance breakdown by source count

## Files

- `train.py` - Main experiment runner
- `config.yaml` - Hyperparameters
- `../../src/adaptive_learning_optimizer.py` - Core optimizer implementation

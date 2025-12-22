# tabu_gradient

Gradient-Informed Tabu Search for heat source identification.

## Approach

This variant enhances standard Tabu Search by **learning gradient information from its own evaluations** during inference.

### Learning Mechanism

```
┌─────────────────────────────────────────────────────────────┐
│              GRADIENT-INFORMED SEARCH LOOP                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Evaluate Neighbors ──► Store in Buffer ──► Learn Gradient│
│         ▲                                          │        │
│         │                                          ▼        │
│         │                          Generate New Neighbors   │
│         │                          (60% gradient-informed)  │
│         │                          (40% exploratory)        │
│         │                                          │        │
│         └──────────────────────────────────────────┘        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Evaluation Buffer**: Stores recent (params, cost) pairs
2. **Gradient Estimation**: Weighted least squares from buffer
3. **Informed Neighbors**: Biased toward descent direction
4. **Exploration Balance**: Mix of gradient and random moves

### Why This Learns During Inference

| Traditional Tabu | Gradient-Informed Tabu |
|------------------|------------------------|
| Random neighborhood | Neighborhood adapts to learned gradient |
| No memory of costs | Learns from cost landscape |
| Fixed exploration | Balances exploit/explore based on info |

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `buffer_size` | 15 | Recent evaluations for gradient |
| `exploitation_ratio` | 0.6 | Fraction using gradient info |
| `min_samples` | 5 | Evals before gradient kicks in |

## Usage

```bash
# Run experiment
python scripts/run_experiment.py --experiment tabu_gradient

# Quick test
python scripts/run_experiment.py --experiment tabu_gradient --n-samples 2

# Compare with baseline
python scripts/run_experiment.py --experiment tabu_search --n-samples 5
python scripts/run_experiment.py --experiment tabu_gradient --n-samples 5
```

## Expected Behavior

- **Early iterations**: Random exploration (not enough data)
- **After min_samples**: Gradient-informed moves begin
- **Gradient usage**: ~60% of moves use learned gradient
- **Better convergence**: Fewer wasted evaluations

## Metrics

| Metric | Description |
|--------|-------------|
| `rmse_mean` | Average RMSE across samples |
| `avg_gradient_usage` | Fraction of moves using gradient |
| `avg_candidates` | Diverse solutions found |

## Results

| Metric | Value |
|--------|-------|
| RMSE (mean) | TBD |
| RMSE (std) | TBD |
| Gradient usage | TBD |

## Notes

- Created: 2024-12-20
- This is the first "learning" optimization
- Compare against `tabu_search` baseline to measure improvement

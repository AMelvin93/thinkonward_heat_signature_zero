# Seeding System for Reproducibility

This document describes the seeding system implemented to ensure reproducible experiment results.

## Problem

The original codebase had reproducibility issues due to:

1. **Global `np.random.seed()` doesn't propagate to worker processes** - Each `ProcessPoolExecutor` worker spawns a separate Python process with its own independent random state
2. **CMA-ES was not seeded** - The `cma` library uses NumPy's RNG but no explicit seed was passed
3. **Seeds were not stored** - STATE.json didn't record the seed used for each run, making reproduction impossible
4. **Per-sample randomness was unpredictable** - Same sample could produce different results across runs

## Solution

### Core Components

1. **`src/seed_manager.py`** - SeedManager class that generates deterministic per-sample seeds
2. **`src/seeded_cmaes.py`** - Helper functions for creating seeded CMA-ES instances

### How It Works

```
Master Seed (e.g., 42)
        │
        ├──▶ SeedManager.get_sample_seed(0) ──▶ Sample 0 seed (deterministic)
        ├──▶ SeedManager.get_sample_seed(1) ──▶ Sample 1 seed (deterministic)
        ├──▶ SeedManager.get_sample_seed(2) ──▶ Sample 2 seed (deterministic)
        └──▶ ...

Each Sample Seed
        │
        ├──▶ np.random.seed(sample_seed)  # Seeds worker's NumPy
        └──▶ derive_cmaes_seed(sample_seed, init_idx) ──▶ CMA-ES seed
```

### Key Principles

1. **Same master_seed always produces same sample_seeds** (via SHA-256 hash)
2. **Each worker is seeded at the start** with its sample-specific seed
3. **CMA-ES gets explicit seed** derived from sample seed + initialization index
4. **Seeds are stored in STATE.json** for full reproducibility

## Usage

### In run.py scripts

```python
from src.seed_manager import SeedManager

def process_single_sample(args):
    idx, sample, meta, config, sample_seed = args

    # CRITICAL: Seed worker at the very start
    np.random.seed(sample_seed)

    # ... rest of processing

def main():
    # Create seed manager
    seed_manager = SeedManager(master_seed=args.seed)

    # Create work items with per-sample seeds
    work_items = [
        (i, samples[i], meta, config, seed_manager.get_sample_seed(i))
        for i in range(n_samples)
    ]

    # ... parallel processing

    # Store seed info in STATE.json
    state['tuning_runs'].append({
        ...
        'seed_info': {
            'master_seed': args.seed,
            'sample_seeds': {r['idx']: r['sample_seed'] for r in results},
        },
    })
```

### In optimizer code

```python
from src.seeded_cmaes import derive_cmaes_seed

def _run_single_optimization(self, ...):
    # Get base seed from numpy's current state (seeded per-sample in worker)
    base_seed = np.random.randint(0, 2**31)

    for init_idx, (init_params, init_type) in enumerate(initializations):
        opts = cma.CMAOptions()
        opts['seed'] = derive_cmaes_seed(base_seed, init_idx)  # Key addition
        # ... rest of CMA-ES setup
```

### Reproducing a Run

1. Find the `seed_info` in STATE.json from the run you want to reproduce
2. Run with the same `--seed` value:
   ```bash
   python run.py --seed 42 --workers 7
   ```
3. Results should be identical (within floating-point precision)

## Files Modified/Created

### New Files
- `src/seed_manager.py` - SeedManager class
- `src/seeded_cmaes.py` - CMA-ES seeding helpers
- `scripts/test_reproducibility.py` - Validation script
- `experiments/seeded_template/run.py` - Template for new experiments

### Updated Files
- `experiments/adaptive_perturbation_scale/run.py` - Added seeding
- `experiments/adaptive_perturbation_scale/optimizer.py` - Added CMA-ES seeding

## Testing Reproducibility

```bash
# Run the reproducibility test
python scripts/test_reproducibility.py --seed 42 --samples 5

# Run twice and compare output
python scripts/test_reproducibility.py --seed 42 --samples 5
# Output should be identical
```

## Limitations

1. **CMA-ES minor non-determinism** - In some dimensions (e.g., 10), `np.linalg.eigh` may have minor non-deterministic behavior. Most results should still be reproducible.

2. **Floating-point accumulation** - Very long optimization runs may accumulate floating-point differences

3. **Different platforms** - Results may differ slightly between Windows/Linux due to different BLAS implementations

## Migration Guide

To update an existing experiment to use seeding:

1. **In run.py:**
   ```python
   # Add import
   from src.seed_manager import SeedManager

   # Update process_single_sample signature
   def process_single_sample(args):
       idx, sample, meta, config, sample_seed = args  # Add sample_seed
       np.random.seed(sample_seed)  # Add this line
       ...

   # In main():
   seed_manager = SeedManager(master_seed=args.seed)

   # Update work_items to include seeds
   work_items = [
       (i, samples[i], meta, config, seed_manager.get_sample_seed(i))
       for i in range(n_samples)
   ]

   # Store seed info in STATE.json
   ```

2. **In optimizer.py:**
   ```python
   # Add import
   from src.seeded_cmaes import derive_cmaes_seed

   # In CMA-ES creation loop:
   base_seed = np.random.randint(0, 2**31)
   for init_idx, (init_params, init_type) in enumerate(initializations):
       opts['seed'] = derive_cmaes_seed(base_seed, init_idx)
       ...
   ```

3. **Update STATE.json schema** to include `seed_info` in tuning runs

## References

- [pycma documentation](https://cma-es.github.io/apidocs-pycma/)
- [NumPy random generator](https://numpy.org/doc/stable/reference/random/generator.html)

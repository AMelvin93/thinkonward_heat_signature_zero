#!/usr/bin/env python
"""
Create a new experiment from template.

Usage:
    python scripts/create_experiment.py my_new_experiment
    python scripts/create_experiment.py neural_surrogate --description "Train neural network surrogate"
"""

import argparse
from pathlib import Path
from datetime import datetime

TRAIN_TEMPLATE = '''#!/usr/bin/env python
"""
{description}

Run with:
    python scripts/run_experiment.py --experiment {name}
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "data" / "Heat_Signature_zero-starter_notebook"))

import pickle
import numpy as np
from src.optimizer import HeatSourceOptimizer
from simulator import Heat2D


def run(config: dict, tracker) -> dict:
    """
    Run the experiment.

    Args:
        config: Configuration dictionary
        tracker: ExperimentTracker instance for logging

    Returns:
        Dictionary with summary metrics
    """
    # Load test data
    data_path = project_root / config["data"]["test_path"]
    with open(data_path, 'rb') as f:
        test_data = pickle.load(f)

    samples = test_data['samples']
    meta = test_data['meta']

    # Limit samples if specified
    n_samples = config.get("n_samples", len(samples))
    samples = samples[:n_samples]

    print(f"Processing {{len(samples)}} samples...")

    # Domain parameters
    Lx = config["domain"]["Lx"]
    Ly = config["domain"]["Ly"]
    nx = config["domain"]["nx"]
    ny = config["domain"]["ny"]

    # Create optimizer
    optimizer = HeatSourceOptimizer(Lx, Ly, nx, ny)

    # Process each sample
    all_predictions = []
    all_rmse = []

    for i, sample in enumerate(samples):
        sample_id = sample['sample_id']
        n_sources = sample['n_sources']

        print(f"  [{{i+1}}/{{len(samples)}}] {{sample_id}} ({{n_sources}} sources)")

        # Run optimization
        estimates, rmse = optimizer.estimate_sources(
            sample, meta,
            q_range=tuple(config["optimizer"]["q_range"]),
            method=config["optimizer"]["method"],
            n_restarts=config["optimizer"]["n_restarts"],
            max_iter=config["optimizer"]["max_iter"],
            parallel=config["optimizer"]["parallel"],
            n_jobs=config["optimizer"]["n_jobs"],
        )

        # Log to tracker
        result = tracker.log_source_estimates(
            sample_id=sample_id,
            estimates=estimates,
            rmse=rmse,
        )
        all_predictions.append(result)
        all_rmse.append(rmse)

    # Log predictions artifact
    tracker.log_predictions(all_predictions)

    # Summary metrics
    results = {{
        "rmse_mean": np.mean(all_rmse),
        "rmse_std": np.std(all_rmse),
        "rmse_median": np.median(all_rmse),
        "n_samples": len(samples),
    }}

    print(f"\\nResults: RMSE = {{results['rmse_mean']:.6f}} +/- {{results['rmse_std']:.6f}}")

    return results


if __name__ == "__main__":
    # Quick test without MLflow
    from tracking import load_config

    class DummyTracker:
        def log_params(self, *args, **kwargs): pass
        def log_metric(self, *args, **kwargs): pass
        def log_predictions(self, *args, **kwargs): pass
        def log_source_estimates(self, sample_id, estimates, rmse, **kwargs):
            return {{"sample_id": sample_id, "estimates": estimates, "rmse": rmse}}

    config = load_config(str(project_root / "configs" / "default.yaml"))
    config["n_samples"] = 2  # Quick test
    run(config, DummyTracker())
'''

CONFIG_TEMPLATE = '''# Experiment: {name}
# {description}
# Created: {date}

# Override default config values here
optimizer:
  method: "L-BFGS-B"
  n_restarts: 4
  max_iter: 50
'''

README_TEMPLATE = '''# {name}

{description}

## Approach

TODO: Describe your approach here.

## Usage

```bash
# Run full experiment
python scripts/run_experiment.py --experiment {name}

# Run on subset for testing
python scripts/run_experiment.py --experiment {name} --n-samples 5

# Run with custom tags
python scripts/run_experiment.py --experiment {name} --tags version=v2 note="increased restarts"
```

## Results

TODO: Document results here.

| Metric | Value |
|--------|-------|
| RMSE (mean) | - |
| RMSE (std) | - |

## Notes

- Created: {date}
'''


def create_experiment(name: str, description: str):
    """Create a new experiment directory with templates."""
    project_root = Path(__file__).parent.parent
    experiment_dir = project_root / "experiments" / name

    if experiment_dir.exists():
        print(f"Error: Experiment '{name}' already exists at {experiment_dir}")
        return False

    # Create directory
    experiment_dir.mkdir(parents=True)

    date = datetime.now().strftime("%Y-%m-%d")

    # Create train.py
    train_content = TRAIN_TEMPLATE.format(name=name, description=description)
    (experiment_dir / "train.py").write_text(train_content)

    # Create config.yaml
    config_content = CONFIG_TEMPLATE.format(name=name, description=description, date=date)
    (experiment_dir / "config.yaml").write_text(config_content)

    # Create README.md
    readme_content = README_TEMPLATE.format(name=name, description=description, date=date)
    (experiment_dir / "README.md").write_text(readme_content)

    print(f"Created experiment: {experiment_dir}")
    print(f"  - train.py")
    print(f"  - config.yaml")
    print(f"  - README.md")
    print(f"\nRun with: python scripts/run_experiment.py --experiment {name}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Create a new experiment")
    parser.add_argument("name", help="Experiment name (will create experiments/<name>/)")
    parser.add_argument("--description", "-d", default="New experiment", help="Short description")
    args = parser.parse_args()

    create_experiment(args.name, args.description)


if __name__ == "__main__":
    main()

"""
MLflow tracking utilities for Heat Signature Zero experiments.

Usage:
    from tracking import ExperimentTracker

    with ExperimentTracker("my_experiment") as tracker:
        tracker.log_params({"method": "L-BFGS-B", "n_restarts": 4})
        tracker.log_metric("rmse", 0.0042)
        tracker.log_artifact("results.csv")
"""

import os
import yaml
import mlflow
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

# Import scoring utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.scoring import (
    compute_simple_score,
    score_from_rmse_and_candidates,
    DEFAULT_LAMBDA,
    DEFAULT_N_MAX,
)


def load_config(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def merge_configs(base: Dict, override: Dict) -> Dict:
    """Recursively merge override config into base config."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


class ExperimentTracker:
    """
    Context manager for MLflow experiment tracking.

    Example:
        with ExperimentTracker("baseline_lbfgs") as tracker:
            tracker.log_params(config)
            # ... run experiment ...
            tracker.log_metric("rmse_1src", rmse)
            tracker.log_predictions(predictions)
    """

    def __init__(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        config_path: str = "configs/default.yaml",
        tags: Optional[Dict[str, str]] = None,
    ):
        self.experiment_name = experiment_name
        self.run_name = run_name or f"{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.config = load_config(config_path)
        self.tags = tags or {}
        self.run = None

        # Set up MLflow
        tracking_uri = self.config.get('mlflow', {}).get('tracking_uri', 'mlruns')
        mlflow.set_tracking_uri(tracking_uri)

    def __enter__(self):
        """Start MLflow run."""
        mlflow.set_experiment(self.config.get('mlflow', {}).get('experiment_name', 'heat-signature-zero'))
        self.run = mlflow.start_run(run_name=self.run_name)

        # Log default tags
        mlflow.set_tags({
            "experiment_type": self.experiment_name,
            **self.tags
        })

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End MLflow run."""
        if exc_type is not None:
            mlflow.set_tag("status", "failed")
            mlflow.set_tag("error", str(exc_val))
        else:
            mlflow.set_tag("status", "completed")
        mlflow.end_run()
        return False

    def log_params(self, params: Dict[str, Any], prefix: str = ""):
        """Log parameters (flattens nested dicts)."""
        flat_params = self._flatten_dict(params, prefix)
        mlflow.log_params(flat_params)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric."""
        mlflow.log_metric(key, value, step=step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log multiple metrics."""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log a file as an artifact."""
        mlflow.log_artifact(local_path, artifact_path)

    def log_figure(self, figure, filename: str):
        """Log a matplotlib or plotly figure."""
        mlflow.log_figure(figure, filename)

    def log_predictions(self, predictions: List[Dict], filename: str = "predictions.yaml"):
        """Log predictions as YAML artifact."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(predictions, f)
            temp_path = f.name
        mlflow.log_artifact(temp_path, "predictions")
        os.unlink(temp_path)

    def log_source_estimates(
        self,
        sample_id: str,
        estimates: List[tuple],
        rmse: float,
        true_sources: Optional[List[Dict]] = None,
        n_candidates: int = 1,
    ):
        """Log source estimation results for a single sample."""
        result = {
            "sample_id": sample_id,
            "estimates": [{"x": e[0], "y": e[1], "q": e[2]} for e in estimates],
            "rmse": rmse,
            "n_candidates": n_candidates,
        }
        if true_sources:
            result["true_sources"] = true_sources
            # Calculate position errors
            for i, (est, true) in enumerate(zip(estimates, true_sources)):
                pos_error = ((est[0] - true['x'])**2 + (est[1] - true['y'])**2)**0.5
                self.log_metric(f"pos_error_{sample_id}_src{i}", pos_error)

        # Calculate sample score
        sample_score = compute_simple_score(rmse, n_candidates, DEFAULT_LAMBDA, DEFAULT_N_MAX)
        result["score"] = sample_score

        self.log_metric(f"rmse_{sample_id}", rmse)
        return result

    def log_final_score(
        self,
        rmse_mean: float,
        avg_candidates: float = 1.0,
        lambda_: float = DEFAULT_LAMBDA,
        n_max: int = DEFAULT_N_MAX,
    ):
        """
        Log the final competition score.

        Score formula: P = (1/(1+L_avg)) + λ * (N_valid/N_max)

        Args:
            rmse_mean: Average RMSE across all samples
            avg_candidates: Average number of candidates per sample
            lambda_: Trade-off weight (default: 0.3)
            n_max: Maximum candidates per sample (default: 3)
        """
        score = score_from_rmse_and_candidates(rmse_mean, avg_candidates, lambda_, n_max)

        # Log score components
        self.log_metric("competition_score", score)
        self.log_metric("score_accuracy_component", 1 / (1 + rmse_mean))
        self.log_metric("score_diversity_component", lambda_ * (min(avg_candidates, n_max) / n_max))

        # Log scoring parameters for reference
        self.log_params({
            "scoring.lambda": lambda_,
            "scoring.n_max": n_max,
        })

        return score

    @property
    def run_id(self) -> str:
        """Get current run ID."""
        return self.run.info.run_id if self.run else None

    def _flatten_dict(self, d: Dict, prefix: str = "") -> Dict[str, Any]:
        """Flatten nested dictionary with dot notation."""
        items = {}
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                items.update(self._flatten_dict(v, key))
            else:
                items[key] = v
        return items


def get_best_run(
    experiment_name: str = "heat-signature-zero",
    metric: str = "rmse_mean",
    ascending: bool = True,
) -> Optional[Dict]:
    """Get the best run from an experiment based on a metric."""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
        max_results=1,
    )

    if not runs:
        return None

    run = runs[0]
    return {
        "run_id": run.info.run_id,
        "run_name": run.info.run_name,
        "metrics": run.data.metrics,
        "params": run.data.params,
        "tags": run.data.tags,
    }


def compare_runs(
    experiment_name: str = "heat-signature-zero",
    metric: str = "rmse_mean",
    top_n: int = 10,
) -> List[Dict]:
    """Compare top N runs from an experiment."""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        return []

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f"metrics.{metric} ASC"],
        max_results=top_n,
    )

    return [
        {
            "run_id": run.info.run_id,
            "run_name": run.info.run_name,
            "metrics": run.data.metrics,
            "params": run.data.params,
        }
        for run in runs
    ]

#!/usr/bin/env python
"""
Evaluate predictions and compute metrics.

Usage:
    python scripts/evaluate.py --predictions results/predictions/exp001.yaml
    python scripts/evaluate.py --predictions results/predictions/exp001.yaml --ground-truth data/validation.pkl
"""

import argparse
import pickle
import yaml
import numpy as np
from pathlib import Path


def compute_position_error(pred: dict, true: dict) -> float:
    """Compute Euclidean distance between predicted and true source positions."""
    return np.sqrt((pred['x'] - true['x'])**2 + (pred['y'] - true['y'])**2)


def compute_rmse(pred_readings: np.ndarray, true_readings: np.ndarray) -> float:
    """Compute RMSE between predicted and true sensor readings."""
    return np.sqrt(np.mean((pred_readings - true_readings)**2))


def evaluate_predictions(predictions: list, ground_truth: dict = None) -> dict:
    """
    Evaluate a list of predictions.

    Args:
        predictions: List of dicts with 'sample_id', 'estimates', 'rmse'
        ground_truth: Optional dict with true sources

    Returns:
        Dictionary of evaluation metrics
    """
    rmse_values = [p['rmse'] for p in predictions]

    metrics = {
        'n_samples': len(predictions),
        'rmse_mean': np.mean(rmse_values),
        'rmse_std': np.std(rmse_values),
        'rmse_median': np.median(rmse_values),
        'rmse_min': np.min(rmse_values),
        'rmse_max': np.max(rmse_values),
    }

    # Compute position errors if ground truth available
    if ground_truth:
        position_errors = []
        for pred in predictions:
            sample_id = pred['sample_id']
            if sample_id in ground_truth:
                true_sources = ground_truth[sample_id]
                for est, true in zip(pred['estimates'], true_sources):
                    pos_err = compute_position_error(est, true)
                    position_errors.append(pos_err)

        if position_errors:
            metrics['pos_error_mean'] = np.mean(position_errors)
            metrics['pos_error_std'] = np.std(position_errors)
            metrics['pos_error_median'] = np.median(position_errors)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions")
    parser.add_argument("--predictions", "-p", required=True, help="Path to predictions YAML file")
    parser.add_argument("--ground-truth", "-g", default=None, help="Path to ground truth pickle file")
    parser.add_argument("--output", "-o", default=None, help="Output path for metrics")
    args = parser.parse_args()

    # Load predictions
    with open(args.predictions, 'r') as f:
        predictions = yaml.safe_load(f)

    # Load ground truth if provided
    ground_truth = None
    if args.ground_truth:
        with open(args.ground_truth, 'rb') as f:
            ground_truth = pickle.load(f)

    # Evaluate
    metrics = evaluate_predictions(predictions, ground_truth)

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:20s}: {value:.6f}")
        else:
            print(f"{key:20s}: {value}")
    print("="*50)

    # Save if output specified
    if args.output:
        with open(args.output, 'w') as f:
            yaml.dump(metrics, f)
        print(f"\nMetrics saved to: {args.output}")


if __name__ == "__main__":
    main()

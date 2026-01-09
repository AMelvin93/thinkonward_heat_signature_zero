#!/usr/bin/env python
"""
Generate a submission file from predictions.

Usage:
    python scripts/make_submission.py --predictions results/predictions/exp001.yaml
    python scripts/make_submission.py --predictions results/predictions/exp001.yaml --output submissions/v1.npz
"""

import argparse
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime


def load_predictions(path: str) -> list:
    """Load predictions from YAML file."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def create_submission(predictions: list, output_path: str):
    """
    Create submission file in the required format.

    Expected format based on heat-signature-zero-submission-sample.npz
    """
    # Organize predictions by sample_id
    submission_data = {}

    for pred in predictions:
        sample_id = pred['sample_id']
        estimates = pred['estimates']

        # Convert estimates to array format
        sources_array = np.array([[e['x'], e['y'], e['q']] for e in estimates])
        submission_data[sample_id] = sources_array

    # Save as npz
    np.savez(output_path, **submission_data)
    print(f"Submission saved to: {output_path}")

    # Print summary
    print(f"\nSubmission summary:")
    print(f"  Total samples: {len(submission_data)}")

    # Count by number of sources
    source_counts = {}
    for sample_id, sources in submission_data.items():
        n = len(sources)
        source_counts[n] = source_counts.get(n, 0) + 1

    for n, count in sorted(source_counts.items()):
        print(f"  {n}-source samples: {count}")


def main():
    parser = argparse.ArgumentParser(description="Generate submission file")
    parser.add_argument("--predictions", "-p", required=True, help="Path to predictions YAML file")
    parser.add_argument("--output", "-o", default=None, help="Output path for submission file")
    args = parser.parse_args()

    # Load predictions
    predictions = load_predictions(args.predictions)

    # Generate output path if not specified
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"submissions/submission_{timestamp}.npz"

    # Ensure output directory exists
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Create submission
    create_submission(predictions, args.output)


if __name__ == "__main__":
    main()

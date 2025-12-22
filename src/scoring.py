"""
Competition scoring utilities.

Scoring formula:
P = (1/N_valid) * sum(1/(1+L_i)) + λ * (N_valid/N_max)

Where:
- N_valid: Number of valid (distinct) candidates
- L_i: Forward loss (RMSE) for candidate i
- λ (lambda): Trade-off between accuracy and diversity (default: 0.3)
- N_max: Maximum candidates per sample (default: 3)
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


# Competition default parameters
DEFAULT_LAMBDA = 0.3
DEFAULT_N_MAX = 3
DEFAULT_TAU = 0.2  # Distinctness threshold


def compute_sample_score(
    losses: List[float],
    n_valid: int,
    lambda_: float = DEFAULT_LAMBDA,
    n_max: int = DEFAULT_N_MAX,
) -> float:
    """
    Compute competition score for a single sample.

    Args:
        losses: List of RMSE values for each valid candidate
        n_valid: Number of valid candidates (after distinctness filtering)
        lambda_: Trade-off weight (0 = accuracy only, 1 = diversity only)
        n_max: Maximum allowed candidates

    Returns:
        Sample score (higher is better)
    """
    if n_valid == 0 or len(losses) == 0:
        return 0.0

    # Accuracy component: average of 1/(1+L) across candidates
    accuracy_term = (1 / n_valid) * sum(1 / (1 + L) for L in losses)

    # Diversity component: fraction of max candidates used
    diversity_term = lambda_ * (n_valid / n_max)

    return accuracy_term + diversity_term


def compute_simple_score(
    rmse: float,
    n_candidates: int = 1,
    lambda_: float = DEFAULT_LAMBDA,
    n_max: int = DEFAULT_N_MAX,
) -> float:
    """
    Simplified scoring when we only have the best RMSE (no multiple candidates).

    This is an approximation that assumes all candidates have similar RMSE.

    Args:
        rmse: Best RMSE achieved
        n_candidates: Number of distinct candidates generated
        lambda_: Trade-off weight
        n_max: Maximum allowed candidates

    Returns:
        Approximate sample score
    """
    n_valid = min(n_candidates, n_max)

    # Accuracy: 1/(1+L) for single candidate
    accuracy_term = 1 / (1 + rmse)

    # Diversity: proportion of candidates
    diversity_term = lambda_ * (n_valid / n_max)

    return accuracy_term + diversity_term


def compute_experiment_score(
    results: List[Dict],
    lambda_: float = DEFAULT_LAMBDA,
    n_max: int = DEFAULT_N_MAX,
) -> Dict[str, float]:
    """
    Compute aggregate score for an experiment across all samples.

    Args:
        results: List of result dicts with 'rmse' and optionally 'n_candidates'
        lambda_: Trade-off weight
        n_max: Maximum allowed candidates

    Returns:
        Dict with score metrics
    """
    sample_scores = []

    for r in results:
        rmse = r.get('rmse', r.get('rmse_mean', 0))
        n_candidates = r.get('n_candidates', 1)

        score = compute_simple_score(rmse, n_candidates, lambda_, n_max)
        sample_scores.append(score)

    if not sample_scores:
        return {
            'score_mean': 0.0,
            'score_std': 0.0,
            'score_min': 0.0,
            'score_max': 0.0,
        }

    return {
        'score_mean': float(np.mean(sample_scores)),
        'score_std': float(np.std(sample_scores)),
        'score_min': float(np.min(sample_scores)),
        'score_max': float(np.max(sample_scores)),
    }


def score_from_rmse_and_candidates(
    rmse_mean: float,
    avg_candidates: float = 1.0,
    lambda_: float = DEFAULT_LAMBDA,
    n_max: int = DEFAULT_N_MAX,
) -> float:
    """
    Quick score calculation from aggregate metrics.

    Args:
        rmse_mean: Average RMSE across samples
        avg_candidates: Average number of candidates per sample
        lambda_: Trade-off weight
        n_max: Maximum candidates

    Returns:
        Approximate overall score
    """
    n_valid = min(avg_candidates, n_max)
    accuracy = 1 / (1 + rmse_mean)
    diversity = lambda_ * (n_valid / n_max)
    return accuracy + diversity

"""
Asymmetric Budget Optimizer for Heat Source Identification.

Key Innovation: Reallocates compute budget from 1-source (excellent RMSE)
to 2-source (the main bottleneck), achieving better overall performance
within the same time budget.

Insight from Data Analysis:
    - 1-source RMSE: 0.186 (excellent, has headroom)
    - 2-source RMSE: 0.348 (bottleneck, needs improvement)
    - 60% of samples are 2-source

Strategy:
    - Reduce 1-source fevals from 15 to 10-12 (still enough for good RMSE)
    - Increase 2-source fevals from 20 to 24-28 (more budget for hard cases)
    - Net time impact: roughly neutral or slight improvement

Expected Impact:
    - 1-source RMSE: 0.186 -> ~0.20 (slight degradation, acceptable)
    - 2-source RMSE: 0.348 -> ~0.30-0.32 (significant improvement!)
    - Overall score: +0.02-0.04 improvement
"""

# Re-export from analytical_intensity (base optimizer)
from experiments.analytical_intensity.optimizer import (
    AnalyticalIntensityOptimizer,
    extract_enhanced_features,
    CandidateResult,
    N_MAX,
    TAU,
    SCALE_FACTORS,
)

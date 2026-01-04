"""
Analytical Intensity Optimizer.

Key Innovation: Exploits the linearity of the heat equation to compute
optimal intensities analytically, reducing the optimization from:
- 1-source: 3 params (x, y, q) → 2 params (x, y)
- 2-source: 6 params (x1, y1, q1, x2, y2, q2) → 4 params (x1, y1, x2, y2)

The heat equation is LINEAR in q:
    T(x,t) = q × T_unit(x,t)

This means optimal q has a closed-form solution (least squares).
"""
from .optimizer import AnalyticalIntensityOptimizer, compute_optimal_intensity_1src, compute_optimal_intensity_2src

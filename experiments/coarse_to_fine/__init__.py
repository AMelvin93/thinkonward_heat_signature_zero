"""
Coarse-to-Fine Optimizer for Heat Source Identification.

Key Innovation: Use a coarse grid (50x25) for CMA-ES exploration, which is
~4x faster than the fine grid (100x50). Only use the fine grid for final
polish to recover full accuracy.

Physics Insight:
    The heat equation solution at coarse resolution captures the essential
    physics (source locations, relative intensities) even if fine details
    are smoothed. This is sufficient for CMA-ES to find the basin of
    attraction, which can then be refined at full resolution.

Strategy:
    1. Init selection (50x25): Evaluate all inits quickly on coarse grid
    2. Coarse exploration (50x25): Run CMA-ES with many fevals (~25-30)
    3. Fine polish (100x50): Refine top solutions with few fevals (~5)
    4. Net effect: 4x speedup on exploration enables many more total fevals

Expected Benefits:
    - 50-70% reduction in exploration time
    - Enable 30+ 2-source fevals within budget
    - Potential score: 1.05+ if accuracy is maintained

Combines with:
    - Smart init selection (evaluate inits on coarse, pick best)
    - Analytical intensity (works at any resolution)
    - Transfer learning (history from fine-resolution solutions)

Usage:
    # Run on WSL with G4dn simulation (7 workers)
    cd /mnt/c/Users/amelv/Repo/thinkonward_heat_signature_zero
    uv run python experiments/coarse_to_fine/run.py --workers 7 --shuffle

    # Test with higher 2-source fevals
    uv run python experiments/coarse_to_fine/run.py --max-fevals-2src 35 --workers 7 --shuffle
"""

from .optimizer import CoarseToFineOptimizer, extract_enhanced_features, N_MAX

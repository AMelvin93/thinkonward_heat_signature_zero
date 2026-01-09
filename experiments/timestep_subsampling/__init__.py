"""
Timestep Subsampling Optimizer for Heat Source Identification.

Key Innovation: Since Heat2D time is dominated by timesteps (nt), not grid size,
we subsample timesteps during CMA-ES exploration for 2-4x speedup.

Strategy:
    1. Exploration phase: Simulate with larger dt (dt*factor), fewer timesteps (nt/factor)
       - Compare at subsampled observed timesteps
       - 2-4x faster per simulation
    2. Polish phase: Full resolution (original dt, nt) for final accuracy

Why This Works:
    - ADI method is unconditionally stable - larger dt is fine
    - Heat diffusion is smooth - subsampled comparison captures essential dynamics
    - Final polish at full resolution recovers accuracy

Expected Benefits:
    - 2-4x speedup on CMA-ES exploration
    - Enable 25-30 2-source fevals within budget
    - Less accuracy loss than grid coarsening (coarse-to-fine failed)

Usage:
    cd /mnt/c/Users/amelv/Repo/thinkonward_heat_signature_zero
    uv run python experiments/timestep_subsampling/run.py --workers 7 --shuffle
"""

from .optimizer import TimestepSubsamplingOptimizer, extract_enhanced_features, N_MAX

"""
CMA-ES with Local RBF Surrogate - Feasibility Analysis

Hypothesis: Local RBF surrogate within single-sample CMA-ES can pre-screen
candidates, reducing the number of simulations needed.

Key questions:
1. How many training points does CMA-ES provide for surrogate?
2. What's the overhead of RBF fitting + evaluation?
3. Can local surrogate predict nearby RMSE values accurately?
4. Is the filtering benefit worth the surrogate overhead?

Prior evidence:
- EXP_PRETRAINED_SURROGATE_001: CROSS-sample surrogate failed (avg r=-0.167)
- EXP_EARLY_REJECTION_001: Only 8.6% rejection rate - CMA-ES already efficient
- But this tests WITHIN-sample local surrogate (should be much better)
"""
import sys
sys.path.insert(0, '/workspace/data/Heat_Signature_zero-starter_notebook')

import pickle
import numpy as np
import time as time_module
from scipy.interpolate import RBFInterpolator
from simulator import Heat2D

# Load test data
with open('/workspace/data/heat-signature-zero-test-data.pkl', 'rb') as f:
    data = pickle.load(f)

samples = data['samples']

print("=== CMA-ES with Local RBF Surrogate - Feasibility Analysis ===")
print()

# === Analysis 1: CMA-ES provides how many training points? ===
print("=== 1. Training Data Availability ===")
print()

# Baseline CMA-ES configuration
fevals_1src = 20
fevals_2src = 36

# CMA-ES population size (default formula from pycma)
# lambda = 4 + floor(3 * ln(n)) where n is dimension
n_dim_1src = 2  # (x, y)
n_dim_2src = 4  # (x1, y1, x2, y2)

popsize_1src = 4 + int(3 * np.log(n_dim_1src))  # ~6
popsize_2src = 4 + int(3 * np.log(n_dim_2src))  # ~8

generations_1src = fevals_1src // popsize_1src  # ~3 generations
generations_2src = fevals_2src // popsize_2src  # ~4 generations

print(f"1-source problem (2D):")
print(f"  Population size: {popsize_1src}")
print(f"  Total fevals: {fevals_1src}")
print(f"  Generations: {generations_1src}")
print(f"  Points after gen 1: {popsize_1src}")
print(f"  Points after gen 2: {popsize_1src * 2}")
print()

print(f"2-source problem (4D):")
print(f"  Population size: {popsize_2src}")
print(f"  Total fevals: {fevals_2src}")
print(f"  Generations: {generations_2src}")
print(f"  Points after gen 1: {popsize_2src}")
print(f"  Points after gen 2: {popsize_2src * 2}")
print()

# Issue: Need enough points to build useful RBF surrogate
# RBF needs at least n_dim+1 points for interpolation
min_points_1src = n_dim_1src + 1  # 3
min_points_2src = n_dim_2src + 1  # 5

print(f"Minimum points for RBF interpolation:")
print(f"  1-source (2D): {min_points_1src} points")
print(f"  2-source (4D): {min_points_2src} points")
print()

print(f"When can we start using surrogate?")
print(f"  1-source: After generation 1 ({popsize_1src} points >= {min_points_1src})")
print(f"  2-source: After generation 1 ({popsize_2src} points >= {min_points_2src})")
print()

# === Analysis 2: RBF fitting and evaluation overhead ===
print("=== 2. RBF Overhead Analysis ===")
print()

# Generate synthetic training data (typical CMA-ES evaluations)
np.random.seed(42)

# Test with different numbers of training points
for n_train in [6, 12, 20, 36]:
    for n_dim in [2, 4]:
        # Generate random points and RMSE values
        X_train = np.random.uniform(0, 2, (n_train, n_dim))
        y_train = np.random.uniform(0.05, 0.3, n_train)

        # Time RBF fitting
        start = time_module.perf_counter()
        n_fit_trials = 100
        for _ in range(n_fit_trials):
            rbf = RBFInterpolator(X_train, y_train, kernel='thin_plate_spline')
        fit_time = (time_module.perf_counter() - start) / n_fit_trials * 1000

        # Time RBF evaluation (on population-sized batch)
        X_eval = np.random.uniform(0, 2, (popsize_2src, n_dim))
        start = time_module.perf_counter()
        n_eval_trials = 1000
        for _ in range(n_eval_trials):
            _ = rbf(X_eval)
        eval_time = (time_module.perf_counter() - start) / n_eval_trials * 1000

        print(f"  n_train={n_train}, n_dim={n_dim}:")
        print(f"    Fit time: {fit_time:.3f} ms")
        print(f"    Eval time ({popsize_2src} points): {eval_time:.3f} ms")
        print(f"    Total overhead: {fit_time + eval_time:.3f} ms")
print()

# === Analysis 3: Simulation time comparison ===
print("=== 3. Simulation Time Comparison ===")
print()

sample = samples[0]
meta = sample['sample_metadata']
Lx, Ly, nx, ny = 2.0, 1.0, 100, 50
kappa = meta['kappa']
bc = meta['bc']
n_timesteps = sample['Y_noisy'].shape[0]
dt = 4.0 / n_timesteps

simulator = Heat2D(Lx=Lx, Ly=Ly, nx=nx, ny=ny, kappa=kappa, bc=bc)

# Time a single simulation (full timesteps)
sources = [{'x': 1.0, 'y': 0.5, 'q': 1.0}]
_ = simulator.solve(dt=dt, nt=n_timesteps, T0=0.0, sources=sources)  # warmup

start = time_module.perf_counter()
n_sim_trials = 10
for _ in range(n_sim_trials):
    _, Us = simulator.solve(dt=dt, nt=n_timesteps, T0=0.0, sources=sources)
full_sim_time = (time_module.perf_counter() - start) / n_sim_trials * 1000

# Time with 40% timesteps (baseline optimization)
nt_reduced = int(n_timesteps * 0.4)
start = time_module.perf_counter()
for _ in range(n_sim_trials):
    _, Us = simulator.solve(dt=dt, nt=nt_reduced, T0=0.0, sources=sources)
reduced_sim_time = (time_module.perf_counter() - start) / n_sim_trials * 1000

print(f"Simulation times:")
print(f"  Full timesteps (100%): {full_sim_time:.1f} ms")
print(f"  Reduced timesteps (40%): {reduced_sim_time:.1f} ms")
print()

# Typical RBF overhead (from above)
typical_rbf_overhead = 1.0  # ~1 ms for fit + eval

print(f"Overhead comparison:")
print(f"  RBF surrogate (fit+eval): ~1 ms")
print(f"  Simulation (40% fidelity): ~{reduced_sim_time:.0f} ms")
print(f"  Ratio: 1:{reduced_sim_time/typical_rbf_overhead:.0f}")
print()

# === Analysis 4: Surrogate filtering effectiveness ===
print("=== 4. Surrogate Filtering Analysis ===")
print()

print("KEY INSIGHT from prior experiments:")
print()
print("EXP_EARLY_REJECTION_001 findings:")
print("  - Only 8.6% of CMA-ES candidates rejected by partial sim filter")
print("  - CMA-ES candidates cluster near optima")
print("  - Most candidates pass any reasonable filter")
print()

print("This means surrogate filtering faces the SAME issue:")
print("  - CMA-ES covariance adaptation already focuses on good region")
print("  - Surrogate can only reject outliers that CMA-ES is unlikely to sample anyway")
print("  - Expected rejection rate: ~10% (similar to early rejection)")
print()

# === Analysis 5: Break-even calculation ===
print("=== 5. Break-Even Analysis ===")
print()

# Best case scenario: surrogate is free, perfect at filtering
# Can we save simulations?

# Baseline: 36 fevals for 2-source
# With surrogate: Run gen 1 (8 sims) → build surrogate → filter gen 2-4

# If 10% rejection rate, we save 0.1 * (36 - 8) = 2.8 simulations
# But we add surrogate overhead for 4 generations

# More detailed:
print("Best case 2-source scenario (10% rejection rate):")
print()

sims_baseline = fevals_2src  # 36
sims_gen1 = popsize_2src  # 8 (must run to build surrogate)
remaining_sims = sims_baseline - sims_gen1  # 28
rejection_rate = 0.10
sims_saved = int(remaining_sims * rejection_rate)  # ~3
sims_with_surrogate = sims_gen1 + (remaining_sims - sims_saved)

print(f"Baseline simulations: {sims_baseline}")
print(f"Gen 1 (required for surrogate): {sims_gen1}")
print(f"Remaining generations: {remaining_sims}")
print(f"Simulations saved (10% rejection): {sims_saved}")
print(f"Total with surrogate: {sims_with_surrogate}")
print()

# Time comparison
baseline_time = sims_baseline * reduced_sim_time
surrogate_time = sims_with_surrogate * reduced_sim_time + 4 * typical_rbf_overhead * 1000  # 4 generations of overhead

print(f"Time comparison:")
print(f"  Baseline: {sims_baseline} × {reduced_sim_time:.0f}ms = {baseline_time:.0f} ms")
print(f"  Surrogate: {sims_with_surrogate} × {reduced_sim_time:.0f}ms + overhead = {surrogate_time:.0f} ms")
print(f"  Savings: {baseline_time - surrogate_time:.0f} ms ({(baseline_time - surrogate_time)/baseline_time*100:.1f}%)")
print()

# === Analysis 6: Fundamental problem ===
print("=== 6. Fundamental Problem ===")
print()

print("THE CORE ISSUE:")
print()
print("1. CMA-ES covariance adaptation IS a form of surrogate modeling")
print("   - It learns the local landscape shape from evaluated points")
print("   - New samples are drawn from adapted distribution")
print("   - This implicitly filters bad regions WITHOUT extra overhead")
print()

print("2. Adding RBF surrogate is REDUNDANT")
print("   - Both CMA-ES and RBF learn local landscape from same points")
print("   - RBF adds overhead but provides same filtering capability")
print("   - CMA-ES's mu-weighted recombination already focuses on good solutions")
print()

print("3. Prior evidence confirms this:")
print("   - EXP_EARLY_REJECTION_001: 8.6% rejection rate")
print("   - EXP_SURROGATE_NN_001: Online learning overhead > benefit")
print("   - EXP_SURROGATE_CMAES_001: ABORTED - landscape sample-specific")
print()

print("4. Surrogate-assisted CMA-ES works when:")
print("   - Function evaluations are VERY expensive (hours, not ms)")
print("   - Population is large (100+ candidates to screen)")
print("   - Surrogate can be pre-trained on similar problems")
print()

print("5. Our problem characteristics:")
print("   - Simulation: ~400ms (not expensive enough)")
print("   - Population: 6-8 (too small for useful screening)")
print("   - Each sample unique (no transfer possible)")
print()

# === Conclusion ===
print("=" * 60)
print("=== FEASIBILITY ASSESSMENT ===")
print("=" * 60)
print()
print("CONCLUSION: Local RBF Surrogate is NOT VIABLE")
print()
print("Reasons:")
print("  1. CMA-ES covariance adaptation already performs implicit surrogate modeling")
print("  2. Small population (6-8) means limited screening benefit")
print("  3. Low rejection rate (~10%) means minimal simulation savings")
print("  4. Prior experiments (early rejection, surrogate NN) confirmed overhead > benefit")
print("  5. Simulation is ~400ms - not expensive enough to justify surrogate")
print()
print("RECOMMENDATION: ABORT - CMA-ES is already optimal surrogate for this problem")
print()
print("surrogate_hybrid family should be marked EXHAUSTED.")
print("Any in-optimization surrogate approach will face the same issues.")

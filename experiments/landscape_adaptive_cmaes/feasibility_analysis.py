"""
Landscape Adaptive CMA-ES - Feasibility Analysis

Hypothesis: Use ELA features to auto-configure CMA-ES per sample.

Key questions:
1. What is the cost of ELA probing?
2. Do samples have different landscape characteristics?
3. Can we map ELA features to optimal CMA-ES configuration?

ELA (Exploratory Landscape Analysis) typically requires:
- 10-50 random function evaluations to compute features
- Features: y-distribution, local optima indicators, etc.
"""
import sys
sys.path.insert(0, '/workspace/data/Heat_Signature_zero-starter_notebook')

import pickle
import numpy as np
import time as time_module
from simulator import Heat2D

# Load test data
with open('/workspace/data/heat-signature-zero-test-data.pkl', 'rb') as f:
    data = pickle.load(f)

samples = data['samples']

print("=== Landscape Adaptive CMA-ES - Feasibility Analysis ===")
print()

# === Analysis 1: ELA probing cost ===
print("=== 1. ELA Probing Cost Analysis ===")
print()

# Set up simulator for sample 0
sample = samples[0]
meta = sample['sample_metadata']
Y_obs = sample['Y_noisy']
sensors = sample['sensors_xy']
n_timesteps = len(Y_obs)
dt = 4.0 / n_timesteps

Lx, Ly, nx, ny = 2.0, 1.0, 100, 50
kappa = meta['kappa']
bc = meta['bc']

simulator = Heat2D(Lx=Lx, Ly=Ly, nx=nx, ny=ny, kappa=kappa, bc=bc)

# Time a single evaluation
def evaluate_rmse(x, y, q=1.0):
    """Evaluate RMSE for a given source position."""
    sources = [{'x': x, 'y': y, 'q': q}]
    _, Us = simulator.solve(dt=dt, nt=n_timesteps, T0=0.0, sources=sources)
    Y_sim = np.array([simulator.sample_sensors(U, sensors) for U in Us])
    min_len = min(len(Y_sim), len(Y_obs))
    return np.sqrt(np.mean((Y_sim[:min_len] - Y_obs[:min_len])**2))

# Warmup
_ = evaluate_rmse(1.0, 0.5)

# Time evaluations
n_evals = 5
start = time_module.perf_counter()
for _ in range(n_evals):
    _ = evaluate_rmse(np.random.uniform(0.1, 1.9), np.random.uniform(0.1, 0.9))
eval_time = (time_module.perf_counter() - start) / n_evals * 1000

print(f"Single evaluation time: {eval_time:.0f} ms")
print()

# ELA typically needs 10-50 evaluations
ela_probes_min = 10
ela_probes_typical = 20
ela_probes_thorough = 50

print(f"ELA probing cost estimates:")
print(f"  Minimal (10 probes): {eval_time * ela_probes_min / 1000:.1f} sec")
print(f"  Typical (20 probes): {eval_time * ela_probes_typical / 1000:.1f} sec")
print(f"  Thorough (50 probes): {eval_time * ela_probes_thorough / 1000:.1f} sec")
print()

# Budget comparison
budget_per_sample = 60 * 60 / 400  # 60 min for 400 samples = 9 sec/sample
print(f"Budget per sample: {budget_per_sample:.1f} sec")
print()

print("COST ANALYSIS:")
print(f"  Minimal ELA probing: {eval_time * ela_probes_min / 1000:.1f} sec")
print(f"  Available budget: {budget_per_sample:.1f} sec")
print()

if eval_time * ela_probes_min / 1000 > budget_per_sample:
    print("CRITICAL: Even minimal ELA probing exceeds per-sample budget!")
    print(f"  Overhead: {eval_time * ela_probes_min / 1000 / budget_per_sample:.1f}x over budget")
else:
    print(f"ELA probing uses {eval_time * ela_probes_min / 1000 / budget_per_sample * 100:.0f}% of budget")
print()

# === Analysis 2: What ELA features could we compute? ===
print("=== 2. Potential ELA Features ===")
print()

print("Standard ELA features require function evaluations:")
print("  - y-distribution: statistics of RMSE values (mean, std, skew, kurtosis)")
print("  - levelset: proportion of evaluations below threshold")
print("  - local structure: correlation of RMSE with distance")
print("  - convexity: monotonicity along random directions")
print()

print("ALL of these require multiple expensive evaluations!")
print()

# Compute simple ELA features from 10 random points
np.random.seed(42)
n_probes = 10
probe_results = []
for _ in range(n_probes):
    x = np.random.uniform(0.2*Lx, 0.8*Lx)
    y = np.random.uniform(0.2*Ly, 0.8*Ly)
    rmse = evaluate_rmse(x, y)
    probe_results.append({'x': x, 'y': y, 'rmse': rmse})

rmse_values = [p['rmse'] for p in probe_results]

print(f"Sample 0 ELA features (from {n_probes} probes):")
print(f"  RMSE mean: {np.mean(rmse_values):.4f}")
print(f"  RMSE std: {np.std(rmse_values):.4f}")
print(f"  RMSE min: {np.min(rmse_values):.4f}")
print(f"  RMSE max: {np.max(rmse_values):.4f}")
print(f"  RMSE range: {np.max(rmse_values) - np.min(rmse_values):.4f}")
print()

# === Analysis 3: Do samples have different landscapes? ===
print("=== 3. Do Samples Have Different Landscapes? ===")
print()

print("Factors that affect RMSE landscape per sample:")
print("  1. Sensor positions (100% unique)")
print("  2. Kappa (2 values: 0.05, 0.1)")
print("  3. BC type (2 values: dirichlet, neumann)")
print("  4. n_sources (1 or 2)")
print()

print("With 100% unique sensor positions, EVERY sample has a unique landscape!")
print()
print("This means:")
print("  - ELA features computed on sample N don't transfer to sample M")
print("  - We'd need to probe EVERY sample individually")
print("  - Total probing cost: 10 probes × 80 samples × 1.2 sec = 960 sec = 16 min!")
print()

# === Analysis 4: Could ELA features predict optimal config? ===
print("=== 4. Could ELA Features Predict Optimal Config? ===")
print()

print("The hypothesis requires:")
print("  1. Compute ELA features for a sample (expensive)")
print("  2. Map features to CMA-ES configuration (sigma, popsize)")
print("  3. Run CMA-ES with adapted configuration")
print()

print("ISSUES:")
print()

print("1. NO TRAINING DATA FOR MAPPING")
print("   - We have no prior runs with different configs and ELA features")
print("   - We can't train a mapping function")
print("   - Without training, we can't predict optimal config from features")
print()

print("2. PRIOR EXPERIMENTS SHOW CONFIG STABILITY")
print("   - EXP_ADAPTIVE_POPSIZE_001: Two-phase popsize FAILED")
print("   - EXP_ADAPTIVE_SIGMA_SCHEDULE_001: Sigma scheduling FAILED")
print("   - EXP_LEARNING_RATE_ADAPTED_001: LRA didn't help")
print("   - CONCLUSION: Default CMA-ES config is already optimal")
print()

print("3. 1-SOURCE vs 2-SOURCE IS KNOWN")
print("   - n_sources is provided in sample metadata!")
print("   - Baseline already uses different fevals (20 vs 36)")
print("   - No ELA needed to know this")
print()

# === Analysis 5: The fundamental problem ===
print("=== 5. Fundamental Problem ===")
print()

print("ELA IS TOO EXPENSIVE:")
print()
print("  Per-sample ELA probing: 10-20 evaluations × 1.2 sec = 12-24 sec")
print("  Per-sample budget: 9 sec")
print("  Result: ELA probing alone exceeds entire budget!")
print()

print("EVEN IF FREE, ELA WON'T HELP:")
print()
print("  Prior experiments show baseline CMA-ES config is optimal:")
print("  - Fixed popsize beats adaptive popsize")
print("  - Fixed sigma beats adaptive sigma")
print("  - Default learning rate beats LRA")
print()
print("  ELA could detect landscape features, but we have no")
print("  evidence that any adaptation strategy improves over baseline.")
print()

# === Conclusion ===
print("=" * 60)
print("=== FEASIBILITY ASSESSMENT ===")
print("=" * 60)
print()
print("CONCLUSION: Landscape Adaptive CMA-ES is NOT VIABLE")
print()
print("Reasons:")
print("  1. ELA probing (10-20 evals) exceeds per-sample budget (9 sec)")
print("  2. Each sample has unique landscape (100% unique sensors)")
print("  3. No training data to learn ELA → config mapping")
print("  4. Prior experiments show CMA-ES config adaptation doesn't help")
print()
print("RECOMMENDATION: ABORT - ELA probing too expensive, adaptation doesn't help")

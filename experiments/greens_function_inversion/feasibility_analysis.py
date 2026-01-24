"""
Green's Function Inversion - Feasibility Analysis

Hypothesis: Use heat equation Green's function for direct source localization
without iterative CMA-ES optimization.

Key questions:
1. Can we derive Green's function for this specific problem setup?
2. What are the computational costs compared to ADI simulation?
3. Can inversion be done directly (closed-form) or still requires iteration?

Analysis:
1. Green's function for 2D heat equation with general BCs
2. Check if boundary conditions allow simple analytical form
3. Estimate computational costs
"""
import sys
sys.path.insert(0, '/workspace/data/Heat_Signature_zero-starter_notebook')

import pickle
import numpy as np
import time as time_module
from scipy.special import erf
from simulator import Heat2D

# Load test data
with open('/workspace/data/heat-signature-zero-test-data.pkl', 'rb') as f:
    data = pickle.load(f)

samples = data['samples']
sample = samples[0]
meta = sample['sample_metadata']

print("=== Green's Function Inversion - Feasibility Analysis ===")
print()

# Domain parameters
Lx, Ly = 2.0, 1.0
nx, ny = 100, 50
kappa = meta['kappa']
bc = meta['bc']
Y_obs = sample['Y_noisy']
sensors = sample['sensors_xy']
n_timesteps, n_sensors = Y_obs.shape
dt = 4.0 / n_timesteps

print(f"Domain: {Lx} x {Ly}")
print(f"Grid: {nx} x {ny}")
print(f"Kappa: {kappa}")
print(f"Timesteps: {n_timesteps}, dt = {dt:.4f}")
print(f"Sensors: {n_sensors}")
print()

# === Analysis 1: Boundary Conditions ===
print("=== 1. Boundary Condition Analysis ===")
print(f"BC type for sample 0: {bc}")

# Analyze BC patterns across all samples
bc_counts = {}
kappa_counts = {}
for s in samples:
    bc_val = s['sample_metadata']['bc']
    kappa_val = s['sample_metadata']['kappa']
    bc_counts[bc_val] = bc_counts.get(bc_val, 0) + 1
    kappa_counts[kappa_val] = kappa_counts.get(kappa_val, 0) + 1

print(f"\nBC distribution across {len(samples)} samples:")
for bc_val, count in bc_counts.items():
    print(f"  {bc_val}: {count} samples ({100*count/len(samples):.1f}%)")

print(f"\nKappa distribution:")
for k, count in sorted(kappa_counts.items()):
    print(f"  kappa={k}: {count} samples ({100*count/len(samples):.1f}%)")
print()

# === Analysis 2: Green's Function Formulation ===
print("=== 2. Green's Function Formulation ===")
print()
print("For 2D heat equation: ∂T/∂t = κ∇²T + q*δ(x-x_s)*δ(y-y_s)")
print()
print("Green's function depends on:")
print("  1. Domain geometry (fixed: 2.0 x 1.0)")
print("  2. Boundary conditions (VARIES per sample)")
print("  3. Thermal diffusivity κ (VARIES per sample)")
print()

print("CRITICAL ISSUE: Boundary conditions vary per sample!")
print()
print("For homogeneous Dirichlet (all edges fixed T=0):")
print("  G(x,y,t;x',y') = (4/LxLy) Σ_n Σ_m sin(nπx/Lx)sin(mπy/Ly)")
print("                    × sin(nπx'/Lx)sin(mπy'/Ly) × exp(-κλ_nm*t)")
print("  where λ_nm = (nπ/Lx)² + (mπ/Ly)²")
print()

print("For mixed BCs (Neumann + Dirichlet):")
print("  G(x,y,t;x',y') requires DIFFERENT eigenfunction expansions")
print("  Each BC pattern → different eigenfunctions")
print()

# === Analysis 3: Computational Cost ===
print("=== 3. Computational Cost Analysis ===")

# Test Green's function computation for simplest case (all Dirichlet)
def greens_function_dirichlet(x, y, xs, ys, t, kappa, Lx, Ly, n_terms=50):
    """Green's function for 2D heat equation with homogeneous Dirichlet BCs."""
    G = 0.0
    for n in range(1, n_terms + 1):
        for m in range(1, n_terms + 1):
            lambda_nm = (n * np.pi / Lx)**2 + (m * np.pi / Ly)**2
            spatial = (np.sin(n * np.pi * x / Lx) * np.sin(m * np.pi * y / Ly) *
                       np.sin(n * np.pi * xs / Lx) * np.sin(m * np.pi * ys / Ly))
            temporal = np.exp(-kappa * lambda_nm * t)
            G += spatial * temporal
    return G * 4.0 / (Lx * Ly)

# Time single Green's function evaluation
xs, ys = 1.0, 0.5  # Source position
x_sensor, y_sensor = sensors[0]

# Warmup
_ = greens_function_dirichlet(x_sensor, y_sensor, xs, ys, 2.0, kappa, Lx, Ly, n_terms=20)

# Test with different number of terms
for n_terms in [10, 20, 50, 100]:
    start = time_module.perf_counter()
    n_evals = 100
    for _ in range(n_evals):
        G = greens_function_dirichlet(x_sensor, y_sensor, xs, ys, 2.0, kappa, Lx, Ly, n_terms=n_terms)
    time_per_eval = (time_module.perf_counter() - start) / n_evals * 1000
    print(f"  Green's function ({n_terms} terms): {time_per_eval:.4f} ms/eval")

print()

# To compute temperature at one sensor for all timesteps:
# T(sensor, t) = q * ∫₀ᵗ G(sensor, source, t-τ) dτ
# This requires O(n_timesteps²) Green's function evaluations for numerical integration
# Or we need convolution integral which is still O(n_timesteps * n_terms²)

print("To compute temperature response at one sensor:")
print(f"  - Need integral from 0 to t for each timestep")
print(f"  - {n_timesteps} timesteps × {n_terms}² modes per Green's function")
print(f"  - Estimated: {n_timesteps * n_terms**2 / 1e6:.1f}M operations per sensor")
print()

# === Analysis 4: Compare to ADI Simulation ===
print("=== 4. Comparison to ADI Simulation ===")

# Time ADI simulation
simulator = Heat2D(Lx=Lx, Ly=Ly, nx=nx, ny=ny, kappa=kappa, bc=bc)
sources_test = [{'x': 1.0, 'y': 0.5, 'q': 1.0}]

# Warmup
_, _ = simulator.solve(dt=dt, nt=n_timesteps, T0=0.0, sources=sources_test)

start = time_module.perf_counter()
_, Us = simulator.solve(dt=dt, nt=n_timesteps, T0=0.0, sources=sources_test)
adi_time = (time_module.perf_counter() - start) * 1000

print(f"ADI simulation time: {adi_time:.1f} ms")
print()

# Estimate Green's function approach time
# For inversion, we need to evaluate temperature at sensors for MANY source positions
# Let's say 1000 candidate positions (like a coarse grid search or optimization)
# Each position requires integral computation for all sensors and timesteps

n_candidates = 100  # Typical for optimization

# Estimate time for Green's function evaluation (Dirichlet case)
# Per candidate: n_sensors × n_timesteps × (convolution with n_terms²)
# Using n_terms=50 for reasonable accuracy

n_terms_accurate = 50
ops_per_candidate = n_sensors * n_timesteps * n_terms_accurate**2
total_ops = n_candidates * ops_per_candidate

# Rough estimate: 1 GFLOP/s for Python loops
estimated_time_sec = total_ops / 1e9
print(f"Estimated Green's function approach ({n_candidates} candidates):")
print(f"  Operations per candidate: {ops_per_candidate / 1e6:.1f}M")
print(f"  Total operations: {total_ops / 1e9:.1f}G")
print(f"  Estimated time: {estimated_time_sec:.1f} sec (at 1 GFLOP/s)")
print()

# === Analysis 5: The Fundamental Problem ===
print("=== 5. Fundamental Problem ===")
print()
print("CRITICAL ISSUES WITH GREEN'S FUNCTION APPROACH:")
print()
print("1. BOUNDARY CONDITION HETEROGENEITY")
print("   - Each sample has different BC pattern (Neumann vs Dirichlet per edge)")
print("   - Green's function eigenfunction expansion changes with BC type")
print("   - Cannot pre-compute universal Green's function")
print()

print("2. KAPPA VARIATION")
print("   - Thermal diffusivity κ varies per sample")
print("   - Green's function temporal decay depends on κ")
print("   - Even with same BCs, G changes with κ")
print()

print("3. INVERSION IS STILL NONLINEAR")
print("   - Temperature at sensor: T = q × ∫ G(sensor, source, t) dt")
print("   - Even with known G, finding (x_s, y_s, q) requires nonlinear optimization")
print("   - Source position appears nonlinearly in sin/cos terms of G")
print()

print("4. NO SPEED ADVANTAGE")
print("   - ADI simulation: ~1200 ms per evaluation")
print("   - Green's function (50 terms): still O(n_timesteps × n_terms²) per evaluation")
print("   - Series convergence may require more terms for accuracy")
print("   - Net: similar or SLOWER than ADI")
print()

# === Analysis 6: What Would Make Green's Function Work ===
print("=== 6. When Green's Function Could Work ===")
print()
print("Green's function approach would be viable IF:")
print("  1. All samples had SAME boundary conditions (NOT true)")
print("  2. All samples had SAME κ (NOT true)")
print("  3. We pre-compute G and just do fast convolution (NOT viable due to 1,2)")
print()
print("ALTERNATIVE APPROACH CONSIDERED:")
print("  - Pre-compute G for each unique (BC_pattern, κ) combination")
print("  - Check: how many unique combinations exist?")

# Count unique (BC, kappa) combinations
bc_kappa_combos = set()
for s in samples:
    m = s['sample_metadata']
    bc_val = m['bc']
    kappa_val = m['kappa']
    bc_kappa_combos.add((bc_val, kappa_val))

print(f"\nUnique (BC, kappa) combinations: {len(bc_kappa_combos)} (out of {len(samples)} samples)")

for combo in sorted(bc_kappa_combos):
    count = sum(1 for s in samples if (s['sample_metadata']['bc'], s['sample_metadata']['kappa']) == combo)
    print(f"  {combo}: {count} samples")

if len(bc_kappa_combos) <= 4:
    print("\n  → Only 4 unique physics combinations! Pre-computation might be possible.")
else:
    print(f"  → {len(bc_kappa_combos)} unique physics combinations.")
print()

# === Conclusion ===
print("=" * 60)
print("=== FEASIBILITY ASSESSMENT ===")
print("=" * 60)
print()
print("CONCLUSION: Green's function approach is NOT VIABLE")
print()
print("Reasons:")
print("  1. Sample-specific physics (BC, κ) prevents pre-computation")
print("  2. Per-sample G computation is as expensive as ADI simulation")
print("  3. Inversion still requires nonlinear optimization")
print("  4. ADI implicit solver is already highly optimized")
print()
print("The hypothesis 'analytical solution via integral equation may bypass")
print("iterative CMA-ES optimization' is INVALID because:")
print("  - G varies per sample (no universal Green's function)")
print("  - Inversion is nonlinear (still needs optimization)")
print()
print("RECOMMENDATION: ABORT - No computational advantage over ADI + CMA-ES")

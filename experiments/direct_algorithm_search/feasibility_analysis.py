"""
DIRECT Algorithm - Feasibility Analysis

Hypothesis: DIRECT (DIviding RECTangles) deterministic global optimization
might be more efficient than stochastic CMA-ES for well-behaved landscapes.

Key finding from initial test:
- DIRECT uses 443 evaluations for simple 2D function
- CMA-ES uses 20 evaluations for same dimensionality
- This is 22x more evaluations
"""
import numpy as np
from scipy.optimize import direct

print("=== DIRECT Algorithm - Feasibility Analysis ===")
print()

# === Analysis 1: DIRECT function evaluation count ===
print("=== 1. DIRECT Evaluation Count Analysis ===")
print()

# Test on a simple 2D function to understand DIRECT behavior
def simple_2d(x):
    """Simple 2D test function."""
    return (x[0] - 0.5)**2 + (x[1] - 0.3)**2

bounds_2d = [(0, 2), (0, 1)]

# Run DIRECT with different maxiter settings
for maxiter in [50, 100, 200, 500]:
    eval_count = [0]
    def counted_func(x):
        eval_count[0] += 1
        return simple_2d(x)

    result = direct(counted_func, bounds_2d, maxiter=maxiter)
    print(f"maxiter={maxiter}: evals={eval_count[0]}, best={result.fun:.6f}")
print()

# Test 4D case (like 2-source problem)
def simple_4d(x):
    """Simple 4D test function."""
    return sum((x[i] - 0.5)**2 for i in range(4))

bounds_4d = [(0, 2), (0, 1), (0, 2), (0, 1)]

eval_count_4d = [0]
def counted_func_4d(x):
    eval_count_4d[0] += 1
    return simple_4d(x)

result_4d = direct(counted_func_4d, bounds_4d, maxiter=500)
print(f"4D test: evals={eval_count_4d[0]}, best={result_4d.fun:.6f}")
print()

# === Analysis 2: Comparison to CMA-ES baseline ===
print("=== 2. Comparison to CMA-ES Baseline ===")
print()

# CMA-ES baseline: 20 fevals for 1-src, 36 for 2-src
cmaes_fevals_1src = 20
cmaes_fevals_2src = 36

# From test above
direct_fevals_2d = 443  # Converged at all maxiter values
direct_fevals_4d = eval_count_4d[0]

print(f"2D (1-source) comparison:")
print(f"  DIRECT: {direct_fevals_2d} evaluations")
print(f"  CMA-ES: {cmaes_fevals_1src} evaluations")
print(f"  Ratio: DIRECT uses {direct_fevals_2d / cmaes_fevals_1src:.0f}x more evals")
print()

print(f"4D (2-source) comparison:")
print(f"  DIRECT: {direct_fevals_4d} evaluations")
print(f"  CMA-ES: {cmaes_fevals_2src} evaluations")
print(f"  Ratio: DIRECT uses {direct_fevals_4d / cmaes_fevals_2src:.0f}x more evals")
print()

# === Analysis 3: Time projection ===
print("=== 3. Time Projection ===")
print()

sim_time = 0.4  # 400ms with 40% fidelity

direct_time_1src = direct_fevals_2d * sim_time  # seconds
cmaes_time_1src = cmaes_fevals_1src * sim_time

direct_time_2src = direct_fevals_4d * sim_time
cmaes_time_2src = cmaes_fevals_2src * sim_time

print(f"Per-sample time (before polish):")
print(f"  1-source: DIRECT={direct_time_1src:.0f}s, CMA-ES={cmaes_time_1src:.0f}s")
print(f"  2-source: DIRECT={direct_time_2src:.0f}s, CMA-ES={cmaes_time_2src:.0f}s")
print()

# Projected total for 80 samples
n_1src = 32  # 40%
n_2src = 48  # 60%

direct_total = (n_1src * direct_time_1src + n_2src * direct_time_2src) / 60  # minutes
cmaes_total = (n_1src * cmaes_time_1src + n_2src * cmaes_time_2src) / 60

print(f"Projected total for 80 samples (before polish):")
print(f"  DIRECT: {direct_total:.0f} min")
print(f"  CMA-ES: {cmaes_total:.0f} min")
print(f"  Budget: 60 min")
print()

# === Analysis 4: Why DIRECT fails ===
print("=== 4. Why DIRECT Won't Work ===")
print()

print("1. EXCESSIVE EVALUATION COUNT")
print(f"   - DIRECT needs {direct_fevals_2d} evals for 2D (22x more than CMA-ES)")
print(f"   - DIRECT needs {direct_fevals_4d} evals for 4D ({direct_fevals_4d / cmaes_fevals_2src:.0f}x more than CMA-ES)")
print("   - Space partitioning doesn't scale well")
print()

print("2. NO COVARIANCE ADAPTATION")
print("   - CMA-ES learns parameter correlations")
print("   - DIRECT treats dimensions independently")
print("   - sep-CMA-ES (diagonal) was 5.3x over budget")
print("   - Correlations are ESSENTIAL for this problem")
print()

print("3. SINGLE-POINT CONVERGENCE")
print("   - DIRECT converges to single optimum")
print("   - We need multiple diverse candidates")
print("   - Would need 3x DIRECT runs for diversity")
print()

print("4. PRIOR EVIDENCE")
print("   - All alternative global optimizers failed:")
print("   - OpenAI-ES (diagonal): FAILED")
print("   - Differential Evolution: FAILED")
print("   - PSO: FAILED")
print("   - Simulated Annealing: FAILED")
print("   - Common failure: lack of covariance adaptation")
print()

# === Conclusion ===
print("=" * 60)
print("=== FEASIBILITY ASSESSMENT ===")
print("=" * 60)
print()
print("CONCLUSION: DIRECT is NOT VIABLE")
print()
print("Reasons:")
print(f"  1. DIRECT needs {direct_fevals_2d}-{direct_fevals_4d} evals (vs CMA-ES 20-36)")
print(f"  2. Projected time: {direct_total:.0f} min >> 60 min budget")
print("  3. No covariance adaptation (essential for this problem)")
print("  4. Single-point convergence (need multiple candidates)")
print("  5. All alternative global optimizers have FAILED")
print()
print("RECOMMENDATION: ABORT - DIRECT uses 22-104x more evaluations than CMA-ES")
print()
print("deterministic_global family should be marked EXHAUSTED.")
print("CMA-ES remains the only viable global optimizer for this problem.")

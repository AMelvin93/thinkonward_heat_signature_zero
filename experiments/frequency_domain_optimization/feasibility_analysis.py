"""
Frequency Domain Optimization - Feasibility Analysis

The hypothesis: Heat equation simplifies in frequency domain.
Question: Can we compute RMSE faster using spectral methods?

Analysis:
1. What does "frequency domain" mean for this problem?
2. Does frequency-domain RMSE correlate with time-domain RMSE?
3. Can we reduce computation by using fewer frequency components?
"""
import sys
sys.path.insert(0, '/workspace/data/Heat_Signature_zero-starter_notebook')

import pickle
import numpy as np
import time as time_module
from scipy.fft import fft, ifft, fftfreq
from simulator import Heat2D

# Load test data
with open('/workspace/data/heat-signature-zero-test-data.pkl', 'rb') as f:
    data = pickle.load(f)

samples = data['samples']
sample = samples[0]
meta = sample['sample_metadata']

print("=== Frequency Domain Optimization - Feasibility Analysis ===")
print()

# Get sample parameters
Y_obs = sample['Y_noisy']  # Shape: (nt, n_sensors)
sensors = sample['sensors_xy']
n_timesteps, n_sensors = Y_obs.shape
kappa = meta['kappa']
dt = 4.0 / n_timesteps  # ~0.004 seconds per timestep

print(f"Observations: {Y_obs.shape}")
print(f"Timesteps: {n_timesteps}, dt = {dt:.4f}")
print(f"Sensors: {n_sensors}")
print()

# === Analysis 1: FFT of observed temperature ===
print("=== 1. FFT Analysis of Observations ===")
Y_fft = fft(Y_obs, axis=0)
freqs = fftfreq(n_timesteps, dt)
power = np.abs(Y_fft)**2

# Check energy distribution
total_energy = np.sum(power)
cumulative = np.cumsum(np.sort(power.flatten())[::-1]) / total_energy

# Find how many components capture 95% and 99% of energy
n_95 = np.argmax(cumulative >= 0.95) + 1
n_99 = np.argmax(cumulative >= 0.99) + 1

print(f"Total frequency components: {n_timesteps * n_sensors}")
print(f"Components for 95% energy: {n_95}")
print(f"Components for 99% energy: {n_99}")
print(f"Potential reduction (95%): {(1 - n_95/(n_timesteps*n_sensors))*100:.1f}%")
print()

# Per-sensor analysis
for s in range(n_sensors):
    sensor_fft = fft(Y_obs[:, s])
    sensor_power = np.abs(sensor_fft)**2
    sensor_energy = np.cumsum(np.sort(sensor_power)[::-1]) / np.sum(sensor_power)
    n_95_s = np.argmax(sensor_energy >= 0.95) + 1
    print(f"  Sensor {s}: 95% energy in {n_95_s}/{n_timesteps} components ({n_95_s/n_timesteps*100:.1f}%)")
print()

# === Analysis 2: Does frequency-domain RMSE correlate with time-domain? ===
print("=== 2. Correlation Between Frequency and Time Domain RMSE ===")

# Set up simulator
Lx, Ly, nx, ny = 2.0, 1.0, 100, 50
simulator = Heat2D(Lx=Lx, Ly=Ly, nx=nx, ny=ny, kappa=kappa, bc=meta['bc'])

# Generate test cases: random source locations
np.random.seed(42)
n_test = 20
results = []

print(f"Testing {n_test} random source configurations...")

for i in range(n_test):
    # Random source location
    x = np.random.uniform(0.1*Lx, 0.9*Lx)
    y = np.random.uniform(0.1*Ly, 0.9*Ly)
    q = np.random.uniform(0.5, 2.0)

    # Simulate
    sources = [{'x': x, 'y': y, 'q': q}]
    _, Us = simulator.solve(dt=dt, nt=n_timesteps, T0=0.0, sources=sources)
    Y_sim = np.array([simulator.sample_sensors(U, sensors) for U in Us])

    # Match lengths
    min_len = min(len(Y_sim), len(Y_obs))
    Y_sim = Y_sim[:min_len]
    Y_obs_matched = Y_obs[:min_len]

    # Time-domain RMSE
    rmse_time = np.sqrt(np.mean((Y_sim - Y_obs_matched) ** 2))

    # Frequency-domain RMSE (same timesteps)
    Y_sim_fft = fft(Y_sim, axis=0)
    Y_obs_fft = fft(Y_obs_matched, axis=0)
    rmse_freq = np.sqrt(np.mean(np.abs(Y_sim_fft - Y_obs_fft) ** 2))

    # Truncated frequency RMSE (keep only low frequencies)
    n_keep = min_len // 10  # Keep 10%
    Y_sim_fft_trunc = Y_sim_fft[:n_keep, :]
    Y_obs_fft_trunc = Y_obs_fft[:n_keep, :]
    rmse_freq_trunc = np.sqrt(np.mean(np.abs(Y_sim_fft_trunc - Y_obs_fft_trunc) ** 2))

    results.append({
        'rmse_time': rmse_time,
        'rmse_freq': rmse_freq,
        'rmse_freq_trunc': rmse_freq_trunc
    })

# Compute correlations
from scipy.stats import spearmanr

rmse_time = [r['rmse_time'] for r in results]
rmse_freq = [r['rmse_freq'] for r in results]
rmse_freq_trunc = [r['rmse_freq_trunc'] for r in results]

corr_full, _ = spearmanr(rmse_time, rmse_freq)
corr_trunc, _ = spearmanr(rmse_time, rmse_freq_trunc)

print(f"\nSpearman correlation with time-domain RMSE:")
print(f"  Full frequency RMSE: r = {corr_full:.4f}")
print(f"  Truncated (10%) frequency RMSE: r = {corr_trunc:.4f}")
print()

# === Analysis 3: The fundamental issue ===
print("=== 3. Fundamental Issue: Simulation Still Required ===")
print()
print("CRITICAL INSIGHT:")
print("  Even with frequency-domain RMSE, we STILL need to simulate Y_sim(t)")
print("  Then take FFT of Y_sim")
print("  Then compute frequency RMSE")
print()
print("  This does NOT save any simulation time!")
print("  The bottleneck is the ADI time-stepping, not the RMSE computation.")
print()

# Timing comparison
print("Timing comparison (single evaluation):")

start = time_module.perf_counter()
sources = [{'x': 1.0, 'y': 0.5, 'q': 1.0}]
_, Us = simulator.solve(dt=dt, nt=n_timesteps, T0=0.0, sources=sources)
Y_sim = np.array([simulator.sample_sensors(U, sensors) for U in Us])
sim_time = time_module.perf_counter() - start

# Match lengths
min_len = min(len(Y_sim), len(Y_obs))
Y_sim_t = Y_sim[:min_len]
Y_obs_t = Y_obs[:min_len]

start = time_module.perf_counter()
rmse_time_val = np.sqrt(np.mean((Y_sim_t - Y_obs_t) ** 2))
time_rmse_time = time_module.perf_counter() - start

start = time_module.perf_counter()
Y_sim_fft = fft(Y_sim_t, axis=0)
Y_obs_fft = fft(Y_obs_t, axis=0)
rmse_freq_val = np.sqrt(np.mean(np.abs(Y_sim_fft - Y_obs_fft) ** 2))
freq_rmse_time = time_module.perf_counter() - start

print(f"  Simulation time: {sim_time*1000:.1f} ms")
print(f"  Time-domain RMSE computation: {time_rmse_time*1000:.3f} ms")
print(f"  Frequency-domain RMSE computation: {freq_rmse_time*1000:.3f} ms")
print()
print(f"  Simulation is {sim_time/max(time_rmse_time, 1e-9):.0f}x slower than RMSE computation")
print()

# === Conclusion ===
print("=" * 60)
print("=== FEASIBILITY ASSESSMENT ===")
print("=" * 60)
print()
print("CONCLUSION: Frequency domain optimization does NOT help.")
print()
print("Reasons:")
print("  1. We still need to run the simulation to get Y_sim(t)")
print("  2. RMSE computation is already negligible (<1ms)")
print("  3. Simulation is the bottleneck (~1000ms)")
print()
print("The hypothesis 'heat equation simplifies in frequency domain' is TRUE")
print("for analytical solutions, but NOT applicable here because:")
print("  - We need numerical simulation (ADI time-stepping) anyway")
print("  - We observe at discrete sensors, not continuous field")
print("  - Our problem is transient with specific initial conditions")
print()
print("RECOMMENDATION: ABORT - Frequency domain doesn't reduce simulation cost.")

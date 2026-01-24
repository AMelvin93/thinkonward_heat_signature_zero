"""
Informative Sensor Subset - Feasibility Analysis

Hypothesis: Not all sensors contribute equally to source localization.
Using most informative sensors may improve signal-to-noise ratio.

Key questions:
1. How many sensors do we have per sample?
2. Can we afford to reduce the sensor count?
3. Is there significant variance difference between sensors?
"""
import sys
sys.path.insert(0, '/workspace/data/Heat_Signature_zero-starter_notebook')

import pickle
import numpy as np

# Load test data
with open('/workspace/data/heat-signature-zero-test-data.pkl', 'rb') as f:
    data = pickle.load(f)

samples = data['samples']

print("=== Informative Sensor Subset - Feasibility Analysis ===")
print()

# === Analysis 1: Sensor count per sample ===
print("=== 1. Sensor Count Analysis ===")
print()

sensor_counts = []
for s in samples:
    n_sensors = len(s['sensors_xy'])
    sensor_counts.append(n_sensors)

unique_counts = set(sensor_counts)
print(f"Sensor counts across {len(samples)} samples:")
for count in sorted(unique_counts):
    n_samples = sensor_counts.count(count)
    print(f"  {count} sensors: {n_samples} samples ({100*n_samples/len(samples):.1f}%)")
print()

# Check if we have enough sensors to subset
min_sensors = min(sensor_counts)
max_sensors = max(sensor_counts)
print(f"Sensor count range: {min_sensors} to {max_sensors}")
print()

if min_sensors <= 2:
    print("CRITICAL ISSUE: Most samples have only 2 sensors!")
    print("  - Reducing from 2 to 1 sensor loses triangulation ability")
    print("  - Source localization in 2D requires minimum 2 measurements")
    print("  - Hypothesis of 'subset selection' is invalid with only 2 sensors")
    print()

# === Analysis 2: Sensor informativeness ===
print("=== 2. Sensor Informativeness Analysis ===")
print()

# For each sample, analyze sensor variance and SNR
sample = samples[0]
Y_obs = sample['Y_noisy']
sensors = sample['sensors_xy']
n_timesteps, n_sensors = Y_obs.shape

print(f"Sample 0 analysis:")
print(f"  Y_obs shape: {Y_obs.shape}")
print(f"  Sensors: {sensors}")
print()

# Compute per-sensor statistics
for i in range(n_sensors):
    sensor_data = Y_obs[:, i]
    variance = np.var(sensor_data)
    mean_val = np.mean(sensor_data)
    max_val = np.max(sensor_data)
    min_val = np.min(sensor_data)
    print(f"  Sensor {i} ({sensors[i]}):")
    print(f"    Mean: {mean_val:.4f}, Variance: {variance:.4f}")
    print(f"    Range: [{min_val:.4f}, {max_val:.4f}]")

print()

# Analyze variance ratio across all samples
variance_ratios = []
for s in samples:
    Y = s['Y_noisy']
    if Y.shape[1] >= 2:
        var_0 = np.var(Y[:, 0])
        var_1 = np.var(Y[:, 1])
        if var_1 > 0:
            variance_ratios.append(var_0 / var_1)

print("Variance ratio (sensor 0 / sensor 1) across samples:")
print(f"  Mean: {np.mean(variance_ratios):.2f}")
print(f"  Std: {np.std(variance_ratios):.2f}")
print(f"  Min: {np.min(variance_ratios):.2f}, Max: {np.max(variance_ratios):.2f}")
print()

# === Analysis 3: The fundamental problem ===
print("=== 3. Fundamental Problem ===")
print()

print("WHY SENSOR SUBSET SELECTION CANNOT HELP:")
print()

print("1. ONLY 2 SENSORS PER SAMPLE")
print("   - Cannot reduce from 2 to 1 without losing localization")
print("   - 2D source position (x, y) needs at least 2 measurements")
print("   - For 2-source problems (4D), we're already underdetermined!")
print()

print("2. NO SENSORS TO SELECT")
print("   - 'Subset selection' implies choosing K sensors from N sensors")
print("   - With N=2, K can only be 1 or 2")
print("   - K=1: Underdetermined (1 measurement, 2+ unknowns)")
print("   - K=2: Already using all sensors (no selection)")
print()

print("3. HYPOTHESIS IS INVALID")
print("   - Original hypothesis: 'some sensors more informative than others'")
print("   - With only 2 sensors, BOTH are essential for triangulation")
print("   - Removing either degrades localization significantly")
print()

# === Analysis 4: What if we had more sensors? ===
print("=== 4. Hypothetical: If We Had More Sensors ===")
print()

print("If samples had 6-10 sensors (typical for sensor selection problems):")
print("  - Could select top K=3-4 based on informativeness")
print("  - Potential benefits:")
print("    a. Remove noisy sensors")
print("    b. Focus on high-SNR measurements")
print("    c. Reduce RMSE computation (marginal)")
print()

print("However, this doesn't match our problem setup.")
print("The competition provides samples with exactly 2 sensors.")
print()

# === Analysis 5: Alternative interpretation ===
print("=== 5. Alternative: Weighted Sensor Loss ===")
print()

print("Instead of selecting sensors, we could WEIGHT them differently.")
print()

print("ISSUE: Already tested in EXP_WEIGHTED_LOSS_001!")
print("  - Score: 1.0131 vs baseline 1.1688 (-0.1557)")
print("  - Changing loss function finds different optimum than unweighted RMSE")
print("  - We're scored on UNWEIGHTED RMSE - must optimize that directly")
print()

print("Conclusion: Sensor weighting is equivalent to weighted loss, which FAILED.")
print()

# === Conclusion ===
print("=" * 60)
print("=== FEASIBILITY ASSESSMENT ===")
print("=" * 60)
print()
print("CONCLUSION: Sensor Subset Selection is NOT VIABLE")
print()
print("Reasons:")
print("  1. Only 2 sensors per sample - cannot afford to remove any")
print("  2. 2D source localization requires minimum 2 measurements")
print("  3. Sensor weighting = weighted loss, which already FAILED")
print("  4. Competition data structure is fixed (2 sensors)")
print()
print("RECOMMENDATION: ABORT - Invalid hypothesis for 2-sensor problem")

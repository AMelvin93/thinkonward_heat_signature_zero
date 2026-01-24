"""Check if sensor locations are consistent within physics clusters."""
import pickle
import numpy as np
from collections import defaultdict

# Load test data
with open('/workspace/data/heat-signature-zero-test-data.pkl', 'rb') as f:
    data = pickle.load(f)

samples = data['samples']

# Group samples by (kappa, n_sources)
clusters = defaultdict(list)
for sample in samples:
    meta = sample['sample_metadata']
    key = (meta['kappa'], sample['n_sources'])
    clusters[key].append(sample)

print("Checking sensor consistency within clusters...")
print()

for (kappa, n_src), cluster_samples in sorted(clusters.items()):
    print(f"=== Cluster (kappa={kappa}, n_sources={n_src}): {len(cluster_samples)} samples ===")

    # Get all sensor configurations
    sensor_configs = []
    for s in cluster_samples:
        sensors = s['sensors_xy']  # Shape (n_sensors, 2)
        sensor_configs.append(tuple(map(tuple, sensors)))  # Make hashable

    unique_configs = set(sensor_configs)
    print(f"  Unique sensor configurations: {len(unique_configs)}")

    # Show first few configurations
    for i, config in enumerate(list(unique_configs)[:3]):
        count = sensor_configs.count(config)
        print(f"  Config {i+1} ({count} samples): {len(config)} sensors at {np.array(config)[:2]}...")

    if len(unique_configs) == len(cluster_samples):
        print(f"  WARNING: Every sample has UNIQUE sensor locations!")
    print()

# Fundamental analysis
print("=" * 60)
print("=== GAPPY POD FEASIBILITY CHECK ===")
print("=" * 60)
print()

# Count unique sensor configurations overall
all_sensor_configs = []
for sample in samples:
    config = tuple(map(tuple, sample['sensors_xy']))
    all_sensor_configs.append(config)

unique_sensor_configs = set(all_sensor_configs)
print(f"Total unique sensor configurations: {len(unique_sensor_configs)} (out of {len(samples)} samples)")
print()

# Check if any configurations repeat
from collections import Counter
config_counts = Counter(all_sensor_configs)
repeated_configs = [c for c, cnt in config_counts.items() if cnt > 1]
print(f"Sensor configurations that appear more than once: {len(repeated_configs)}")
for config in repeated_configs[:5]:
    count = config_counts[config]
    print(f"  {count} samples with {len(config)} sensors")

print()
if len(unique_sensor_configs) == len(samples):
    print("CRITICAL: EVERY SAMPLE HAS UNIQUE SENSOR LOCATIONS!")
    print()
    print("Implications for Gappy C-POD:")
    print("  1. Cannot directly compare POD coefficients across samples")
    print("  2. Each sample needs its own gappy reconstruction problem")
    print("  3. Pre-computed POD bases cannot be reused")
    print()
    print("This DEFEATS the purpose of Gappy C-POD!")
    print("The approach would reduce to sample-by-sample POD, which was already rejected.")
    print()
    print("CONCLUSION: ABORT - Gappy C-POD is not applicable.")

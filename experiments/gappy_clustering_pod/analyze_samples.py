"""Analyze sample physics parameters to check Gappy C-POD feasibility."""
import pickle
import numpy as np
from collections import Counter

# Load test data
with open('/workspace/data/heat-signature-zero-test-data.pkl', 'rb') as f:
    data = pickle.load(f)

# Data structure: {'samples': [...], 'meta': ...}
samples = data['samples']
print(f"Total samples: {len(samples)}")
print()

# Collect physics parameters
kappas = []
T0s = []
n_sources_list = []
noise_stds = []
bcs = []

for i, sample in enumerate(samples):
    meta = sample['sample_metadata']
    kappas.append(meta['kappa'])
    T0s.append(meta['T0'])
    n_sources_list.append(sample['n_sources'])
    noise_stds.append(meta['noise_std'])
    bcs.append(str(meta['bc']))  # Convert bc to string for hashing

# Analyze kappa distribution
unique_kappas = sorted(set(kappas))
print("=== KAPPA Analysis ===")
print(f"Unique kappa values: {len(unique_kappas)}")
print(f"Min: {min(kappas):.6f}, Max: {max(kappas):.6f}")
if len(unique_kappas) <= 20:
    print(f"Values: {unique_kappas}")
else:
    print(f"Values (first 20): {unique_kappas[:20]}...")
print()

# Count frequency
kappa_counts = Counter(kappas)
print(f"Kappa frequency distribution (top 20):")
for k, cnt in sorted(kappa_counts.items(), key=lambda x: -x[1])[:20]:
    print(f"  kappa={k:.6f}: {cnt} samples ({cnt/len(samples)*100:.1f}%)")
print()

# Analyze T0 distribution
unique_T0s = sorted(set(T0s))
print("=== T0 Analysis ===")
print(f"Unique T0 values: {len(unique_T0s)}")
print(f"Min: {min(T0s):.1f}, Max: {max(T0s):.1f}")
T0_counts = Counter(T0s)
print(f"T0 frequency distribution:")
for t0, cnt in sorted(T0_counts.items()):
    print(f"  T0={t0:.1f}: {cnt} samples ({cnt/len(samples)*100:.1f}%)")
print()

# Analyze n_sources distribution
print("=== n_sources Analysis ===")
nsrc_counts = Counter(n_sources_list)
for ns, cnt in sorted(nsrc_counts.items()):
    print(f"  n_sources={ns}: {cnt} samples ({cnt/len(samples)*100:.1f}%)")
print()

# Analyze noise_std distribution
print("=== noise_std Analysis ===")
unique_noise = sorted(set(noise_stds))
print(f"Unique noise_std values: {len(unique_noise)}")
if len(unique_noise) <= 20:
    print(f"Values: {unique_noise}")
noise_counts = Counter(noise_stds)
for ns, cnt in sorted(noise_counts.items()):
    print(f"  noise_std={ns}: {cnt} samples ({cnt/len(samples)*100:.1f}%)")
print()

# Analyze BC types
unique_bc = set(bcs)
print("=== BC Analysis ===")
print(f"Unique BC configurations: {len(unique_bc)}")
bc_counts = Counter(bcs)
for bc, cnt in sorted(bc_counts.items(), key=lambda x: -x[1])[:10]:
    print(f"  {cnt} samples ({cnt/len(samples)*100:.1f}%)")
    # Print truncated bc
    print(f"    BC: {bc[:100]}...")
print()

# Check sensor configurations
sensor_shapes = Counter([sample['sensors_xy'].shape for sample in samples])
print("=== Sensor Configuration Analysis ===")
for shape, cnt in sorted(sensor_shapes.items()):
    print(f"  sensors_xy shape {shape}: {cnt} samples")
print()

# Combined clustering analysis
# Try clustering by (kappa, T0, n_sources)
def cluster_key(s):
    meta = s['sample_metadata']
    return (meta['kappa'], meta['T0'], s['n_sources'])

clusters = Counter([cluster_key(s) for s in samples])

print("=== Combined Clustering (kappa, T0, n_sources) ===")
print(f"Number of unique clusters: {len(clusters)}")
print()
cluster_sizes = sorted(clusters.values(), reverse=True)
print(f"Cluster sizes: {cluster_sizes}")
print()

# If there are many small clusters, Gappy C-POD won't work
# POD typically needs 10+ snapshots per cluster for a good basis
MIN_CLUSTER_SIZE = 5  # Minimum samples per cluster for reliable POD
viable_clusters = sum(1 for size in cluster_sizes if size >= MIN_CLUSTER_SIZE)
samples_in_viable = sum(size for size in cluster_sizes if size >= MIN_CLUSTER_SIZE)

print(f"Clusters with >= {MIN_CLUSTER_SIZE} samples: {viable_clusters}")
print(f"Samples in viable clusters: {samples_in_viable} / {len(samples)} ({samples_in_viable/len(samples)*100:.1f}%)")
print()

# Try simpler clustering by just kappa
print("=== Clustering by KAPPA ONLY ===")
kappa_cluster_sizes = sorted(kappa_counts.values(), reverse=True)
viable_kappa = sum(1 for size in kappa_cluster_sizes if size >= MIN_CLUSTER_SIZE)
samples_in_viable_kappa = sum(size for size in kappa_cluster_sizes if size >= MIN_CLUSTER_SIZE)
print(f"Clusters with >= {MIN_CLUSTER_SIZE} samples: {viable_kappa}")
print(f"Samples in viable clusters: {samples_in_viable_kappa} / {len(samples)} ({samples_in_viable_kappa/len(samples)*100:.1f}%)")
print()

# Final assessment
print("=" * 50)
print("=== FEASIBILITY ASSESSMENT ===")
print("=" * 50)
print()

issues = []
if len(unique_kappas) > len(samples) // 10:
    issues.append(f"CRITICAL: Kappa is nearly continuous ({len(unique_kappas)} unique values for {len(samples)} samples)")
if len(clusters) > len(samples) // 5:
    issues.append(f"CRITICAL: Too many unique (kappa, T0, n_sources) combinations ({len(clusters)} clusters)")
if samples_in_viable < len(samples) * 0.5:
    issues.append(f"CRITICAL: Only {samples_in_viable/len(samples)*100:.1f}% samples in viable clusters")

if issues:
    print("ABORT CRITERIA MET:")
    for issue in issues:
        print(f"  - {issue}")
    print()
    print("CONCLUSION: Gappy C-POD is NOT VIABLE for this problem.")
    print("Reason: Kappa variation is too continuous for effective clustering.")
    print("Each sample would need its own POD basis, defeating the purpose.")
else:
    print("Clustering appears feasible. Proceed with Gappy C-POD implementation.")

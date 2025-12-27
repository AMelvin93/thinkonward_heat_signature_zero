#!/usr/bin/env python
"""Check nt distribution in test data."""
import pickle
import os
from collections import Counter

data_path = os.path.join(
    os.path.dirname(__file__), '..', 'data', 'heat-signature-zero-test-data.pkl'
)

with open(data_path, 'rb') as f:
    data = pickle.load(f)

samples = data['samples']
meta = data['meta']

print(f"dt = {meta['dt']}")
print(f"Total samples: {len(samples)}")

nt_values = [s['sample_metadata']['nt'] for s in samples]
nt_counter = Counter(nt_values)

print(f"\nnt distribution:")
for nt, count in sorted(nt_counter.items()):
    print(f"  nt={nt}: {count} samples")

print(f"\nnt range: {min(nt_values)} - {max(nt_values)}")
print(f"Mean nt: {sum(nt_values)/len(nt_values):.1f}")

# Also check n_sources distribution
n_sources = [s['n_sources'] for s in samples]
print(f"\nn_sources distribution:")
for ns, count in sorted(Counter(n_sources).items()):
    print(f"  {ns} source(s): {count} samples ({100*count/len(samples):.0f}%)")

#!/usr/bin/env python
"""Quick JAX GPU diagnostic test."""
import pickle
import time
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Load data
with open(project_root / 'data/heat-signature-zero-test-data.pkl', 'rb') as f:
    data = pickle.load(f)

# Check if nt varies across samples
nts = [s['sample_metadata']['nt'] for s in data['samples'][:10]]
print(f'nt values in first 10 samples: {nts}')
print(f'Unique nt values: {set(nts)}')

# Quick single-sample JAX test
from src.jax_optimizer import JAXOptimizer
opt = JAXOptimizer()

sample = data['samples'][0]
meta = data['meta']

print(f'\nRunning single sample with 1 restart, 10 iterations...')
start = time.time()
result = opt.estimate_sources(sample, meta, n_restarts=1, max_iter=10, verbose=True)
print(f'Time: {time.time() - start:.1f}s')
print(f'RMSE: {result[1]:.4f}')

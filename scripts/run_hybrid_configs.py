#!/usr/bin/env python
"""Generate 2 hybrid submission .npz files for last-minute submissions."""

import os
import sys
import pickle
import time
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'experiments' / 'tighter_sigma_range'))

from optimizer import TabuBasinHoppingOptimizer
from src.seed_manager import SeedManager

LAMBDA, N_MAX = 0.3, 3


def score_comp(rmses):
    n = len(rmses)
    if n == 0:
        return 0.0
    return sum(1.0 / (1.0 + r) for r in rmses) / n + LAMBDA * (n / N_MAX)


def process_sample(args):
    idx, sample, meta, config, seed = args
    np.random.seed(seed)
    opt = TabuBasinHoppingOptimizer(**{**config, 'seed': seed})
    try:
        start = time.time()
        cands, best_rmse, results, n_sims = opt.estimate_sources(
            sample, meta, q_range=(0.5, 2.0), verbose=False
        )
        elapsed = time.time() - start
        sources, rmses = [], []
        for i, c in enumerate(cands):
            sources.append([(float(x), float(y), float(q)) for x, y, q in c])
            rmses.append(float(results[i].rmse) if i < len(results) else float(best_rmse))
        return {
            'idx': idx,
            'sample_id': sample.get('sample_id', f'sample_{idx}'),
            'sources': sources,
            'rmses': rmses,
            'best_rmse': float(best_rmse),
            'score': score_comp(rmses),
            'time': elapsed,
            'ok': True,
            'n_src': sample['n_sources'],
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            'idx': idx,
            'sample_id': sample.get('sample_id', f'sample_{idx}'),
            'sources': [],
            'rmses': [],
            'best_rmse': 999,
            'score': 0,
            'time': 0,
            'ok': False,
            'n_src': 0,
        }


CONFIGS = [
    {
        'name': 'hybrid_4pert_sigma016_020_scale006',
        'seed': 42,
        'opt': {
            'sigma0_1src': 0.16, 'sigma0_2src': 0.20,
            'max_fevals_1src': 20, 'max_fevals_2src': 44,
            'timestep_fraction': 0.40, 'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 4, 'perturbation_scale': 0.06,
            'perturb_nm_iters': 2, 'tabu_distance': 0.04,
            'max_tabu_attempts': 10,
        },
    },
    {
        'name': '4pert_nm2_scale006_highfevals',
        'seed': 42,
        'opt': {
            'sigma0_1src': 0.18, 'sigma0_2src': 0.22,
            'max_fevals_1src': 26, 'max_fevals_2src': 48,
            'timestep_fraction': 0.40, 'refine_maxiter': 8,
            'enable_tabu_hopping': True,
            'n_perturbations': 4, 'perturbation_scale': 0.06,
            'perturb_nm_iters': 2, 'tabu_distance': 0.04,
            'max_tabu_attempts': 10,
        },
    },
]


def main():
    data_path = project_root / 'data' / 'heat-signature-zero-test-data.pkl'
    print(f"Loading data from {data_path}...")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    samples, meta = data['samples'], data['meta']
    n_workers = os.cpu_count()
    outdir = project_root / 'submissions' / 'final5'
    outdir.mkdir(parents=True, exist_ok=True)

    for cfg in CONFIGS:
        name = cfg['name']
        sm = SeedManager(master_seed=cfg['seed'])
        np.random.seed(cfg['seed'])
        work = [(i, samples[i], meta, cfg['opt'], sm.get_sample_seed(i)) for i in range(80)]

        print(f'\n{"="*60}')
        print(f'  {name} (seed={cfg["seed"]})')
        print(f'  Workers: {n_workers}')
        print(f'{"="*60}')

        t0 = time.time()
        res = []
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(process_sample, w): w[0] for w in work}
            for f in as_completed(futs):
                r = f.result()
                res.append(r)
                print(f'  [{len(res):3d}/80] Sample {r["idx"]:3d}: '
                      f'{r["n_src"]}-src RMSE={r["best_rmse"]:.4f} '
                      f'cands={len(r["sources"])} '
                      f'score={r["score"]:.4f} t={r["time"]:.1f}s')

        elapsed = time.time() - t0
        res.sort(key=lambda r: r['idx'])

        scores = [r['score'] for r in res if r['ok']]
        avg_score = float(np.mean(scores))
        proj = (elapsed / 80) * 400 / 60
        avg_cands = float(np.mean([len(r['sources']) for r in res if r['ok']]))

        # Save .npz
        sub = [{'sample_id': r['sample_id'], 'estimated_sources': r['sources']} for r in res]
        npz_path = outdir / f'{name}.npz'
        np.savez(str(npz_path), samples=sub)

        budget = "IN BUDGET" if proj <= 60 else f"OVER by {proj-60:.1f}m"
        print(f'\n  {name} DONE')
        print(f'  Actual score: {avg_score:.4f}')
        print(f'  Time: {elapsed/60:.1f}min -> projected 400: {proj:.1f}min [{budget}]')
        print(f'  Avg candidates: {avg_cands:.2f}')
        print(f'  Saved: {npz_path}')

    print(f'\n{"="*60}')
    print('Both configs complete!')
    print(f'{"="*60}')


if __name__ == '__main__':
    main()

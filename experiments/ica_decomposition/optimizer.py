"""
ICA Decomposition Optimizer for Heat Source Identification.

Key Innovation: Uses Independent Component Analysis (ICA) to decompose
2-source sensor signals into individual source contributions, providing
much better initialization for position optimization.

Physics Insight:
    The heat equation is LINEAR - temperature fields from multiple sources ADD:
    T_total = T_source1 + T_source2

    ICA can decompose the mixed signals to extract:
    1. Individual source temporal signatures
    2. Spatial mixing coefficients (which encode position information!)

    The mixing matrix A in Y = A @ S tells us how much each sensor "sees"
    each source - this IS spatial information about source locations!
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from sklearn.decomposition import FastICA
import cma

# Add project root to path
_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

from src.triangulation import triangulation_init

# Add simulator path
sys.path.insert(0, os.path.join(_project_root, 'data', 'Heat_Signature_zero-starter_notebook'))
from simulator import Heat2D


# Competition parameters
N_MAX = 3
TAU = 0.2
SCALE_FACTORS = (2.0, 1.0, 2.0)


@dataclass
class CandidateResult:
    """Result from optimization."""
    params: np.ndarray  # Full params including computed q
    rmse: float
    init_type: str
    n_evals: int


def normalize_sources(sources: List[Tuple[float, float, float]]) -> np.ndarray:
    """Normalize source parameters using scale factors."""
    return np.array([[x/SCALE_FACTORS[0], y/SCALE_FACTORS[1], q/SCALE_FACTORS[2]]
                     for x, y, q in sources])


def candidate_distance(sources1: List[Tuple], sources2: List[Tuple]) -> float:
    """Compute minimum distance between two candidate sets."""
    norm1 = normalize_sources(sources1)
    norm2 = normalize_sources(sources2)
    n = len(sources1)

    if n != len(sources2):
        return float('inf')

    if n == 1:
        return np.linalg.norm(norm1[0] - norm2[0])

    from itertools import permutations
    min_total = float('inf')
    for perm in permutations(range(n)):
        total = sum(np.linalg.norm(norm1[i] - norm2[j])**2 for i, j in enumerate(perm))
        min_total = min(min_total, np.sqrt(total / n))

    return min_total


def filter_dissimilar(candidates: List[Tuple], tau: float = TAU, n_max: int = N_MAX) -> List[Tuple]:
    """Filter to keep only dissimilar candidates."""
    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda x: x[1])
    kept = [candidates[0]]

    for cand in candidates[1:]:
        if all(candidate_distance(cand[0], k[0]) >= tau for k in kept):
            kept.append(cand)
            if len(kept) >= n_max:
                break

    return kept


def extract_enhanced_features(sample: Dict, meta: Dict = None) -> np.ndarray:
    """Enhanced feature extraction for similarity matching (11 features)."""
    Y = sample['Y_noisy']
    sensors = np.array(sample['sensors_xy'])
    kappa = sample['sample_metadata']['kappa']

    # Basic features
    basic = [
        np.max(Y) / 10.0,
        np.mean(Y) / 5.0,
        np.std(Y) / 2.0,
        kappa * 10,
        len(sensors) / 10.0,
    ]

    # Spatial features
    max_temps_per_sensor = Y.max(axis=0)
    weights = max_temps_per_sensor / (max_temps_per_sensor.sum() + 1e-8)
    centroid_x = np.average(sensors[:, 0], weights=weights) / 2.0
    centroid_y = np.average(sensors[:, 1], weights=weights)
    spatial_spread = np.sqrt(
        np.average((sensors[:, 0] / 2.0 - centroid_x)**2 +
                   (sensors[:, 1] - centroid_y)**2, weights=weights)
    )
    spatial = [centroid_x, centroid_y, spatial_spread]

    # Temporal features
    onset_times = []
    for i in range(Y.shape[1]):
        signal = Y[:, i]
        threshold = 0.1 * (signal.max() + 1e-8)
        onset_idx = np.argmax(signal > threshold)
        onset_times.append(onset_idx)
    onset_times = np.array(onset_times)
    onset_mean = np.mean(onset_times) / 100.0
    onset_std = np.std(onset_times) / 50.0
    temporal = [onset_mean, onset_std]

    # Correlation feature
    if Y.shape[1] > 1:
        try:
            corr_matrix = np.corrcoef(Y.T)
            triu_indices = np.triu_indices_from(corr_matrix, k=1)
            correlations = corr_matrix[triu_indices]
            correlations = correlations[~np.isnan(correlations)]
            avg_corr = np.mean(correlations) if len(correlations) > 0 else 0.5
        except:
            avg_corr = 0.5
    else:
        avg_corr = 1.0
    correlation = [avg_corr]

    return np.array(basic + spatial + temporal + correlation)


def find_similar_solutions(
    features: np.ndarray,
    history: List[Tuple[np.ndarray, np.ndarray]],
    k: int = 1
) -> List[np.ndarray]:
    """Find k most similar solutions from history (positions only)."""
    if not history or k == 0:
        return []

    distances = [(np.linalg.norm(features - h_feat), h_sol) for h_feat, h_sol in history]
    distances.sort(key=lambda x: x[0])
    return [sol.copy() for _, sol in distances[:k]]


def simulate_unit_source(x: float, y: float, solver: Heat2D, dt: float, nt: int,
                         T0: float, sensors_xy: np.ndarray) -> np.ndarray:
    """Simulate a source with q=1.0 and return sensor readings."""
    sources = [{'x': x, 'y': y, 'q': 1.0}]
    times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources)
    Y_unit = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])
    return Y_unit


def compute_optimal_intensity_1src(
    x: float, y: float, Y_observed: np.ndarray,
    solver: Heat2D, dt: float, nt: int, T0: float, sensors_xy: np.ndarray,
    q_range: Tuple[float, float] = (0.5, 2.0)
) -> Tuple[float, np.ndarray, float]:
    """Compute optimal intensity for 1-source analytically."""
    Y_unit = simulate_unit_source(x, y, solver, dt, nt, T0, sensors_xy)
    Y_unit_flat = Y_unit.flatten()
    Y_obs_flat = Y_observed.flatten()

    denominator = np.dot(Y_unit_flat, Y_unit_flat)
    if denominator < 1e-10:
        q_optimal = 1.0
    else:
        numerator = np.dot(Y_unit_flat, Y_obs_flat)
        q_optimal = numerator / denominator

    q_optimal = np.clip(q_optimal, q_range[0], q_range[1])
    Y_pred = q_optimal * Y_unit
    rmse = np.sqrt(np.mean((Y_pred - Y_observed) ** 2))

    return q_optimal, Y_pred, rmse


def compute_optimal_intensity_2src(
    x1: float, y1: float, x2: float, y2: float, Y_observed: np.ndarray,
    solver: Heat2D, dt: float, nt: int, T0: float, sensors_xy: np.ndarray,
    q_range: Tuple[float, float] = (0.5, 2.0)
) -> Tuple[Tuple[float, float], np.ndarray, float]:
    """Compute optimal intensities for 2-source analytically using linearity."""
    Y1 = simulate_unit_source(x1, y1, solver, dt, nt, T0, sensors_xy)
    Y2 = simulate_unit_source(x2, y2, solver, dt, nt, T0, sensors_xy)

    Y1_flat = Y1.flatten()
    Y2_flat = Y2.flatten()
    Y_obs_flat = Y_observed.flatten()

    A = np.array([
        [np.dot(Y1_flat, Y1_flat), np.dot(Y1_flat, Y2_flat)],
        [np.dot(Y2_flat, Y1_flat), np.dot(Y2_flat, Y2_flat)]
    ])
    b = np.array([np.dot(Y1_flat, Y_obs_flat), np.dot(Y2_flat, Y_obs_flat)])

    try:
        A_reg = A + 1e-6 * np.eye(2)
        q1, q2 = np.linalg.solve(A_reg, b)
    except np.linalg.LinAlgError:
        q1, q2 = 1.0, 1.0

    q1 = np.clip(q1, q_range[0], q_range[1])
    q2 = np.clip(q2, q_range[0], q_range[1])

    Y_pred = q1 * Y1 + q2 * Y2
    rmse = np.sqrt(np.mean((Y_pred - Y_observed) ** 2))

    return (q1, q2), Y_pred, rmse


# =============================================================================
# ICA DECOMPOSITION - THE KEY INNOVATION
# =============================================================================

def ica_decompose_2source(
    Y_obs: np.ndarray,
    sensors_xy: np.ndarray,
    Lx: float = 2.0,
    Ly: float = 1.0,
    margin: float = 0.05
) -> Tuple[Optional[np.ndarray], Dict]:
    """
    Decompose 2-source sensor signals using ICA to estimate source positions.

    The heat equation is LINEAR, so:
        Y_observed = Y_source1 + Y_source2

    ICA decomposes: Y = A @ S where:
        - S: source signals (temporal patterns of each source)
        - A: mixing matrix (spatial coefficients - how much each sensor sees each source)

    The mixing matrix A encodes spatial information:
        - Each column corresponds to a source
        - Each row shows how strongly that sensor "sees" that source
        - Sensors closer to a source have higher mixing coefficients

    We estimate source positions from the mixing matrix using weighted centroids.

    Args:
        Y_obs: Observed sensor readings (time, n_sensors)
        sensors_xy: Sensor positions (n_sensors, 2)
        Lx, Ly: Domain dimensions
        margin: Margin from boundaries

    Returns:
        positions: Array [x1, y1, x2, y2] or None if decomposition fails
        info: Dictionary with ICA diagnostics
    """
    info = {
        'success': False,
        'method': 'ica',
        'n_components': 2,
        'separation_quality': 0.0,
    }

    if Y_obs.shape[1] < 2:
        info['error'] = 'Not enough sensors for ICA'
        return None, info

    try:
        # Apply ICA to decompose mixed signals
        # Y_obs.T has shape (n_sensors, time) - each sensor is a "signal"
        # We want to find 2 independent temporal components
        ica = FastICA(n_components=2, random_state=42, max_iter=500, tol=1e-4)

        # Transpose: ICA expects (n_samples, n_features) = (time, n_sensors)
        # We want to extract 2 source signals from n_sensors mixed signals
        S = ica.fit_transform(Y_obs)  # (time, 2) - estimated source signals
        A = ica.mixing_  # (n_sensors, 2) - mixing matrix

        info['success'] = True

        # Measure separation quality using kurtosis (higher = more non-Gaussian = better separation)
        kurtosis_vals = []
        for i in range(2):
            s = S[:, i]
            s_norm = (s - s.mean()) / (s.std() + 1e-8)
            kurt = np.mean(s_norm**4) - 3  # Excess kurtosis
            kurtosis_vals.append(abs(kurt))
        info['separation_quality'] = np.mean(kurtosis_vals)

        # === EXTRACT POSITIONS FROM MIXING MATRIX ===
        # Each column of A shows how much each sensor "sees" that source
        # Use weighted centroid where weights = abs(mixing coefficients)

        positions = []
        for i in range(2):
            # Get mixing coefficients for this source (all sensors)
            mixing_coeffs = np.abs(A[:, i])

            # Normalize to get weights
            weights = mixing_coeffs / (mixing_coeffs.sum() + 1e-8)

            # Weighted centroid gives estimated position
            x_est = np.average(sensors_xy[:, 0], weights=weights)
            y_est = np.average(sensors_xy[:, 1], weights=weights)

            # Clip to valid domain
            x_est = np.clip(x_est, margin * Lx, (1 - margin) * Lx)
            y_est = np.clip(y_est, margin * Ly, (1 - margin) * Ly)

            positions.extend([x_est, y_est])

        return np.array(positions), info

    except Exception as e:
        info['error'] = str(e)
        return None, info


def nmf_decompose_2source(
    Y_obs: np.ndarray,
    sensors_xy: np.ndarray,
    Lx: float = 2.0,
    Ly: float = 1.0,
    margin: float = 0.05
) -> Tuple[Optional[np.ndarray], Dict]:
    """
    Alternative: Non-negative Matrix Factorization (NMF) for decomposition.

    NMF is useful because temperature readings are non-negative,
    which matches the physical constraint of heat sources.

    Y ≈ W @ H where both W and H have non-negative entries.
    """
    from sklearn.decomposition import NMF

    info = {
        'success': False,
        'method': 'nmf',
        'n_components': 2,
    }

    if Y_obs.shape[1] < 2:
        info['error'] = 'Not enough sensors for NMF'
        return None, info

    try:
        # Ensure non-negative input
        Y_pos = np.maximum(Y_obs, 0)

        nmf = NMF(n_components=2, random_state=42, max_iter=500)
        W = nmf.fit_transform(Y_pos)  # (time, 2) - temporal patterns
        H = nmf.components_  # (2, n_sensors) - spatial patterns

        info['success'] = True
        info['reconstruction_error'] = nmf.reconstruction_err_

        # H has shape (2, n_sensors) - each row is a source's spatial pattern
        positions = []
        for i in range(2):
            spatial_pattern = H[i, :]  # How this source affects each sensor

            weights = spatial_pattern / (spatial_pattern.sum() + 1e-8)

            x_est = np.average(sensors_xy[:, 0], weights=weights)
            y_est = np.average(sensors_xy[:, 1], weights=weights)

            x_est = np.clip(x_est, margin * Lx, (1 - margin) * Lx)
            y_est = np.clip(y_est, margin * Ly, (1 - margin) * Ly)

            positions.extend([x_est, y_est])

        return np.array(positions), info

    except Exception as e:
        info['error'] = str(e)
        return None, info


class ICADecompositionOptimizer:
    """
    CMA-ES optimizer with ICA-based initialization for 2-source problems.

    Key Innovation: Uses ICA to decompose 2-source signals and extract
    position estimates, providing much better initialization than
    hottest-sensor or triangulation alone.

    For 1-source problems, falls back to triangulation/smart init.
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 50,
        max_fevals_1src: int = 15,
        max_fevals_2src: int = 20,
        sigma0_1src: float = 0.15,
        sigma0_2src: float = 0.20,
        use_triangulation: bool = True,
        use_ica: bool = True,
        use_nmf: bool = False,  # Alternative decomposition
        ica_replaces_triangulation: bool = False,  # If True, use ICA instead of triangulation for 2-src
        n_candidates: int = N_MAX,
        candidate_pool_size: int = 10,
        k_similar: int = 1,
        use_enhanced_features: bool = True,
    ):
        """Initialize the ICA decomposition optimizer."""
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.max_fevals_1src = max_fevals_1src
        self.max_fevals_2src = max_fevals_2src
        self.sigma0_1src = sigma0_1src
        self.sigma0_2src = sigma0_2src
        self.use_triangulation = use_triangulation
        self.use_ica = use_ica
        self.use_nmf = use_nmf
        self.ica_replaces_triangulation = ica_replaces_triangulation
        self.n_candidates = min(n_candidates, N_MAX)
        self.candidate_pool_size = candidate_pool_size
        self.k_similar = k_similar
        self.use_enhanced_features = use_enhanced_features

    def _create_solver(self, kappa: float, bc: str) -> Heat2D:
        """Create a Heat2D solver instance."""
        return Heat2D(self.Lx, self.Ly, self.nx, self.ny, kappa, bc=bc)

    def _get_position_bounds(self, n_sources: int, margin: float = 0.05):
        """Get bounds for POSITION parameters only (no intensity)."""
        lb, ub = [], []
        for _ in range(n_sources):
            lb.extend([margin * self.Lx, margin * self.Ly])
            ub.extend([(1 - margin) * self.Lx, (1 - margin) * self.Ly])
        return lb, ub

    def _smart_init_positions(self, sample: Dict, n_sources: int) -> np.ndarray:
        """Get position-only initialization from hottest sensors."""
        readings = sample['Y_noisy']
        sensors = sample['sensors_xy']
        avg_temps = np.mean(readings, axis=0)
        hot_idx = np.argsort(avg_temps)[::-1]

        selected = []
        for idx in hot_idx:
            if len(selected) >= n_sources:
                break
            if all(np.linalg.norm(sensors[idx] - sensors[p]) >= 0.25 for p in selected):
                selected.append(idx)

        while len(selected) < n_sources:
            for idx in hot_idx:
                if idx not in selected:
                    selected.append(idx)
                    break

        params = []
        for idx in selected:
            x, y = sensors[idx]
            params.extend([x, y])

        return np.array(params)

    def _triangulation_init_positions(self, sample: Dict, meta: Dict,
                                       n_sources: int, q_range: Tuple[float, float]) -> Optional[np.ndarray]:
        """Get position-only initialization from triangulation."""
        if not self.use_triangulation:
            return None
        try:
            full_init = triangulation_init(sample, meta, n_sources, q_range, self.Lx, self.Ly)
            positions = []
            for i in range(n_sources):
                positions.extend([full_init[i*3], full_init[i*3 + 1]])
            return np.array(positions)
        except Exception:
            return None

    def _ica_init_positions(self, sample: Dict) -> Tuple[Optional[np.ndarray], Dict]:
        """Get position-only initialization from ICA decomposition (2-source only)."""
        if not self.use_ica or sample['n_sources'] != 2:
            return None, {'success': False, 'reason': 'ICA only for 2-source'}

        Y_obs = sample['Y_noisy']
        sensors_xy = np.array(sample['sensors_xy'])

        return ica_decompose_2source(Y_obs, sensors_xy, self.Lx, self.Ly)

    def _nmf_init_positions(self, sample: Dict) -> Tuple[Optional[np.ndarray], Dict]:
        """Get position-only initialization from NMF decomposition (2-source only)."""
        if not self.use_nmf or sample['n_sources'] != 2:
            return None, {'success': False, 'reason': 'NMF only for 2-source'}

        Y_obs = sample['Y_noisy']
        sensors_xy = np.array(sample['sensors_xy'])

        return nmf_decompose_2source(Y_obs, sensors_xy, self.Lx, self.Ly)

    def estimate_sources(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        history_1src: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        history_2src: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        verbose: bool = False,
    ) -> Tuple[List[List[Tuple]], float, List[CandidateResult], np.ndarray, np.ndarray, int, Dict]:
        """
        Estimate sources with ICA-enhanced initialization.

        Returns additional ICA info dict compared to base optimizer.
        """
        n_sources = sample['n_sources']
        sensors_xy = np.array(sample['sensors_xy'])
        Y_observed = sample['Y_noisy']

        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        T0 = sample['sample_metadata']['T0']

        solver = self._create_solver(kappa, bc)

        # Feature extraction
        if self.use_enhanced_features:
            features = extract_enhanced_features(sample, meta)
        else:
            features = extract_enhanced_features(sample, meta)[:5]

        # Transfer learning
        history = history_1src if n_sources == 1 else history_2src
        if history is None:
            history = []
        similar_solutions = find_similar_solutions(features, history, k=self.k_similar)
        n_transferred = len(similar_solutions)

        ica_info = {'used': False, 'success': False}

        # === OBJECTIVE FUNCTION WITH ANALYTICAL INTENSITY ===
        n_sims = [0]

        if n_sources == 1:
            def objective(xy_params):
                x, y = xy_params
                n_sims[0] += 1
                q, Y_pred, rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
                return rmse
        else:
            def objective(xy_params):
                x1, y1, x2, y2 = xy_params
                n_sims[0] += 2
                (q1, q2), Y_pred, rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
                return rmse

        # === BUILD INITIALIZATION POOL ===
        initializations = []

        # 1. ICA decomposition (2-source only) - THE KEY INNOVATION
        if n_sources == 2 and self.use_ica:
            ica_init, ica_info = self._ica_init_positions(sample)
            if ica_init is not None:
                initializations.append((ica_init, 'ica'))
                if verbose:
                    print(f"  ICA init: success={ica_info.get('success')}, "
                          f"quality={ica_info.get('separation_quality', 0):.3f}")

            # Also try NMF as alternative
            if self.use_nmf:
                nmf_init, nmf_info = self._nmf_init_positions(sample)
                if nmf_init is not None:
                    initializations.append((nmf_init, 'nmf'))

        # 2. Triangulation - skip for 2-source if ICA replaces it
        use_tri_for_this_sample = self.use_triangulation
        if n_sources == 2 and self.ica_replaces_triangulation and self.use_ica:
            use_tri_for_this_sample = False  # ICA replaces triangulation for 2-source

        if use_tri_for_this_sample:
            tri_init = self._triangulation_init_positions(sample, meta, n_sources, q_range)
            if tri_init is not None:
                initializations.append((tri_init, 'triangulation'))

        # 3. Hottest sensor
        smart_init = self._smart_init_positions(sample, n_sources)
        initializations.append((smart_init, 'smart'))

        # 4. Transfer learning
        for i, sol in enumerate(similar_solutions):
            lb, ub = self._get_position_bounds(n_sources)
            sol_clipped = np.clip(sol, lb, ub)
            initializations.append((sol_clipped, f'transfer_{i}'))

        if verbose:
            print(f"  Initializations: {[init[1] for init in initializations]}")

        # === CMA-ES OPTIMIZATION ===
        max_fevals = self.max_fevals_1src if n_sources == 1 else self.max_fevals_2src
        sigma0 = self.sigma0_1src if n_sources == 1 else self.sigma0_2src
        lb, ub = self._get_position_bounds(n_sources)

        fevals_per_init = max(5, max_fevals // len(initializations))

        all_solutions = []

        for init_params, init_type in initializations:
            opts = cma.CMAOptions()
            opts['maxfevals'] = fevals_per_init
            opts['bounds'] = [lb, ub]
            opts['verbose'] = -9
            opts['tolfun'] = 1e-6
            opts['tolx'] = 1e-6

            es = cma.CMAEvolutionStrategy(init_params.tolist(), sigma0, opts)

            while not es.stop():
                solutions = es.ask()
                fitness = [objective(s) for s in solutions]
                es.tell(solutions, fitness)

                for sol, fit in zip(solutions, fitness):
                    all_solutions.append((np.array(sol), fit, init_type))

        # === CONVERT TO FULL PARAMS ===
        all_solutions.sort(key=lambda x: x[1])
        top_solutions = all_solutions[:self.candidate_pool_size]

        candidates_raw = []
        for pos_params, rmse, init_type in top_solutions:
            if n_sources == 1:
                x, y = pos_params
                q, _, final_rmse = compute_optimal_intensity_1src(
                    x, y, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
                full_params = np.array([x, y, q])
                sources = [(float(x), float(y), float(q))]
            else:
                x1, y1, x2, y2 = pos_params
                (q1, q2), _, final_rmse = compute_optimal_intensity_2src(
                    x1, y1, x2, y2, Y_observed, solver, dt, nt, T0, sensors_xy, q_range)
                full_params = np.array([x1, y1, q1, x2, y2, q2])
                sources = [(float(x1), float(y1), float(q1)),
                          (float(x2), float(y2), float(q2))]

            candidates_raw.append((sources, full_params, final_rmse, init_type))

        # Dissimilarity filtering
        filtered = filter_dissimilar([(c[0], c[2]) for c in candidates_raw], tau=TAU)

        # Match filtered back to full candidates
        final_candidates = []
        for sources, rmse in filtered:
            for c in candidates_raw:
                if c[0] == sources and abs(c[2] - rmse) < 1e-10:
                    final_candidates.append(c)
                    break

        # === BUILD RESULTS ===
        candidate_sources = [c[0] for c in final_candidates]
        candidate_rmses = [c[2] for c in final_candidates]
        best_rmse = min(candidate_rmses) if candidate_rmses else float('inf')

        # Best params for history
        if final_candidates:
            best_idx = np.argmin([c[2] for c in final_candidates])
            best_full_params = final_candidates[best_idx][1]
            if n_sources == 1:
                best_positions = best_full_params[:2]
            else:
                best_positions = np.array([best_full_params[0], best_full_params[1],
                                           best_full_params[3], best_full_params[4]])
        else:
            best_positions = smart_init

        n_evals = n_sims[0]

        results = [
            CandidateResult(
                params=c[1],
                rmse=c[2],
                init_type=c[3],
                n_evals=n_evals // len(final_candidates) if final_candidates else n_evals
            )
            for c in final_candidates
        ]

        # Track which init type won
        if final_candidates:
            best_init_type = final_candidates[np.argmin([c[2] for c in final_candidates])][3]
            ica_info['best_init'] = best_init_type
            ica_info['used'] = n_sources == 2 and self.use_ica

        return candidate_sources, best_rmse, results, features, best_positions, n_transferred, ica_info

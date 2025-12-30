"""
Direct Solution Optimizer for Heat Source Identification.

Paradigm shift: Instead of iterative optimization (Guess → Simulate → Compare → Repeat),
exploit physics structure to get DIRECT solutions:

1-source: Geometric trilateration from sensor onset times (microseconds)
2-source: ICA signal decomposition + trilateration (milliseconds)

Then use minimal CMA-ES polish if needed.

This approach could be 10-100x faster than iterative methods.
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from sklearn.decomposition import FastICA, NMF
from itertools import permutations

# Add project root to path for imports
_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)

# Add the starter notebook path to import the simulator
sys.path.insert(0, os.path.join(_project_root, 'data', 'Heat_Signature_zero-starter_notebook'))
from simulator import Heat2D

# Import our proven triangulation
from src.triangulation import triangulation_init


@dataclass
class CandidateResult:
    """Result from optimization."""
    params: np.ndarray
    rmse: float
    init_type: str
    n_evals: int


# Competition parameters for dissimilarity filtering
N_MAX = 3
TAU = 0.2
SCALE_FACTORS = (2.0, 1.0, 2.0)  # (Lx, Ly, q_max)


def normalize_sources(sources: List[Tuple[float, float, float]]) -> np.ndarray:
    """Normalize source parameters using scale factors."""
    normalized = []
    for x, y, q in sources:
        normalized.append([
            x / SCALE_FACTORS[0],
            y / SCALE_FACTORS[1],
            q / SCALE_FACTORS[2],
        ])
    return np.array(normalized)


def candidate_distance(sources1: List[Tuple], sources2: List[Tuple]) -> float:
    """Compute minimum distance between two candidate source sets."""
    norm1 = normalize_sources(sources1)
    norm2 = normalize_sources(sources2)

    n = len(sources1)
    if n != len(sources2):
        return float('inf')

    if n == 1:
        return np.linalg.norm(norm1[0] - norm2[0])

    # For 2-source: try both permutations
    min_total = float('inf')
    for perm in permutations(range(n)):
        total = 0
        for i, j in enumerate(perm):
            total += np.linalg.norm(norm1[i] - norm2[j]) ** 2
        total = np.sqrt(total / n)
        min_total = min(min_total, total)

    return min_total


def filter_dissimilar(candidates: List[Tuple], tau: float = TAU, n_max: int = N_MAX) -> List[Tuple]:
    """Filter candidates to keep only dissimilar ones."""
    if not candidates:
        return []

    # Sort by RMSE first
    candidates = sorted(candidates, key=lambda x: x[1])

    kept = [candidates[0]]

    for cand in candidates[1:]:
        is_similar = False
        for kept_cand in kept:
            dist = candidate_distance(cand[0], kept_cand[0])
            if dist < tau:
                is_similar = True
                break
        if not is_similar:
            kept.append(cand)
            if len(kept) >= n_max:
                break

    return kept


class DirectSolutionOptimizer:
    """
    Direct solution optimizer using signal processing and geometry.

    For 1-source:
        1. Extract onset times from sensor signals
        2. Convert to distances via diffusion physics: r = sqrt(4*kappa*t)
        3. Solve overdetermined trilateration system via least squares
        4. Compute analytical intensity
        5. Optional: short CMA-ES polish

    For 2-source:
        1. Use ICA to decompose mixed sensor signals into source components
        2. Extract spatial weights from mixing matrix
        3. Estimate positions from weighted sensor centroids
        4. Compute analytical intensities (linear least squares)
        5. Optional: short CMA-ES polish
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 50,
        use_cmaes_polish: bool = True,
        cmaes_polish_fevals: int = 15,
        onset_threshold_fraction: float = 0.05,
        n_candidates: int = 3,
    ):
        """
        Initialize the optimizer.

        Args:
            Lx, Ly: Domain dimensions
            nx, ny: Grid resolution
            use_cmaes_polish: Whether to polish with CMA-ES after direct solution
            cmaes_polish_fevals: Max CMA-ES evaluations for polish
            onset_threshold_fraction: Fraction of max signal for onset detection
            n_candidates: Number of candidates to generate
        """
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.use_cmaes_polish = use_cmaes_polish
        self.cmaes_polish_fevals = cmaes_polish_fevals
        self.onset_threshold_fraction = onset_threshold_fraction
        self.n_candidates = n_candidates

    def _create_solver(self, kappa: float, bc: str) -> Heat2D:
        """Create a Heat2D solver instance."""
        return Heat2D(self.Lx, self.Ly, self.nx, self.ny, kappa, bc=bc)

    def _extract_onset_times(
        self,
        Y: np.ndarray,
        dt: float,
        threshold_fraction: float = None
    ) -> np.ndarray:
        """
        Extract onset times for each sensor.

        Onset = first time when signal exceeds threshold_fraction * max_signal
        """
        if threshold_fraction is None:
            threshold_fraction = self.onset_threshold_fraction

        n_sensors = Y.shape[1]
        onset_times = []

        for i in range(n_sensors):
            signal = Y[:, i]
            max_val = signal.max()

            if max_val < 1e-6:
                # Signal never rises significantly
                onset_times.append(float('inf'))
                continue

            threshold = threshold_fraction * max_val
            onset_idx = np.argmax(signal > threshold)

            if onset_idx == 0 and signal[0] <= threshold:
                # Never crossed threshold
                onset_times.append(float('inf'))
            else:
                onset_times.append(onset_idx * dt)

        return np.array(onset_times)

    def _geometric_trilateration(
        self,
        sensors: np.ndarray,
        distances: np.ndarray,
        bounds: Tuple[Tuple[float, float], Tuple[float, float]] = None
    ) -> Tuple[float, float]:
        """
        Solve source position via overdetermined trilateration.

        Uses distance differences to linearize the problem:
        ||p - s_i||² - ||p - s_0||² = d_i² - d_0²
        2(s_0 - s_i)·p = d_i² - d_0² + ||s_i||² - ||s_0||²

        This is a linear system Ap = b, solved via least squares.
        """
        if bounds is None:
            bounds = ((0.05 * self.Lx, 0.95 * self.Lx),
                      (0.05 * self.Ly, 0.95 * self.Ly))

        # Filter out sensors with infinite distance (no signal)
        valid_mask = np.isfinite(distances)
        if valid_mask.sum() < 3:
            # Not enough sensors, fall back to hottest sensor
            return None, None

        sensors_valid = sensors[valid_mask]
        distances_valid = distances[valid_mask]

        # Build linear system using first sensor as reference
        s0 = sensors_valid[0]
        d0 = distances_valid[0]

        A = []
        b = []

        for i in range(1, len(sensors_valid)):
            si = sensors_valid[i]
            di = distances_valid[i]

            A.append(2 * (s0 - si))
            b.append(di**2 - d0**2 + np.dot(si, si) - np.dot(s0, s0))

        A = np.array(A)
        b = np.array(b)

        if len(A) < 2:
            return None, None

        # Solve via least squares
        try:
            position, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
            x, y = position

            # Clip to bounds
            x = np.clip(x, bounds[0][0], bounds[0][1])
            y = np.clip(y, bounds[1][0], bounds[1][1])

            return x, y
        except:
            return None, None

    def _ica_decomposition(
        self,
        Y: np.ndarray,
        n_components: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose sensor signals into independent source components using ICA.

        Returns:
            source_signals: (n_timesteps, n_components) - temporal pattern of each source
            mixing_matrix: (n_sensors, n_components) - spatial weights for each source
        """
        try:
            # ICA to separate mixed signals
            ica = FastICA(n_components=n_components, random_state=42, max_iter=500)
            source_signals = ica.fit_transform(Y)
            mixing_matrix = ica.mixing_

            return source_signals, mixing_matrix
        except:
            # Fall back to NMF if ICA fails
            try:
                nmf = NMF(n_components=n_components, random_state=42, max_iter=500)
                source_signals = nmf.fit_transform(Y)
                mixing_matrix = nmf.components_.T  # (n_sensors, n_components)

                return source_signals, mixing_matrix
            except:
                return None, None

    def _positions_from_mixing_matrix(
        self,
        sensors: np.ndarray,
        mixing_matrix: np.ndarray
    ) -> List[Tuple[float, float]]:
        """
        Estimate source positions from ICA mixing matrix.

        Each column of mixing matrix shows how strongly each sensor "sees" each source.
        The source position is estimated as the weighted centroid of sensor positions.
        """
        n_sources = mixing_matrix.shape[1]
        positions = []

        for i in range(n_sources):
            weights = np.abs(mixing_matrix[:, i])

            if weights.sum() < 1e-8:
                # Fall back to center
                positions.append((self.Lx / 2, self.Ly / 2))
                continue

            # Normalize weights
            weights = weights / weights.sum()

            # Weighted centroid
            x_est = np.average(sensors[:, 0], weights=weights)
            y_est = np.average(sensors[:, 1], weights=weights)

            # Clip to domain
            x_est = np.clip(x_est, 0.05 * self.Lx, 0.95 * self.Lx)
            y_est = np.clip(y_est, 0.05 * self.Ly, 0.95 * self.Ly)

            positions.append((x_est, y_est))

        return positions

    def _analytical_intensity_1source(
        self,
        x: float,
        y: float,
        sample: Dict,
        meta: Dict,
        solver: Heat2D
    ) -> float:
        """
        Compute optimal intensity analytically for 1-source.

        Since T(sensors, t) = q * T_unit(sensors, t), optimal q is:
        q* = (T_unit · T_observed) / (T_unit · T_unit)
        """
        Y_observed = sample['Y_noisy']
        sensors_xy = sample['sensors_xy']
        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        T0 = sample['sample_metadata']['T0']

        # Simulate with unit intensity
        sources = [{'x': x, 'y': y, 'q': 1.0}]
        times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources)
        Y_unit = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])

        # Linear least squares for q
        numerator = np.dot(Y_unit.flatten(), Y_observed.flatten())
        denominator = np.dot(Y_unit.flatten(), Y_unit.flatten())

        if denominator < 1e-10:
            return 1.0

        q_optimal = numerator / denominator
        return np.clip(q_optimal, 0.5, 2.0)

    def _analytical_intensity_2source(
        self,
        positions: List[Tuple[float, float]],
        sample: Dict,
        meta: Dict,
        solver: Heat2D
    ) -> List[float]:
        """
        Compute optimal intensities for 2-source via linear least squares.

        T_observed ≈ q1 * T_unit1 + q2 * T_unit2
        Solve: [T_unit1, T_unit2] @ [q1, q2]^T = T_observed
        """
        Y_observed = sample['Y_noisy']
        sensors_xy = sample['sensors_xy']
        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        T0 = sample['sample_metadata']['T0']

        # Simulate unit response for each source
        Y_units = []
        for x, y in positions:
            sources = [{'x': x, 'y': y, 'q': 1.0}]
            times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources)
            Y_unit = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])
            Y_units.append(Y_unit.flatten())

        # Build design matrix
        A = np.column_stack(Y_units)
        b = Y_observed.flatten()

        # Solve constrained least squares (q >= 0.5)
        try:
            from scipy.optimize import lsq_linear
            result = lsq_linear(A, b, bounds=(0.5, 2.0))
            q_values = result.x
        except:
            # Fallback to unconstrained + clipping
            q_values, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            q_values = np.clip(q_values, 0.5, 2.0)

        return list(q_values)

    def _cmaes_polish(
        self,
        init_params: np.ndarray,
        sample: Dict,
        meta: Dict,
        solver: Heat2D,
        q_range: Tuple[float, float],
        max_fevals: int = None,
        collect_all: bool = False
    ) -> Tuple[np.ndarray, float, int, List]:
        """
        Short CMA-ES polish to refine direct solution.

        If collect_all=True, also returns all evaluated solutions for candidate extraction.
        """
        import cma

        if max_fevals is None:
            max_fevals = self.cmaes_polish_fevals

        n_sources = sample['n_sources']
        sensors_xy = sample['sensors_xy']
        Y_observed = sample['Y_noisy']
        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        T0 = sample['sample_metadata']['T0']

        def objective(params):
            sources = []
            for i in range(n_sources):
                sources.append({
                    'x': params[i * 3],
                    'y': params[i * 3 + 1],
                    'q': params[i * 3 + 2]
                })
            times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources)
            Y_pred = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])
            return np.sqrt(np.mean((Y_pred - Y_observed) ** 2))

        # Bounds
        margin = 0.05
        lb, ub = [], []
        for _ in range(n_sources):
            lb.extend([margin * self.Lx, margin * self.Ly, q_range[0]])
            ub.extend([(1 - margin) * self.Lx, (1 - margin) * self.Ly, q_range[1]])

        # CMA-ES options
        opts = cma.CMAOptions()
        opts['maxfevals'] = max_fevals
        opts['bounds'] = [lb, ub]
        opts['verbose'] = -9
        opts['tolfun'] = 1e-6

        # Small sigma for polish (already close to solution)
        sigma0 = 0.05

        es = cma.CMAEvolutionStrategy(init_params, sigma0, opts)
        n_evals = 0
        all_solutions = []

        while not es.stop():
            solutions = es.ask()
            fitness = [objective(s) for s in solutions]
            es.tell(solutions, fitness)
            n_evals += len(solutions)

            if collect_all:
                for sol, fit in zip(solutions, fitness):
                    all_solutions.append((np.array(sol), fit))

        return es.result.xbest, es.result.fbest, n_evals, all_solutions

    def _hottest_sensor_fallback(
        self,
        sample: Dict,
        q_range: Tuple[float, float]
    ) -> Tuple[float, float, float]:
        """Fallback initialization from hottest sensor."""
        Y = sample['Y_noisy']
        sensors = sample['sensors_xy']

        avg_temps = np.mean(Y, axis=0)
        hottest_idx = np.argmax(avg_temps)
        x, y = sensors[hottest_idx]

        # Estimate intensity from temperature
        max_temp = avg_temps.max()
        q = 0.5 + (max_temp / (max_temp + 1)) * 1.5
        q = np.clip(q, q_range[0], q_range[1])

        return x, y, q

    def _generate_initializations(
        self,
        sample: Dict,
        meta: Dict,
        solver: Heat2D,
        q_range: Tuple[float, float],
    ) -> List[Tuple[np.ndarray, str]]:
        """
        Generate multiple diverse initializations.

        Returns list of (params, init_type) tuples.
        """
        n_sources = sample['n_sources']
        sensors_xy = sample['sensors_xy']
        Y_observed = sample['Y_noisy']

        initializations = []

        if n_sources == 1:
            # === 1-SOURCE: Multiple initializations ===

            # 1. Triangulation (primary)
            try:
                params = triangulation_init(
                    sample, meta, n_sources, q_range, self.Lx, self.Ly
                )
                initializations.append((params, 'triangulation'))
            except Exception:
                pass

            # 2. Hottest sensor
            x, y, q = self._hottest_sensor_fallback(sample, q_range)
            initializations.append((np.array([x, y, q]), 'hottest_sensor'))

            # 3. Perturbed triangulation (if we have it)
            if initializations and initializations[0][1] == 'triangulation':
                base = initializations[0][0].copy()
                # Add perturbation
                perturbed = base.copy()
                perturbed[0] += np.random.uniform(-0.2, 0.2)  # x
                perturbed[1] += np.random.uniform(-0.1, 0.1)  # y
                perturbed[0] = np.clip(perturbed[0], 0.1, 1.9)
                perturbed[1] = np.clip(perturbed[1], 0.05, 0.95)
                initializations.append((perturbed, 'perturbed'))

        else:
            # === 2-SOURCE: Multiple initializations ===

            # 1. ICA decomposition (primary)
            source_signals, mixing_matrix = self._ica_decomposition(Y_observed, n_components=2)
            if mixing_matrix is not None:
                positions = self._positions_from_mixing_matrix(sensors_xy, mixing_matrix)
                q_values = self._analytical_intensity_2source(positions, sample, meta, solver)
                params = np.array([
                    positions[0][0], positions[0][1], q_values[0],
                    positions[1][0], positions[1][1], q_values[1]
                ])
                initializations.append((params, 'ica'))

            # 2. ICA with different random seed
            try:
                ica2 = FastICA(n_components=2, random_state=123, max_iter=500)
                source_signals2 = ica2.fit_transform(Y_observed)
                mixing_matrix2 = ica2.mixing_
                if mixing_matrix2 is not None:
                    positions2 = self._positions_from_mixing_matrix(sensors_xy, mixing_matrix2)
                    q_values2 = self._analytical_intensity_2source(positions2, sample, meta, solver)
                    params2 = np.array([
                        positions2[0][0], positions2[0][1], q_values2[0],
                        positions2[1][0], positions2[1][1], q_values2[1]
                    ])
                    initializations.append((params2, 'ica_alt'))
            except Exception:
                pass

            # 3. NMF decomposition
            try:
                nmf = NMF(n_components=2, random_state=42, max_iter=500)
                source_signals_nmf = nmf.fit_transform(np.maximum(Y_observed, 0))
                mixing_nmf = nmf.components_.T
                positions_nmf = self._positions_from_mixing_matrix(sensors_xy, mixing_nmf)
                q_values_nmf = self._analytical_intensity_2source(positions_nmf, sample, meta, solver)
                params_nmf = np.array([
                    positions_nmf[0][0], positions_nmf[0][1], q_values_nmf[0],
                    positions_nmf[1][0], positions_nmf[1][1], q_values_nmf[1]
                ])
                initializations.append((params_nmf, 'nmf'))
            except Exception:
                pass

            # 4. Hottest sensors fallback
            avg_temps = np.mean(Y_observed, axis=0)
            hot_indices = np.argsort(avg_temps)[-2:]
            positions_hot = [(sensors_xy[i, 0], sensors_xy[i, 1]) for i in hot_indices]
            q_values_hot = self._analytical_intensity_2source(positions_hot, sample, meta, solver)
            params_hot = np.array([
                positions_hot[0][0], positions_hot[0][1], q_values_hot[0],
                positions_hot[1][0], positions_hot[1][1], q_values_hot[1]
            ])
            initializations.append((params_hot, 'hottest_sensors'))

        return initializations

    def estimate_sources(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        verbose: bool = False,
    ) -> Tuple[List[List[Tuple[float, float, float]]], float, List[CandidateResult]]:
        """
        Estimate source parameters using direct solution methods with multiple candidates.

        Returns:
            candidates: List of candidate solutions
            best_rmse: RMSE of best candidate
            results: List of CandidateResult objects
        """
        n_sources = sample['n_sources']
        sensors_xy = sample['sensors_xy']
        Y_observed = sample['Y_noisy']

        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        T0 = sample['sample_metadata']['T0']

        solver = self._create_solver(kappa, bc)
        n_evals = 0

        # Generate multiple initializations
        initializations = self._generate_initializations(sample, meta, solver, q_range)

        if verbose:
            print(f"  Generated {len(initializations)} initializations")

        # Collect all solutions from all CMA-ES runs
        all_candidates = []  # List of (sources_list, rmse, init_type)

        for init_params, init_type in initializations:
            if self.use_cmaes_polish:
                # Run CMA-ES and collect all solutions
                best_params, best_rmse, polish_evals, all_solutions = self._cmaes_polish(
                    init_params, sample, meta, solver, q_range, collect_all=True
                )
                n_evals += polish_evals

                # Add all solutions to candidates pool
                for sol_params, sol_rmse in all_solutions:
                    sources = []
                    for i in range(n_sources):
                        x, y, q = sol_params[i*3:(i+1)*3]
                        sources.append((float(x), float(y), float(q)))
                    all_candidates.append((sources, sol_rmse, init_type))

                if verbose:
                    print(f"  {init_type}: best RMSE={best_rmse:.4f}, collected {len(all_solutions)} solutions")
            else:
                # Just evaluate the direct solution
                sources_dict = []
                for i in range(n_sources):
                    sources_dict.append({
                        'x': init_params[i * 3],
                        'y': init_params[i * 3 + 1],
                        'q': init_params[i * 3 + 2]
                    })
                times, Us = solver.solve(dt=dt, nt=nt, T0=T0, sources=sources_dict)
                Y_pred = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])
                rmse = np.sqrt(np.mean((Y_pred - Y_observed) ** 2))
                n_evals += 1

                sources = []
                for i in range(n_sources):
                    x, y, q = init_params[i*3:(i+1)*3]
                    sources.append((float(x), float(y), float(q)))
                all_candidates.append((sources, rmse, init_type))

        if verbose:
            print(f"  Total candidates before filtering: {len(all_candidates)}")

        # Filter to keep dissimilar candidates
        filtered_candidates = filter_dissimilar(all_candidates, tau=TAU, n_max=self.n_candidates)

        if verbose:
            print(f"  After dissimilarity filtering: {len(filtered_candidates)}")

        # Build results
        final_sources = [cand[0] for cand in filtered_candidates]
        final_rmses = [cand[1] for cand in filtered_candidates]
        best_rmse = min(final_rmses) if final_rmses else float('inf')

        results = [
            CandidateResult(
                params=np.array([p for src in sources for p in src]),
                rmse=rmse,
                init_type=init_type,
                n_evals=n_evals // len(filtered_candidates) if filtered_candidates else n_evals
            )
            for sources, rmse, init_type in filtered_candidates
        ]

        return final_sources, best_rmse, results

"""
Fast Adjoint Method Optimizer with Checkpointing.

Uses checkpointing to reduce memory from O(nt) to O(sqrt(nt)) by storing
only every k-th timestep and recomputing intermediate states during backward pass.

For nt=200 timesteps:
- Original: Stores 201 temperature fields (~200 MB for 100x50 grid)
- Checkpointed (k=15): Stores ~14 fields (~14 MB), recomputes ~14x per segment

References:
- Griewank & Walther: "Algorithm 799: Revolve" (optimal checkpointing)
- https://www.cs.utexas.edu/~rvdg/papers/checkpointing.pdf
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import diags, eye
from scipy.sparse.linalg import splu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data', 'Heat_Signature_zero-starter_notebook'))


@dataclass
class CandidateResult:
    """Result from a single optimization run."""
    params: np.ndarray
    rmse: float
    init_type: str


class FastAdjointSolver:
    """
    Heat equation solver with checkpointed adjoint for memory-efficient gradients.
    """

    def __init__(self, Lx: float, Ly: float, nx: int, ny: int, kappa: float, bc: str = "dirichlet"):
        self.Lx, self.Ly = Lx, Ly
        self.nx, self.ny = nx, ny
        self.kappa = kappa
        self.bc = bc

        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

        self.sigma = 2.5 * max(self.dx, self.dy)

        # Build 1D Laplacian operators
        self.Lx_1D = self._build_lap1d(nx, self.dx)
        self.Ly_1D = self._build_lap1d(ny, self.dy)

        self._lu_cache = {}

    def _build_lap1d(self, n: int, d: float) -> np.ndarray:
        """Build 1D Laplacian matrix."""
        main = -2.0 * np.ones(n)
        off = np.ones(n - 1)
        L = diags([off, main, off], [-1, 0, 1], shape=(n, n)).tolil()

        if self.bc == "neumann":
            L[0, 0] = -2.0
            L[0, 1] = 2.0
            L[-1, -1] = -2.0
            L[-1, -2] = 2.0
        elif self.bc == "dirichlet":
            L[0, :] = 0.0
            L[-1, :] = 0.0

        return (1.0 / d**2) * L.tocsr()

    def _get_lu_factors(self, dt: float):
        """Get or create LU factorizations for given dt."""
        if dt not in self._lu_cache:
            r = self.kappa * dt / 2.0
            Ix = eye(self.nx, format="csr")
            Iy = eye(self.ny, format="csr")

            Ax = (Ix - r * self.Lx_1D).tocsc()
            Ay = (Iy - r * self.Ly_1D).tocsc()

            self._lu_cache[dt] = (splu(Ax), splu(Ay), r)

        return self._lu_cache[dt]

    def _gaussian_source(self, x0: float, y0: float, q: float) -> np.ndarray:
        """Compute normalized Gaussian source field."""
        r_sq = (self.X - x0)**2 + (self.Y - y0)**2
        G = np.exp(-r_sq / (2.0 * self.sigma**2))
        integral = np.sum(G) * self.dx * self.dy
        if integral < 1e-12:
            return np.zeros((self.nx, self.ny))
        return G * q / integral

    def _gaussian_derivatives(self, x0: float, y0: float, q: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute Gaussian source and its derivatives."""
        r_sq = (self.X - x0)**2 + (self.Y - y0)**2
        G = np.exp(-r_sq / (2.0 * self.sigma**2))
        integral = np.sum(G) * self.dx * self.dy

        if integral < 1e-12:
            zero = np.zeros((self.nx, self.ny))
            return zero, zero, zero

        dS_dq = G / integral

        dG_dx0 = G * (self.X - x0) / (self.sigma**2)
        dIntegral_dx0 = np.sum(dG_dx0) * self.dx * self.dy
        dS_dx0 = q * (dG_dx0 * integral - G * dIntegral_dx0) / (integral**2)

        dG_dy0 = G * (self.Y - y0) / (self.sigma**2)
        dIntegral_dy0 = np.sum(dG_dy0) * self.dx * self.dy
        dS_dy0 = q * (dG_dy0 * integral - G * dIntegral_dy0) / (integral**2)

        return dS_dq, dS_dx0, dS_dy0

    def _sample_sensors(self, U: np.ndarray, sensors_xy: np.ndarray) -> np.ndarray:
        """Bilinear interpolation at sensor locations."""
        n_sensors = sensors_xy.shape[0]
        out = np.zeros(n_sensors)

        for i, (sx, sy) in enumerate(sensors_xy):
            fx, fy = sx / self.dx, sy / self.dy
            ix = int(np.clip(np.floor(fx), 0, self.nx - 2))
            iy = int(np.clip(np.floor(fy), 0, self.ny - 2))
            tx, ty = fx - ix, fy - iy

            out[i] = ((1-tx)*(1-ty)*U[ix, iy] + tx*(1-ty)*U[ix+1, iy] +
                      (1-tx)*ty*U[ix, iy+1] + tx*ty*U[ix+1, iy+1])

        return out

    def _inject_adjoint_source(self, residuals: np.ndarray, sensors_xy: np.ndarray) -> np.ndarray:
        """Inject adjoint source term from sensor residuals."""
        S_adj = np.zeros((self.nx, self.ny))

        for i, (sx, sy) in enumerate(sensors_xy):
            fx, fy = sx / self.dx, sy / self.dy
            ix = int(np.clip(np.floor(fx), 0, self.nx - 2))
            iy = int(np.clip(np.floor(fy), 0, self.ny - 2))
            tx, ty = fx - ix, fy - iy

            r = residuals[i]
            S_adj[ix, iy] += (1-tx) * (1-ty) * r
            S_adj[ix+1, iy] += tx * (1-ty) * r
            S_adj[ix, iy+1] += (1-tx) * ty * r
            S_adj[ix+1, iy+1] += tx * ty * r

        return S_adj

    def _forward_step(self, U: np.ndarray, S_half: np.ndarray, Ax_lu, Ay_lu, r) -> np.ndarray:
        """Single ADI forward step."""
        # Step 1: Implicit in X, Explicit in Y
        RHS_x = U.copy()
        for i in range(self.nx):
            RHS_x[i, :] += r * (self.Ly_1D @ U[i, :])
        RHS_x += S_half

        U_star = np.zeros_like(U)
        for j in range(self.ny):
            U_star[:, j] = Ax_lu.solve(RHS_x[:, j])

        # Step 2: Implicit in Y, Explicit in X
        RHS_y = U_star.copy()
        for j in range(self.ny):
            RHS_y[:, j] += r * (self.Lx_1D @ U_star[:, j])
        RHS_y += S_half

        U_next = np.zeros_like(U)
        for i in range(self.nx):
            U_next[i, :] = Ay_lu.solve(RHS_y[i, :])

        return U_next

    def _adjoint_step(self, Lambda: np.ndarray, S_adj: np.ndarray, Ax_lu, Ay_lu, r, dt) -> np.ndarray:
        """Single ADI adjoint step (backward in time)."""
        S_adj_half = S_adj * dt / 2.0

        # Step 1: Implicit in Y, Explicit in X
        RHS_y = Lambda.copy()
        for j in range(self.ny):
            RHS_y[:, j] += r * (self.Lx_1D @ Lambda[:, j])
        RHS_y += S_adj_half

        Lambda_star = np.zeros_like(Lambda)
        for i in range(self.nx):
            Lambda_star[i, :] = Ay_lu.solve(RHS_y[i, :])

        # Step 2: Implicit in X, Explicit in Y
        RHS_x = Lambda_star.copy()
        for i in range(self.nx):
            RHS_x[i, :] += r * (self.Ly_1D @ Lambda_star[i, :])
        RHS_x += S_adj_half

        Lambda_next = np.zeros_like(Lambda)
        for j in range(self.ny):
            Lambda_next[:, j] = Ax_lu.solve(RHS_x[:, j])

        return Lambda_next

    def forward_solve_with_checkpoints(
        self,
        sources: List[Dict],
        dt: float,
        nt: int,
        T0: float,
        sensors_xy: np.ndarray,
        checkpoint_interval: int = None
    ) -> Tuple[np.ndarray, List[Tuple[int, np.ndarray]]]:
        """
        Forward solve with checkpointing.

        Args:
            checkpoint_interval: Store every k-th timestep. Default: sqrt(nt)

        Returns:
            Y_sim: Sensor readings at all timesteps
            checkpoints: List of (timestep, temperature_field) tuples
        """
        if checkpoint_interval is None:
            checkpoint_interval = max(1, int(math.sqrt(nt)))

        Ax_lu, Ay_lu, r = self._get_lu_factors(dt)

        # Build source field
        S_total = np.zeros((self.nx, self.ny))
        for src in sources:
            S_total += self._gaussian_source(src['x'], src['y'], src['q'])
        S_half = S_total * dt / 2.0

        U = np.full((self.nx, self.ny), T0)

        n_sensors = sensors_xy.shape[0]
        Y_sim = np.zeros((nt + 1, n_sensors))
        Y_sim[0] = self._sample_sensors(U, sensors_xy)

        checkpoints = [(0, U.copy())]

        for n in range(nt):
            U = self._forward_step(U, S_half, Ax_lu, Ay_lu, r)
            Y_sim[n + 1] = self._sample_sensors(U, sensors_xy)

            # Store checkpoint
            if (n + 1) % checkpoint_interval == 0 or n + 1 == nt:
                checkpoints.append((n + 1, U.copy()))

        return Y_sim, checkpoints

    def _recompute_segment(
        self,
        start_U: np.ndarray,
        start_n: int,
        end_n: int,
        S_half: np.ndarray,
        Ax_lu, Ay_lu, r
    ) -> List[np.ndarray]:
        """Recompute forward states from checkpoint to end."""
        states = [start_U.copy()]
        U = start_U.copy()

        for n in range(start_n, end_n):
            U = self._forward_step(U, S_half, Ax_lu, Ay_lu, r)
            states.append(U.copy())

        return states

    def compute_gradient_checkpointed(
        self,
        sources: List[Dict],
        Y_obs: np.ndarray,
        dt: float,
        nt: int,
        T0: float,
        sensors_xy: np.ndarray,
        checkpoint_interval: int = None
    ) -> Tuple[float, np.ndarray]:
        """
        Compute MSE and gradient using checkpointed adjoint.

        Memory: O(sqrt(nt) * nx * ny) instead of O(nt * nx * ny)
        Computation: ~2x forward solves (checkpoint + recompute)
        """
        if checkpoint_interval is None:
            checkpoint_interval = max(1, int(math.sqrt(nt)))

        # Forward pass with checkpoints
        Y_sim, checkpoints = self.forward_solve_with_checkpoints(
            sources, dt, nt, T0, sensors_xy, checkpoint_interval
        )

        # Compute residuals and MSE
        residuals = Y_sim - Y_obs
        mse = np.mean(residuals**2)

        # Get solver components
        Ax_lu, Ay_lu, r = self._get_lu_factors(dt)

        # Build source field for recomputation
        S_total = np.zeros((self.nx, self.ny))
        for src in sources:
            S_total += self._gaussian_source(src['x'], src['y'], src['q'])
        S_half = S_total * dt / 2.0

        # Precompute source derivatives
        n_sources = len(sources)
        source_derivs = []
        for src in sources:
            dS_dq, dS_dx0, dS_dy0 = self._gaussian_derivatives(src['x'], src['y'], src['q'])
            source_derivs.append((dS_dq, dS_dx0, dS_dy0))

        # Initialize gradients
        gradients = np.zeros(n_sources * 3)

        # Backward pass with checkpointing
        Lambda = np.zeros((self.nx, self.ny))  # Terminal condition

        # Process checkpoints in reverse
        checkpoint_times = [c[0] for c in checkpoints]

        for i in range(len(checkpoints) - 1, 0, -1):
            end_n = checkpoints[i][0]
            start_n = checkpoints[i-1][0]
            start_U = checkpoints[i-1][1]

            # Recompute forward states for this segment
            segment_states = self._recompute_segment(
                start_U, start_n, end_n, S_half, Ax_lu, Ay_lu, r
            )

            # Backward through segment
            for local_n in range(end_n - start_n - 1, -1, -1):
                global_n = start_n + local_n + 1  # Global timestep (1 to nt)

                # Adjoint source
                S_adj = self._inject_adjoint_source(residuals[global_n], sensors_xy)

                # Adjoint step
                Lambda = self._adjoint_step(Lambda, S_adj, Ax_lu, Ay_lu, r, dt)

                # Accumulate gradients
                for k, (dS_dq, dS_dx0, dS_dy0) in enumerate(source_derivs):
                    gradients[k * 3] += np.sum(Lambda * dS_dx0)
                    gradients[k * 3 + 1] += np.sum(Lambda * dS_dy0)
                    gradients[k * 3 + 2] += np.sum(Lambda * dS_dq)

        # Scale gradients for MSE
        n_obs = Y_obs.size
        gradients *= 2.0 / n_obs

        return mse, gradients


class FastAdjointOptimizer:
    """
    Heat source optimizer using checkpointed adjoint method.
    """

    def __init__(
        self,
        Lx: float = 2.0,
        Ly: float = 1.0,
        nx: int = 100,
        ny: int = 50,
        n_smart_inits: int = 1,
        n_random_inits: int = 1,
        min_candidate_distance: float = 0.15,
        n_max_candidates: int = 3,
        checkpoint_interval: int = None,
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.n_smart_inits = n_smart_inits
        self.n_random_inits = n_random_inits
        self.min_candidate_distance = min_candidate_distance
        self.n_max_candidates = n_max_candidates
        self.checkpoint_interval = checkpoint_interval

        self._solver_cache = {}

    def _get_solver(self, kappa: float, bc: str) -> FastAdjointSolver:
        """Get or create solver for given parameters."""
        key = (kappa, bc)
        if key not in self._solver_cache:
            self._solver_cache[key] = FastAdjointSolver(
                self.Lx, self.Ly, self.nx, self.ny, kappa, bc
            )
        return self._solver_cache[key]

    def _smart_init(self, sample: Dict, n_sources: int, q_range: Tuple[float, float]) -> np.ndarray:
        """Initialize from hottest sensors."""
        readings = sample['Y_noisy']
        sensors = sample['sensors_xy']
        avg_temps = np.mean(readings, axis=0)
        hot_idx = np.argsort(avg_temps)[::-1]

        selected = []
        for idx in hot_idx:
            if len(selected) >= n_sources:
                break
            if all(np.linalg.norm(sensors[idx] - sensors[s]) >= 0.2 for s in selected):
                selected.append(idx)

        while len(selected) < n_sources:
            for idx in hot_idx:
                if idx not in selected:
                    selected.append(idx)
                    break

        params = []
        max_temp = np.max(avg_temps) + 1e-8
        for idx in selected:
            x, y = sensors[idx]
            q = 0.5 + (avg_temps[idx] / max_temp) * 1.5
            q = np.clip(q, q_range[0], q_range[1])
            params.extend([x, y, q])

        return np.array(params)

    def _random_init(self, n_sources: int, q_range: Tuple[float, float], margin: float = 0.1) -> np.ndarray:
        """Random initialization."""
        params = []
        for _ in range(n_sources):
            x = np.random.uniform(margin * self.Lx, (1 - margin) * self.Lx)
            y = np.random.uniform(margin * self.Ly, (1 - margin) * self.Ly)
            q = np.random.uniform(q_range[0], q_range[1])
            params.extend([x, y, q])
        return np.array(params)

    def _get_bounds(self, n_sources: int, q_range: Tuple[float, float], margin: float = 0.05):
        """Get optimization bounds."""
        bounds = []
        for _ in range(n_sources):
            bounds.append((margin * self.Lx, (1 - margin) * self.Lx))
            bounds.append((margin * self.Ly, (1 - margin) * self.Ly))
            bounds.append(q_range)
        return bounds

    def _params_to_sources(self, params: np.ndarray, n_sources: int) -> List[Dict]:
        """Convert flat parameter array to source list."""
        sources = []
        for i in range(n_sources):
            sources.append({
                'x': params[i * 3],
                'y': params[i * 3 + 1],
                'q': params[i * 3 + 2]
            })
        return sources

    def estimate_sources(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        max_iter: int = 50,
        verbose: bool = False,
    ) -> Tuple[List[Tuple[float, float, float]], float, List[CandidateResult]]:
        """Estimate heat sources using checkpointed adjoint optimization."""
        n_sources = sample['n_sources']
        sensors_xy = sample['sensors_xy']
        Y_obs = sample['Y_noisy']

        dt = meta['dt']
        nt = sample['sample_metadata']['nt']
        kappa = sample['sample_metadata']['kappa']
        bc = sample['sample_metadata']['bc']
        T0 = sample['sample_metadata']['T0']

        solver = self._get_solver(kappa, bc)
        bounds = self._get_bounds(n_sources, q_range)

        # Checkpoint interval
        cp_interval = self.checkpoint_interval or max(1, int(math.sqrt(nt)))

        candidates = []

        # Generate initial points
        inits = []
        for i in range(self.n_smart_inits):
            inits.append(('smart', self._smart_init(sample, n_sources, q_range)))
        for i in range(self.n_random_inits):
            inits.append(('random', self._random_init(n_sources, q_range)))

        if verbose:
            print(f"  Running {len(inits)} inits with checkpointed adjoint (cp_interval={cp_interval})...")

        for init_type, x0 in inits:
            try:
                def objective_grad(params):
                    sources = self._params_to_sources(params, n_sources)
                    mse, grad = solver.compute_gradient_checkpointed(
                        sources, Y_obs, dt, nt, T0, sensors_xy, cp_interval
                    )
                    return mse, grad

                result = minimize(
                    objective_grad,
                    x0=x0,
                    method='L-BFGS-B',
                    jac=True,
                    bounds=bounds,
                    options={'maxiter': max_iter, 'disp': False}
                )

                rmse = np.sqrt(result.fun)
                candidates.append(CandidateResult(result.x, rmse, init_type))

                if verbose:
                    print(f"    {init_type}: RMSE={rmse:.4f}, iters={result.nit}")

            except Exception as e:
                if verbose:
                    print(f"    {init_type}: failed - {e}")

        if not candidates:
            return [], float('inf'), []

        candidates.sort(key=lambda c: c.rmse)
        filtered = self._filter_distinct(candidates, n_sources)

        best = filtered[0]
        sources = []
        for i in range(n_sources):
            x, y, q = best.params[i*3:(i+1)*3]
            sources.append((float(x), float(y), float(q)))

        return sources, best.rmse, filtered[:self.n_max_candidates]

    def _filter_distinct(self, candidates: List[CandidateResult], n_sources: int) -> List[CandidateResult]:
        """Filter candidates to keep only distinct ones."""
        if len(candidates) <= 1:
            return candidates

        filtered = [candidates[0]]

        for cand in candidates[1:]:
            is_distinct = True
            for kept in filtered:
                dist = self._candidate_distance(cand.params, kept.params, n_sources)
                if dist < self.min_candidate_distance:
                    is_distinct = False
                    break

            if is_distinct:
                filtered.append(cand)
                if len(filtered) >= self.n_max_candidates:
                    break

        return filtered

    def _candidate_distance(self, params1: np.ndarray, params2: np.ndarray, n_sources: int) -> float:
        """Compute normalized distance between two candidates."""
        sources1 = [(params1[i*3], params1[i*3+1], params1[i*3+2]) for i in range(n_sources)]
        sources2 = [(params2[i*3], params2[i*3+1], params2[i*3+2]) for i in range(n_sources)]

        sources1.sort()
        sources2.sort()

        total_dist = 0.0
        for s1, s2 in zip(sources1, sources2):
            dx = (s1[0] - s2[0]) / self.Lx
            dy = (s1[1] - s2[1]) / self.Ly
            dq = (s1[2] - s2[2]) / 2.0
            total_dist += np.sqrt(dx**2 + dy**2 + dq**2)

        return total_dist / n_sources


def validate_checkpointed_gradients():
    """Validate checkpointed gradients against finite differences."""
    print("Validating checkpointed adjoint gradients...")
    print("=" * 60)

    solver = FastAdjointSolver(2.0, 1.0, 50, 25, 0.1, "dirichlet")

    sources = [{'x': 1.0, 'y': 0.5, 'q': 1.0}]
    sensors_xy = np.array([[0.5, 0.25], [1.0, 0.5], [1.5, 0.75]])
    dt = 0.01
    nt = 50
    T0 = 0.0

    # Generate observed data
    true_sources = [{'x': 1.05, 'y': 0.52, 'q': 1.1}]
    Y_obs, _ = solver.forward_solve_with_checkpoints(true_sources, dt, nt, T0, sensors_xy, checkpoint_interval=10)

    # Checkpointed gradient
    mse, adj_grad = solver.compute_gradient_checkpointed(
        sources, Y_obs, dt, nt, T0, sensors_xy, checkpoint_interval=10
    )

    print(f"MSE: {mse:.6f}")
    print(f"Checkpointed adjoint gradient: {adj_grad}")

    # Finite difference gradient
    eps = 1e-5
    n_sources = len(sources)
    params = np.array([v for s in sources for v in [s['x'], s['y'], s['q']]])
    fd_grad = np.zeros_like(params)

    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        sources_plus = [{'x': params_plus[j*3], 'y': params_plus[j*3+1], 'q': params_plus[j*3+2]}
                        for j in range(n_sources)]
        Y_plus, _ = solver.forward_solve_with_checkpoints(sources_plus, dt, nt, T0, sensors_xy, checkpoint_interval=10)
        mse_plus = np.mean((Y_plus - Y_obs)**2)

        params_minus = params.copy()
        params_minus[i] -= eps
        sources_minus = [{'x': params_minus[j*3], 'y': params_minus[j*3+1], 'q': params_minus[j*3+2]}
                         for j in range(n_sources)]
        Y_minus, _ = solver.forward_solve_with_checkpoints(sources_minus, dt, nt, T0, sensors_xy, checkpoint_interval=10)
        mse_minus = np.mean((Y_minus - Y_obs)**2)

        fd_grad[i] = (mse_plus - mse_minus) / (2 * eps)

    print(f"FD gradient: {fd_grad}")

    rel_error = np.abs(adj_grad - fd_grad) / (np.abs(fd_grad) + 1e-12)
    print(f"Relative errors: {rel_error}")
    print(f"Max relative error: {np.max(rel_error):.2e}")

    if np.max(rel_error) < 0.05:
        print("[OK] Checkpointed gradients validated!")
        return True
    else:
        print("[FAIL] Gradient mismatch")
        return False


if __name__ == "__main__":
    import time

    # First validate gradients
    if not validate_checkpointed_gradients():
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Performance comparison: Checkpointed vs Full storage")
    print("=" * 60)

    # Test with larger problem
    solver = FastAdjointSolver(2.0, 1.0, 100, 50, 0.1, "dirichlet")
    sources = [{'x': 1.0, 'y': 0.5, 'q': 1.0}]
    sensors_xy = np.array([[0.5, 0.25], [1.0, 0.5], [1.5, 0.75]])
    dt = 0.004
    nt = 200
    T0 = 0.0

    true_sources = [{'x': 1.05, 'y': 0.52, 'q': 1.1}]
    Y_obs, _ = solver.forward_solve_with_checkpoints(true_sources, dt, nt, T0, sensors_xy)

    # Test different checkpoint intervals
    for cp_interval in [10, 15, 20, 50, nt]:
        start = time.time()
        mse, grad = solver.compute_gradient_checkpointed(
            sources, Y_obs, dt, nt, T0, sensors_xy, checkpoint_interval=cp_interval
        )
        elapsed = time.time() - start

        n_checkpoints = (nt // cp_interval) + 1
        print(f"cp_interval={cp_interval:3d}: {n_checkpoints:3d} checkpoints, time={elapsed:.3f}s, grad_norm={np.linalg.norm(grad):.4f}")

    print("\n" + "=" * 60)
    print("Comparison: Checkpointed Adjoint vs Finite Differences")
    print("=" * 60)

    # Checkpointed adjoint (optimal interval)
    cp_interval = int(math.sqrt(nt))
    start = time.time()
    for _ in range(5):
        mse, adj_grad = solver.compute_gradient_checkpointed(
            sources, Y_obs, dt, nt, T0, sensors_xy, checkpoint_interval=cp_interval
        )
    adj_time = (time.time() - start) / 5
    print(f"Checkpointed Adjoint (cp={cp_interval}): {adj_time:.3f}s per gradient")

    # Finite differences (central)
    n_params = len(sources) * 3
    eps = 1e-5
    params = np.array([v for s in sources for v in [s['x'], s['y'], s['q']]])

    start = time.time()
    for _ in range(3):
        fd_grad = np.zeros(n_params)
        for i in range(n_params):
            params_plus = params.copy()
            params_plus[i] += eps
            sources_plus = [{'x': params_plus[j*3], 'y': params_plus[j*3+1], 'q': params_plus[j*3+2]}
                            for j in range(len(sources))]
            Y_plus, _ = solver.forward_solve_with_checkpoints(sources_plus, dt, nt, T0, sensors_xy, checkpoint_interval=nt)
            mse_plus = np.mean((Y_plus - Y_obs)**2)

            params_minus = params.copy()
            params_minus[i] -= eps
            sources_minus = [{'x': params_minus[j*3], 'y': params_minus[j*3+1], 'q': params_minus[j*3+2]}
                             for j in range(len(sources))]
            Y_minus, _ = solver.forward_solve_with_checkpoints(sources_minus, dt, nt, T0, sensors_xy, checkpoint_interval=nt)
            mse_minus = np.mean((Y_minus - Y_obs)**2)

            fd_grad[i] = (mse_plus - mse_minus) / (2 * eps)
    fd_time = (time.time() - start) / 3
    print(f"Finite Differences ({n_params} params x 2 evals): {fd_time:.3f}s per gradient")
    print(f"Speedup: {fd_time / adj_time:.2f}x")

    # For 2-source problems
    print(f"\nProjected for 2-source (6 params):")
    print(f"  Adjoint: {adj_time:.3f}s (same)")
    print(f"  FD: {fd_time * 2:.3f}s (2x more params)")
    print(f"  Projected speedup: {(fd_time * 2) / adj_time:.2f}x")

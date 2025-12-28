"""
Adjoint Method Optimizer for Heat Source Identification.

The adjoint method computes exact gradients with cost independent of parameter count.
Instead of O(n_params) forward simulations for finite differences, we need only:
- 1 forward simulation (storing all timesteps)
- 1 backward (adjoint) simulation
- O(1) gradient computation

Key equations:
- Forward:  ∂T/∂t = κ∇²T + S(x,y)
- Adjoint:  -∂λ/∂t = κ∇²λ + Σ_sensors δ(x-xs)(T-Tobs)
- Gradients: ∂J/∂θ = ∫∫∫ λ · ∂S/∂θ dx dy dt

References:
- NC State: https://aalexan3.math.ncsu.edu/articles/adjoint_based_gradient_and_hessian.pdf
- MIT Notes: https://math.mit.edu/~stevenj/18.336/adjoint.pdf
"""

import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import diags, eye
from scipy.sparse.linalg import splu

# Add the starter notebook path for the original simulator (for validation)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data', 'Heat_Signature_zero-starter_notebook'))


@dataclass
class CandidateResult:
    """Result from a single optimization run."""
    params: np.ndarray
    rmse: float
    init_type: str


class AdjointHeatSolver:
    """
    Heat equation solver with adjoint capability for gradient computation.

    Uses ADI (Alternating Direction Implicit) method for both forward and adjoint.
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

        # Gaussian spread (grid-independent)
        self.sigma = 2.5 * max(self.dx, self.dy)

        # Build 1D Laplacian operators
        self.Lx_1D = self._build_lap1d(nx, self.dx)
        self.Ly_1D = self._build_lap1d(ny, self.dy)

        # LU factorizations will be cached per dt
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
        """
        Compute Gaussian source and its derivatives w.r.t. x0, y0, q.

        Returns:
            dS_dq: ∂S/∂q (shape: nx x ny)
            dS_dx0: ∂S/∂x0 (shape: nx x ny)
            dS_dy0: ∂S/∂y0 (shape: nx x ny)
        """
        r_sq = (self.X - x0)**2 + (self.Y - y0)**2
        G = np.exp(-r_sq / (2.0 * self.sigma**2))
        integral = np.sum(G) * self.dx * self.dy

        if integral < 1e-12:
            zero = np.zeros((self.nx, self.ny))
            return zero, zero, zero

        # S = G * q / integral
        # dS/dq = G / integral
        dS_dq = G / integral

        # dG/dx0 = G * (x - x0) / sigma^2
        dG_dx0 = G * (self.X - x0) / (self.sigma**2)
        # dIntegral/dx0 = sum(dG/dx0) * dx * dy
        dIntegral_dx0 = np.sum(dG_dx0) * self.dx * self.dy
        # dS/dx0 = (dG/dx0 * integral - G * dIntegral/dx0) * q / integral^2
        dS_dx0 = q * (dG_dx0 * integral - G * dIntegral_dx0) / (integral**2)

        # Similarly for y0
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
        """
        Inject adjoint source term from sensor residuals.

        The adjoint equation has source: Σ_sensors δ(x-xs) * residual
        We approximate delta functions using bilinear interpolation weights.
        """
        S_adj = np.zeros((self.nx, self.ny))

        for i, (sx, sy) in enumerate(sensors_xy):
            fx, fy = sx / self.dx, sy / self.dy
            ix = int(np.clip(np.floor(fx), 0, self.nx - 2))
            iy = int(np.clip(np.floor(fy), 0, self.ny - 2))
            tx, ty = fx - ix, fy - iy

            # Distribute residual to neighboring grid points
            r = residuals[i]
            S_adj[ix, iy] += (1-tx) * (1-ty) * r
            S_adj[ix+1, iy] += tx * (1-ty) * r
            S_adj[ix, iy+1] += (1-tx) * ty * r
            S_adj[ix+1, iy+1] += tx * ty * r

        return S_adj

    def forward_solve(
        self,
        sources: List[Dict],
        dt: float,
        nt: int,
        T0: float,
        sensors_xy: np.ndarray,
        store_fields: bool = True
    ) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
        """
        Forward solve with optional storage of all temperature fields.

        Args:
            sources: List of source dicts with 'x', 'y', 'q' keys
            dt: Time step
            nt: Number of time steps
            T0: Initial temperature
            sensors_xy: Sensor locations (n_sensors x 2)
            store_fields: Whether to store all T fields (needed for adjoint)

        Returns:
            Y_sim: Simulated sensor readings (nt+1 x n_sensors)
            T_history: List of temperature fields if store_fields=True, else None
        """
        Ax_lu, Ay_lu, r = self._get_lu_factors(dt)

        # Build total source field (constant in time for this problem)
        S_total = np.zeros((self.nx, self.ny))
        for src in sources:
            S_total += self._gaussian_source(src['x'], src['y'], src['q'])

        U = np.full((self.nx, self.ny), T0)

        n_sensors = sensors_xy.shape[0]
        Y_sim = np.zeros((nt + 1, n_sensors))
        Y_sim[0] = self._sample_sensors(U, sensors_xy)

        T_history = [U.copy()] if store_fields else None

        # ADI time stepping
        for n in range(nt):
            S_half = S_total * dt / 2.0

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

            U = U_next
            Y_sim[n + 1] = self._sample_sensors(U, sensors_xy)

            if store_fields:
                T_history.append(U.copy())

        return Y_sim, T_history

    def adjoint_solve(
        self,
        residuals: np.ndarray,
        sensors_xy: np.ndarray,
        dt: float,
        nt: int
    ) -> List[np.ndarray]:
        """
        Backward (adjoint) solve.

        The adjoint equation is:
        -∂λ/∂t = κ∇²λ + Σ_sensors δ(x-xs)(T-Tobs)

        Which we solve backward in time from t=T to t=0.
        Terminal condition: λ(T) = 0

        Args:
            residuals: (nt+1 x n_sensors) array of (T_sim - T_obs)
            sensors_xy: Sensor locations
            dt: Time step
            nt: Number of time steps

        Returns:
            lambda_history: List of adjoint fields from t=T to t=0
        """
        Ax_lu, Ay_lu, r = self._get_lu_factors(dt)

        # Terminal condition: λ(T) = 0
        Lambda = np.zeros((self.nx, self.ny))

        lambda_history = [Lambda.copy()]

        # Backward time stepping (from nt down to 0)
        for n in range(nt, 0, -1):
            # Adjoint source at this time step
            # The adjoint source term is the derivative of the objective w.r.t. temperature
            # For J = (1/N) * sum((T-Tobs)^2), this is (2/N) * (T-Tobs) at sensors
            # The injection distributes point values to grid using bilinear weights
            S_adj = self._inject_adjoint_source(residuals[n], sensors_xy)

            # ADI backward step (same structure as forward, since Laplacian is self-adjoint)
            # Step 1: Implicit in Y, Explicit in X (reversed order)
            RHS_y = Lambda.copy()
            for j in range(self.ny):
                RHS_y[:, j] += r * (self.Lx_1D @ Lambda[:, j])
            RHS_y += S_adj * dt / 2.0

            Lambda_star = np.zeros_like(Lambda)
            for i in range(self.nx):
                Lambda_star[i, :] = Ay_lu.solve(RHS_y[i, :])

            # Step 2: Implicit in X, Explicit in Y
            RHS_x = Lambda_star.copy()
            for i in range(self.nx):
                RHS_x[i, :] += r * (self.Ly_1D @ Lambda_star[i, :])
            RHS_x += S_adj * dt / 2.0

            Lambda_next = np.zeros_like(Lambda)
            for j in range(self.ny):
                Lambda_next[:, j] = Ax_lu.solve(RHS_x[:, j])

            Lambda = Lambda_next
            lambda_history.append(Lambda.copy())

        # Reverse so index 0 = t=0, index nt = t=T
        lambda_history.reverse()

        return lambda_history

    def compute_gradients(
        self,
        sources: List[Dict],
        lambda_history: List[np.ndarray],
        dt: float
    ) -> np.ndarray:
        """
        Compute gradients using adjoint variables.

        For each source k with parameters (x_k, y_k, q_k):
        ∂J/∂q_k = ∫_0^T ∫∫ λ · (∂S/∂q_k) dx dy dt
        ∂J/∂x_k = ∫_0^T ∫∫ λ · (∂S/∂x_k) dx dy dt
        ∂J/∂y_k = ∫_0^T ∫∫ λ · (∂S/∂y_k) dx dy dt

        Args:
            sources: List of source dicts
            lambda_history: Adjoint fields at each timestep
            dt: Time step

        Returns:
            gradients: Array of shape (n_sources * 3,) with [x0, y0, q0, x1, y1, q1, ...]
        """
        n_sources = len(sources)
        gradients = np.zeros(n_sources * 3)

        nt = len(lambda_history) - 1
        dV = self.dx * self.dy  # Volume element

        for k, src in enumerate(sources):
            x0, y0, q = src['x'], src['y'], src['q']

            # Compute source derivatives (constant in time)
            dS_dq, dS_dx0, dS_dy0 = self._gaussian_derivatives(x0, y0, q)

            # Integrate over time
            grad_x, grad_y, grad_q = 0.0, 0.0, 0.0

            for n in range(nt + 1):
                Lambda = lambda_history[n]

                # For discrete adjoint, we sum over all timesteps
                # The adjoint equation already incorporates dt in its evolution,
                # so we just sum the contributions without additional time weighting
                grad_x += np.sum(Lambda * dS_dx0)
                grad_y += np.sum(Lambda * dS_dy0)
                grad_q += np.sum(Lambda * dS_dq)

            # Store in order: x, y, q for each source
            gradients[k * 3] = grad_x
            gradients[k * 3 + 1] = grad_y
            gradients[k * 3 + 2] = grad_q

        # Debug: print gradient magnitudes
        # print(f"  DEBUG: grad magnitudes: {np.abs(gradients)}")

        return gradients


class AdjointOptimizer:
    """
    Heat source optimizer using adjoint method for exact gradients.

    Advantages over finite differences:
    - O(1) gradient computation vs O(n_params)
    - Exact gradients (no approximation error)
    - Better convergence with L-BFGS-B
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
    ):
        self.Lx = Lx
        self.Ly = Ly
        self.nx = nx
        self.ny = ny
        self.n_smart_inits = n_smart_inits
        self.n_random_inits = n_random_inits
        self.min_candidate_distance = min_candidate_distance
        self.n_max_candidates = n_max_candidates

        # Cache solvers by (kappa, bc) tuple
        self._solver_cache = {}

    def _get_solver(self, kappa: float, bc: str) -> AdjointHeatSolver:
        """Get or create solver for given parameters."""
        key = (kappa, bc)
        if key not in self._solver_cache:
            self._solver_cache[key] = AdjointHeatSolver(
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

    def _objective_and_gradient(
        self,
        params: np.ndarray,
        solver: AdjointHeatSolver,
        n_sources: int,
        dt: float,
        nt: int,
        T0: float,
        sensors_xy: np.ndarray,
        Y_obs: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Compute objective (MSE) and exact gradients using adjoint method.

        This is the key function - it replaces finite differences with:
        1. Forward solve (store all fields)
        2. Backward adjoint solve
        3. Gradient computation

        Total cost: 2 simulations instead of (1 + n_params) simulations.
        """
        sources = self._params_to_sources(params, n_sources)

        # Forward solve with field storage
        Y_sim, T_history = solver.forward_solve(
            sources, dt, nt, T0, sensors_xy, store_fields=True
        )

        # Compute residuals and objective
        residuals = Y_sim - Y_obs
        mse = np.mean(residuals**2)

        # Adjoint solve
        lambda_history = solver.adjoint_solve(residuals, sensors_xy, dt, nt)

        # Compute gradients
        # The adjoint gives gradient of sum((T-Tobs)^2)/2
        # For MSE = (1/N) * sum((T-Tobs)^2), we need to multiply by 2/N
        gradients = solver.compute_gradients(sources, lambda_history, dt)
        n_obs = Y_obs.size
        gradients *= 2.0 / n_obs

        return mse, gradients

    def estimate_sources(
        self,
        sample: Dict,
        meta: Dict,
        q_range: Tuple[float, float] = (0.5, 2.0),
        max_iter: int = 50,
        verbose: bool = False,
    ) -> Tuple[List[Tuple[float, float, float]], float, List[CandidateResult]]:
        """
        Estimate heat sources using adjoint-based optimization.

        Args:
            sample: Sample dict with 'n_sources', 'sensors_xy', 'Y_noisy', 'sample_metadata'
            meta: Metadata dict with 'dt'
            q_range: Intensity bounds (q_min, q_max)
            max_iter: Maximum L-BFGS-B iterations
            verbose: Print progress

        Returns:
            best_sources: List of (x, y, q) tuples
            best_rmse: RMSE of best solution
            candidates: List of CandidateResult objects
        """
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

        candidates = []

        # Generate initial points
        inits = []
        for i in range(self.n_smart_inits):
            inits.append(('smart', self._smart_init(sample, n_sources, q_range)))
        for i in range(self.n_random_inits):
            inits.append(('random', self._random_init(n_sources, q_range)))

        if verbose:
            print(f"  Running {len(inits)} initializations with adjoint gradients...")

        for init_type, x0 in inits:
            try:
                # Wrapper for scipy
                def objective_grad(params):
                    mse, grad = self._objective_and_gradient(
                        params, solver, n_sources, dt, nt, T0, sensors_xy, Y_obs
                    )
                    return mse, grad

                result = minimize(
                    objective_grad,
                    x0=x0,
                    method='L-BFGS-B',
                    jac=True,  # We provide gradients
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

        # Sort by RMSE
        candidates.sort(key=lambda c: c.rmse)

        # Filter distinct candidates
        filtered = self._filter_distinct(candidates, n_sources)

        # Return best solution
        best = filtered[0]
        sources = []
        for i in range(n_sources):
            x, y, q = best.params[i*3:(i+1)*3]
            sources.append((float(x), float(y), float(q)))

        return sources, best.rmse, filtered[:self.n_max_candidates]

    def _filter_distinct(
        self,
        candidates: List[CandidateResult],
        n_sources: int
    ) -> List[CandidateResult]:
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
        # Extract and sort sources for permutation invariance
        sources1 = [(params1[i*3], params1[i*3+1], params1[i*3+2]) for i in range(n_sources)]
        sources2 = [(params2[i*3], params2[i*3+1], params2[i*3+2]) for i in range(n_sources)]

        sources1.sort()
        sources2.sort()

        # Normalized distance
        total_dist = 0.0
        for s1, s2 in zip(sources1, sources2):
            dx = (s1[0] - s2[0]) / self.Lx
            dy = (s1[1] - s2[1]) / self.Ly
            dq = (s1[2] - s2[2]) / 2.0  # q_range is [0.5, 2.0], span = 1.5
            total_dist += np.sqrt(dx**2 + dy**2 + dq**2)

        return total_dist / n_sources


def validate_gradients(
    solver: AdjointHeatSolver,
    sources: List[Dict],
    dt: float,
    nt: int,
    T0: float,
    sensors_xy: np.ndarray,
    Y_obs: np.ndarray,
    eps: float = 1e-5
) -> Dict:
    """
    Validate adjoint gradients against finite differences.

    Returns dict with 'adjoint_grad', 'fd_grad', 'relative_error'.
    """
    n_sources = len(sources)
    params = np.array([v for s in sources for v in [s['x'], s['y'], s['q']]])

    # Adjoint gradient
    Y_sim, T_history = solver.forward_solve(sources, dt, nt, T0, sensors_xy, store_fields=True)
    residuals = Y_sim - Y_obs
    mse = np.mean(residuals**2)

    lambda_history = solver.adjoint_solve(residuals, sensors_xy, dt, nt)
    adj_grad = solver.compute_gradients(sources, lambda_history, dt)
    adj_grad *= 2.0 / Y_obs.size  # Scale for MSE

    # Finite difference gradient
    fd_grad = np.zeros_like(params)

    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        sources_plus = [{'x': params_plus[j*3], 'y': params_plus[j*3+1], 'q': params_plus[j*3+2]}
                        for j in range(n_sources)]
        Y_plus, _ = solver.forward_solve(sources_plus, dt, nt, T0, sensors_xy, store_fields=False)
        mse_plus = np.mean((Y_plus - Y_obs)**2)

        params_minus = params.copy()
        params_minus[i] -= eps
        sources_minus = [{'x': params_minus[j*3], 'y': params_minus[j*3+1], 'q': params_minus[j*3+2]}
                         for j in range(n_sources)]
        Y_minus, _ = solver.forward_solve(sources_minus, dt, nt, T0, sensors_xy, store_fields=False)
        mse_minus = np.mean((Y_minus - Y_obs)**2)

        fd_grad[i] = (mse_plus - mse_minus) / (2 * eps)

    # Relative error
    rel_error = np.abs(adj_grad - fd_grad) / (np.abs(fd_grad) + 1e-12)

    return {
        'adjoint_grad': adj_grad,
        'fd_grad': fd_grad,
        'relative_error': rel_error,
        'max_rel_error': np.max(rel_error),
        'mean_rel_error': np.mean(rel_error)
    }


if __name__ == "__main__":
    # Quick validation test
    print("Testing Adjoint Optimizer...")
    print("="*60)

    # Use smaller problem for debugging
    solver = AdjointHeatSolver(2.0, 1.0, 50, 25, 0.1, "dirichlet")

    # Test sources
    sources = [{'x': 1.0, 'y': 0.5, 'q': 1.0}]
    sensors_xy = np.array([[0.5, 0.25], [1.0, 0.5], [1.5, 0.75]])
    dt = 0.01
    nt = 50
    T0 = 0.0

    print(f"Grid: {solver.nx}x{solver.ny}, dt={dt}, nt={nt}")
    print(f"dx={solver.dx:.4f}, dy={solver.dy:.4f}")
    print(f"Source: x={sources[0]['x']}, y={sources[0]['y']}, q={sources[0]['q']}")

    # Generate "observed" data with slight offset from source
    true_sources = [{'x': 1.05, 'y': 0.52, 'q': 1.1}]
    Y_obs, _ = solver.forward_solve(true_sources, dt, nt, T0, sensors_xy, store_fields=False)

    # Compute forward with test sources
    Y_sim, T_history = solver.forward_solve(sources, dt, nt, T0, sensors_xy, store_fields=True)

    print(f"\nY_sim shape: {Y_sim.shape}")
    print(f"Y_sim range: [{Y_sim.min():.4f}, {Y_sim.max():.4f}]")
    print(f"Y_obs range: [{Y_obs.min():.4f}, {Y_obs.max():.4f}]")

    residuals = Y_sim - Y_obs
    mse = np.mean(residuals**2)
    print(f"MSE: {mse:.6f}")
    print(f"Residuals range: [{residuals.min():.4f}, {residuals.max():.4f}]")

    # Adjoint solve
    lambda_history = solver.adjoint_solve(residuals, sensors_xy, dt, nt)
    print(f"\nLambda history length: {len(lambda_history)}")
    print(f"Lambda[0] range: [{lambda_history[0].min():.6f}, {lambda_history[0].max():.6f}]")
    print(f"Lambda[nt//2] range: [{lambda_history[nt//2].min():.6f}, {lambda_history[nt//2].max():.6f}]")
    print(f"Lambda[nt] range: [{lambda_history[nt].min():.6f}, {lambda_history[nt].max():.6f}]")

    # Compute gradients
    adj_grad_raw = solver.compute_gradients(sources, lambda_history, dt)

    print(f"\nAdjoint gradient (raw): {adj_grad_raw}")
    print(f"Note: FD computes gradient of MSE = (1/N)*sum((T-Tobs)^2)")
    print(f"      Adjoint computes gradient of sum((T-Tobs)^2)/2")
    print(f"      So we need to scale by 2/N = 2/{Y_obs.size} = {2.0/Y_obs.size:.6f}")

    # For comparison, don't scale - check raw values first
    adj_grad = adj_grad_raw.copy()

    # Finite difference gradient
    eps = 1e-5
    n_sources = len(sources)
    params = np.array([v for s in sources for v in [s['x'], s['y'], s['q']]])
    fd_grad = np.zeros_like(params)

    # Compute FD for same loss function as adjoint: J = sum((T-Tobs)^2) / 2
    print(f"\nFinite difference gradients for J = sum((T-Tobs)^2)/2:")
    for i in range(len(params)):
        params_plus = params.copy()
        params_plus[i] += eps
        sources_plus = [{'x': params_plus[j*3], 'y': params_plus[j*3+1], 'q': params_plus[j*3+2]}
                        for j in range(n_sources)]
        Y_plus, _ = solver.forward_solve(sources_plus, dt, nt, T0, sensors_xy, store_fields=False)
        J_plus = np.sum((Y_plus - Y_obs)**2) / 2.0

        params_minus = params.copy()
        params_minus[i] -= eps
        sources_minus = [{'x': params_minus[j*3], 'y': params_minus[j*3+1], 'q': params_minus[j*3+2]}
                         for j in range(n_sources)]
        Y_minus, _ = solver.forward_solve(sources_minus, dt, nt, T0, sensors_xy, store_fields=False)
        J_minus = np.sum((Y_minus - Y_obs)**2) / 2.0

        fd_grad[i] = (J_plus - J_minus) / (2 * eps)
        param_name = ['x', 'y', 'q'][i % 3]
        print(f"  d/d{param_name}: J_plus={J_plus:.6f}, J_minus={J_minus:.6f}, grad={fd_grad[i]:.6f}")

    print(f"\n{'='*60}")
    print(f"Adjoint gradient: {adj_grad}")
    print(f"FD gradient:      {fd_grad}")

    rel_error = np.abs(adj_grad - fd_grad) / (np.abs(fd_grad) + 1e-12)
    print(f"Relative errors:  {rel_error}")
    print(f"Max rel error:    {np.max(rel_error):.2e}")

    if np.max(rel_error) < 0.1:
        print("\n[OK] Gradients validated!")
    else:
        print("\n[FAIL] Gradient mismatch - check implementation")
        print("\nRatio (FD/Adjoint):", fd_grad / (adj_grad + 1e-20))

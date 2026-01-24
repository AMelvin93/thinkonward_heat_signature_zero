"""
Gradient verification test for checkpointed adjoint method.

The previous adjoint implementation had the ADI operations in the wrong order.
This script tests a corrected discrete adjoint and verifies against finite differences.
"""

import os
import sys
import numpy as np
from scipy.sparse.linalg import splu

_project_root = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, _project_root)
sys.path.insert(0, os.path.join(_project_root, 'data', 'Heat_Signature_zero-starter_notebook'))
from simulator import Heat2D


class CorrectedAdjointSolver:
    """
    Heat equation solver with corrected discrete adjoint gradient computation.

    Key fix: The adjoint of the ADI scheme must reverse the order of operations:
    Forward:  U' = Ay^{-1} * (I+rLx) * Ax^{-1} * (I+rLy) * U
    Adjoint: λ_n = (I+rLy) * Ax^{-1} * (I+rLx) * Ay^{-1} * λ_{n+1}
    """

    def __init__(self, Lx, Ly, nx, ny, kappa, bc, dt, nt, T0, sensors_xy, Y_observed, q_range):
        self.Lx, self.Ly = Lx, Ly
        self.nx, self.ny = nx, ny
        self.kappa = kappa
        self.bc = bc
        self.dt = dt
        self.nt = nt
        self.T0 = T0
        self.sensors_xy = np.array(sensors_xy)
        self.Y_observed = Y_observed
        self.q_range = q_range

        self.dx = Lx / (nx - 1)
        self.dy = Ly / (ny - 1)
        self.x = np.linspace(0, Lx, nx)
        self.y = np.linspace(0, Ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")

        # Build solver
        self.solver = Heat2D(Lx, Ly, nx, ny, kappa, bc=bc)

        # Pre-compute ADI matrices and factorizations
        r = kappa * dt / 2.0
        self.r = r

        # Forward ADI: Ax = (I - r*Lx), Ay = (I - r*Ly)
        self.Ax = (self.solver.Ix_1D - r * self.solver.Lx_1D).tocsc()
        self.Ay = (self.solver.Iy_1D - r * self.solver.Ly_1D).tocsc()
        self.Ax_lu = splu(self.Ax)
        self.Ay_lu = splu(self.Ay)

        # Lx and Ly for explicit terms
        self.Lx_1D = self.solver.Lx_1D
        self.Ly_1D = self.solver.Ly_1D

        # Source spread
        max_ds = max(self.dx, self.dy)
        self.source_sigma = 2.5 * max_ds

        self.n_evals = 0

    def _gaussian_field(self, x0, y0, q=1.0):
        """Create Gaussian source field."""
        r_sq = (self.X - x0) ** 2 + (self.Y - y0) ** 2
        G = np.exp(-r_sq / (2.0 * self.source_sigma**2))
        integral_G = np.sum(G) * self.dx * self.dy
        if integral_G < 1e-10:
            return np.zeros((self.nx, self.ny))
        return G * q / integral_G

    def _gaussian_gradient_x(self, x0, y0, q=1.0):
        """Gradient of Gaussian field w.r.t. x0."""
        r_sq = (self.X - x0) ** 2 + (self.Y - y0) ** 2
        G = np.exp(-r_sq / (2.0 * self.source_sigma**2))
        integral_G = np.sum(G) * self.dx * self.dy
        if integral_G < 1e-10:
            return np.zeros((self.nx, self.ny))
        # dG/dx0 = G * (x - x0) / sigma^2
        dG_dx0 = G * (self.X - x0) / (self.source_sigma**2)
        return dG_dx0 * q / integral_G

    def _gaussian_gradient_y(self, x0, y0, q=1.0):
        """Gradient of Gaussian field w.r.t. y0."""
        r_sq = (self.X - x0) ** 2 + (self.Y - y0) ** 2
        G = np.exp(-r_sq / (2.0 * self.source_sigma**2))
        integral_G = np.sum(G) * self.dx * self.dy
        if integral_G < 1e-10:
            return np.zeros((self.nx, self.ny))
        dG_dy0 = G * (self.Y - y0) / (self.source_sigma**2)
        return dG_dy0 * q / integral_G

    def _interpolate_at_sensors(self, U_field):
        """Interpolate temperature at sensor locations."""
        return self.solver.sample_sensors(U_field, self.sensors_xy)

    def _sensor_to_field(self, sensor_values):
        """Distribute sensor values to grid (adjoint of sensor interpolation)."""
        S = np.zeros((self.nx, self.ny))
        for i, (sx, sy) in enumerate(self.sensors_xy):
            fx = sx / self.dx
            fy = sy / self.dy
            ix = np.clip(int(np.floor(fx)), 0, self.nx - 2)
            iy = np.clip(int(np.floor(fy)), 0, self.ny - 2)
            tx = fx - ix
            ty = fy - iy

            val = sensor_values[i]
            S[ix, iy] += (1 - tx) * (1 - ty) * val
            S[ix + 1, iy] += tx * (1 - ty) * val
            S[ix, iy + 1] += (1 - tx) * ty * val
            S[ix + 1, iy + 1] += tx * ty * val
        return S

    def forward_solve(self, sources_list):
        """Run forward simulation, return temperature at sensors and all temperature fields."""
        sources = [{'x': s[0], 'y': s[1], 'q': s[2]} for s in sources_list]
        times, Us = self.solver.solve(dt=self.dt, nt=self.nt, T0=self.T0, sources=sources)
        Y_sim = np.array([self._interpolate_at_sensors(U) for U in Us])
        self.n_evals += 1
        return Y_sim, Us

    def forward_solve_with_checkpoints(self, sources_list, checkpoint_interval=50):
        """Forward solve storing checkpoints every checkpoint_interval timesteps."""
        # Initialize
        U = np.full((self.nx, self.ny), self.T0, dtype=float)

        # Source field
        max_ds = max(self.dx, self.dy)
        sigma = 2.5 * max_ds
        S_field = np.zeros((self.nx, self.ny))
        for x0, y0, q in sources_list:
            r_sq = (self.X - x0) ** 2 + (self.Y - y0) ** 2
            G = np.exp(-r_sq / (2.0 * sigma**2))
            integral_G = np.sum(G) * self.dx * self.dy
            if integral_G > 1e-10:
                S_field += G * q / integral_G
        S_scaled = S_field * self.dt / 2.0

        checkpoints = {0: U.copy()}
        Us = [U.copy()]
        Y_sim = [self._interpolate_at_sensors(U)]

        for n in range(self.nt):
            # ADI Step 1: Implicit in X, Explicit in Y
            RHS_x = U.copy()
            for i in range(self.nx):
                RHS_x[i, :] += self.r * (self.Ly_1D @ U[i, :])
            RHS_x += S_scaled

            U_star = np.zeros_like(U)
            for j in range(self.ny):
                U_star[:, j] = self.Ax_lu.solve(RHS_x[:, j])

            # ADI Step 2: Implicit in Y, Explicit in X
            RHS_y = U_star.copy()
            for j in range(self.ny):
                RHS_y[:, j] += self.r * (self.Lx_1D @ U_star[:, j])
            RHS_y += S_scaled

            U_next = np.zeros_like(U)
            for i in range(self.nx):
                U_next[i, :] = self.Ay_lu.solve(RHS_y[i, :])

            U = U_next
            Us.append(U.copy())
            Y_sim.append(self._interpolate_at_sensors(U))

            if (n + 1) % checkpoint_interval == 0:
                checkpoints[n + 1] = U.copy()

        self.n_evals += 1
        return np.array(Y_sim), Us, checkpoints, S_scaled

    def adjoint_timestep_corrected(self, lambda_in):
        """
        Corrected adjoint of one ADI timestep w.r.t. U.

        Forward: U' = Ay^{-1} * (I+rLx) * Ax^{-1} * (I+rLy) * U + source terms
        Adjoint: λ_n = (I+rLy)^T * Ax^{-T} * (I+rLx)^T * Ay^{-T} * λ_{n+1}

        Since Lx, Ly, Ax, Ay are symmetric:
        λ_n = (I+rLy) * Ax^{-1} * (I+rLx) * Ay^{-1} * λ_{n+1}
        """
        # Step 1: Ay^{-1} * lambda_in
        lambda_temp = np.zeros_like(lambda_in)
        for i in range(self.nx):
            lambda_temp[i, :] = self.Ay_lu.solve(lambda_in[i, :])

        # Step 2: (I + r*Lx) * lambda_temp
        lambda_temp2 = lambda_temp.copy()
        for j in range(self.ny):
            lambda_temp2[:, j] += self.r * (self.Lx_1D @ lambda_temp[:, j])

        # Step 3: Ax^{-1} * lambda_temp2
        lambda_temp3 = np.zeros_like(lambda_temp2)
        for j in range(self.ny):
            lambda_temp3[:, j] = self.Ax_lu.solve(lambda_temp2[:, j])

        # Step 4: (I + r*Ly) * lambda_temp3
        lambda_out = lambda_temp3.copy()
        for i in range(self.nx):
            lambda_out[i, :] += self.r * (self.Ly_1D @ lambda_temp3[i, :])

        return lambda_out

    def adjoint_source_term(self, lambda_in):
        """
        Compute adjoint of one ADI timestep w.r.t. source term S.

        The source S appears in BOTH ADI half-steps:
        Step 1: Ax * U* = (I+rLy)*U + S_scaled
        Step 2: Ay * U' = (I+rLx)*U* + S_scaled

        Total d(U')/d(S) = Ay^{-1} + Ay^{-1} * (I+rLx) * Ax^{-1}

        So: d(L)/d(S) = [Ay^{-1} + Ay^{-1} * (I+rLx) * Ax^{-1}]^T * lambda
                      = Ay^{-1} * lambda + Ax^{-1} * (I+rLx) * Ay^{-1} * lambda
        """
        # Term 1: Ay^{-1} * lambda
        term1 = np.zeros_like(lambda_in)
        for i in range(self.nx):
            term1[i, :] = self.Ay_lu.solve(lambda_in[i, :])

        # Term 2: Ax^{-1} * (I+rLx) * Ay^{-1} * lambda
        # First: Ay^{-1} * lambda (already computed as term1)
        # Then: (I+rLx) * term1
        temp = term1.copy()
        for j in range(self.ny):
            temp[:, j] += self.r * (self.Lx_1D @ term1[:, j])
        # Finally: Ax^{-1} * temp
        term2 = np.zeros_like(temp)
        for j in range(self.ny):
            term2[:, j] = self.Ax_lu.solve(temp[:, j])

        return term1 + term2

    def compute_rmse_and_gradient_1src(self, x, y):
        """
        Compute RMSE and gradient w.r.t. (x, y) for 1-source.
        Uses optimal intensity q computed via least squares.
        """
        # Forward solve with unit intensity
        Y_unit, Us_unit, checkpoints, S_scaled = self.forward_solve_with_checkpoints([(x, y, 1.0)])

        # Compute optimal intensity
        Y_unit_flat = Y_unit.flatten()
        Y_obs_flat = self.Y_observed.flatten()
        denom = np.dot(Y_unit_flat, Y_unit_flat)
        q_opt = np.dot(Y_unit_flat, Y_obs_flat) / denom if denom > 1e-10 else 1.0
        q_opt = np.clip(q_opt, self.q_range[0], self.q_range[1])

        # Compute RMSE
        Y_pred = q_opt * Y_unit
        rmse = np.sqrt(np.mean((Y_pred - self.Y_observed) ** 2))

        # Scale temperature fields by q_opt
        Us = [U * q_opt for U in Us_unit]

        # Compute residuals: d(RMSE)/d(Y_sim) = (Y_sim - Y_obs) / (N * RMSE)
        residuals = (Y_pred - self.Y_observed)  # (nt+1, n_sensors)
        n_obs = self.Y_observed.size
        # d(RMSE)/d(Y_sim_ij) = (Y_sim_ij - Y_obs_ij) / (n_obs * RMSE)
        # For simplicity, we'll use d(RMSE^2)/d(Y_sim) = 2*(Y_sim - Y_obs)/n_obs
        # Then d(RMSE)/d(params) = d(RMSE^2)/d(params) / (2*RMSE)
        dRMSE2_dY = 2.0 * residuals / n_obs  # (nt+1, n_sensors)

        # Run adjoint backward
        # The adjoint propagates the gradient of the loss through the simulation
        #
        # Key insight: At each timestep, the source S affects T^{t+1} via the ADI step.
        # The adjoint of this is: grad += λ^{t+1} · (d(ADI)/d(S) * d(S)/d(x0))
        # For ADI with source, d(ADI)/d(S) ≈ I (source adds directly to T)
        # So: grad += λ^{t+1} · d(S)/d(x0) * dt
        #
        # Backward pass order at each step:
        # 1. Add observation gradient at current time (λ += d(L)/d(T))
        # 2. Accumulate source gradient (grad += λ · d(S)/d(x0) * dt)
        # 3. Propagate to previous timestep (λ_prev = adjoint_step(λ))

        # Source gradient field (constant over time)
        dG_dx = self._gaussian_gradient_x(x, y, q_opt)
        dG_dy = self._gaussian_gradient_y(x, y, q_opt)

        # Initialize at final time
        lambda_field = np.zeros((self.nx, self.ny))
        grad_x_total = 0.0
        grad_y_total = 0.0

        # Backward pass from time nt to 0
        # Key insight: Source at time t affects T^{t+1}, so we need λ^{t+1} for source gradient
        #
        # At each step:
        # 1. λ currently holds partial derivatives from times > t (including observation at t+1...nt)
        # 2. Add observation gradient at current time t: λ += d(L)/d(T^t)
        # 3. NOW λ holds d(L)/d(T^t) - use this for source gradient at time t-1
        # 4. Propagate: λ_{new} = adjoint_step(λ)

        for t_idx in range(self.nt, -1, -1):
            # 1. Add observation gradient at this timestep
            sensor_grad = dRMSE2_dY[t_idx]
            field_grad = self._sensor_to_field(sensor_grad)
            lambda_field += field_grad

            # At this point, lambda_field = d(L)/d(T^t_idx)
            # The source at timestep (t_idx - 1) affects T^t_idx
            # So we compute source gradient for (t_idx - 1) using current lambda

            # 2. Propagate adjoint backward one timestep
            # This computes d(L)/d(T^{t_idx-1}) from d(L)/d(T^t_idx)
            if t_idx > 0:
                # BEFORE propagating, accumulate source gradient for timestep (t_idx - 1)
                # Source at (t_idx - 1) -> T^t_idx, so gradient uses λ^t_idx (current lambda)
                adj_for_source = self.adjoint_source_term(lambda_field)
                grad_x_total += np.sum(adj_for_source * dG_dx) * self.dx * self.dy
                grad_y_total += np.sum(adj_for_source * dG_dy) * self.dx * self.dy

                # Now propagate
                lambda_field = self.adjoint_timestep_corrected(lambda_field)

        # Scale to get d(RMSE)/d(params) instead of d(RMSE^2)/d(params)
        grad_x = grad_x_total / (2.0 * rmse) if rmse > 1e-10 else 0.0
        grad_y = grad_y_total / (2.0 * rmse) if rmse > 1e-10 else 0.0

        return rmse, np.array([grad_x, grad_y]), q_opt


def finite_diff_gradient(func, x, eps=1e-5):
    """Compute gradient using central differences."""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += eps
        x_minus = x.copy()
        x_minus[i] -= eps
        grad[i] = (func(x_plus) - func(x_minus)) / (2 * eps)
    return grad


def test_gradient_accuracy():
    """Test adjoint gradient against finite differences."""
    import pickle

    # Load test data
    data_path = os.path.join(_project_root, 'data', 'heat-signature-zero-test-data.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # Test on a 1-source sample
    samples = data['samples']
    meta = data['meta']

    # Find a 1-source sample
    for idx, sample in enumerate(samples[:10]):
        if sample['n_sources'] == 1:
            break

    print(f"\n=== Testing on sample {idx} (1-source) ===")

    kappa = sample['sample_metadata']['kappa']
    bc = sample['sample_metadata']['bc']
    nt = sample['sample_metadata']['nt']
    T0 = sample['sample_metadata']['T0']
    dt = meta['dt']
    sensors_xy = np.array(sample['sensors_xy'])
    Y_observed = sample['Y_noisy']
    q_range = (0.5, 2.0)

    # Create solver
    solver = CorrectedAdjointSolver(
        Lx=2.0, Ly=1.0, nx=100, ny=50,
        kappa=kappa, bc=bc, dt=dt, nt=nt, T0=T0,
        sensors_xy=sensors_xy, Y_observed=Y_observed, q_range=q_range
    )

    # Test point (near center of domain)
    x_test = 1.0
    y_test = 0.5

    print(f"\nTest point: x={x_test}, y={y_test}")
    print(f"Grid: {solver.nx}x{solver.ny}, nt={solver.nt}, dt={solver.dt}")
    print(f"dx={solver.dx:.6f}, dy={solver.dy:.6f}")

    # Compute adjoint gradient
    rmse, grad_adj, q_opt = solver.compute_rmse_and_gradient_1src(x_test, y_test)
    print(f"RMSE = {rmse:.6f}, q_opt = {q_opt:.4f}")
    print(f"Adjoint gradient: dRMSE/dx = {grad_adj[0]:.6f}, dRMSE/dy = {grad_adj[1]:.6f}")

    # Compute finite difference gradient
    def rmse_func(params):
        x, y = params
        rmse_val, _, _ = solver.compute_rmse_and_gradient_1src(x, y)
        return rmse_val

    grad_fd = finite_diff_gradient(rmse_func, np.array([x_test, y_test]), eps=1e-4)
    print(f"Finite diff gradient: dRMSE/dx = {grad_fd[0]:.6f}, dRMSE/dy = {grad_fd[1]:.6f}")

    # Compare
    print("\n=== Gradient Comparison ===")
    print(f"  dRMSE/dx: adjoint={grad_adj[0]:.6f}, FD={grad_fd[0]:.6f}, ratio={grad_adj[0]/grad_fd[0] if abs(grad_fd[0]) > 1e-10 else float('nan'):.6f}")
    print(f"  dRMSE/dy: adjoint={grad_adj[1]:.6f}, FD={grad_fd[1]:.6f}, ratio={grad_adj[1]/grad_fd[1] if abs(grad_fd[1]) > 1e-10 else float('nan'):.6f}")

    # Check if gradients are close
    rel_err_x = abs(grad_adj[0] - grad_fd[0]) / max(abs(grad_fd[0]), 1e-10)
    rel_err_y = abs(grad_adj[1] - grad_fd[1]) / max(abs(grad_fd[1]), 1e-10)

    print(f"\nRelative errors: x={rel_err_x:.2%}, y={rel_err_y:.2%}")

    # Additional debug: what's the expected scaling?
    ratio = grad_adj[0] / grad_fd[0]
    print(f"\nDebug:")
    print(f"  Ratio: {ratio:.6f}")
    print(f"  1/ratio: {1/ratio:.2f}")
    print(f"  nt * ratio: {solver.nt * ratio:.4f}")
    print(f"  Expected if missing 1/nt: {1/solver.nt:.6f}")
    print(f"  Expected if missing dt: {solver.dt:.6f}")

    if rel_err_x < 0.1 and rel_err_y < 0.1:
        print("✓ Gradients match within 10% - SUCCESS!")
        return True
    else:
        print("✗ Gradients do NOT match - FAILED")
        return False


def test_single_timestep_adjoint():
    """Test adjoint of a single ADI timestep against finite differences."""
    import pickle

    # Load test data
    data_path = os.path.join(_project_root, 'data', 'heat-signature-zero-test-data.pkl')
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    sample = data['samples'][0]
    meta = data['meta']

    kappa = sample['sample_metadata']['kappa']
    bc = sample['sample_metadata']['bc']
    T0 = sample['sample_metadata']['T0']
    dt = meta['dt']

    print("\n=== Testing single ADI timestep adjoint ===")

    # Create a simple test: forward ADI step and its adjoint
    nx, ny = 20, 10  # Small grid for fast testing
    Lx, Ly = 2.0, 1.0
    dx, dy = Lx / (nx - 1), Ly / (ny - 1)

    solver = Heat2D(Lx, Ly, nx, ny, kappa, bc=bc)
    r = kappa * dt / 2.0

    # ADI matrices
    Ax = (solver.Ix_1D - r * solver.Lx_1D).tocsc()
    Ay = (solver.Iy_1D - r * solver.Ly_1D).tocsc()
    Ax_lu = splu(Ax)
    Ay_lu = splu(Ay)
    Lx_1D = solver.Lx_1D
    Ly_1D = solver.Ly_1D

    def adi_forward_step(U, S_scaled):
        """Forward ADI step: U' = ADI(U, S_scaled)"""
        # Step 1: Implicit X, Explicit Y
        RHS_x = U.copy()
        for i in range(nx):
            RHS_x[i, :] += r * (Ly_1D @ U[i, :])
        RHS_x += S_scaled

        U_star = np.zeros_like(U)
        for j in range(ny):
            U_star[:, j] = Ax_lu.solve(RHS_x[:, j])

        # Step 2: Implicit Y, Explicit X
        RHS_y = U_star.copy()
        for j in range(ny):
            RHS_y[:, j] += r * (Lx_1D @ U_star[:, j])
        RHS_y += S_scaled

        U_next = np.zeros_like(U)
        for i in range(nx):
            U_next[i, :] = Ay_lu.solve(RHS_y[i, :])

        return U_next

    def adi_adjoint_step(lambda_in):
        """Adjoint of ADI step (corrected order)"""
        # Step 1: Ay^{-1}
        lambda_temp = np.zeros_like(lambda_in)
        for i in range(nx):
            lambda_temp[i, :] = Ay_lu.solve(lambda_in[i, :])

        # Step 2: (I + r*Lx)
        lambda_temp2 = lambda_temp.copy()
        for j in range(ny):
            lambda_temp2[:, j] += r * (Lx_1D @ lambda_temp[:, j])

        # Step 3: Ax^{-1}
        lambda_temp3 = np.zeros_like(lambda_temp2)
        for j in range(ny):
            lambda_temp3[:, j] = Ax_lu.solve(lambda_temp2[:, j])

        # Step 4: (I + r*Ly)
        lambda_out = lambda_temp3.copy()
        for i in range(nx):
            lambda_out[i, :] += r * (Ly_1D @ lambda_temp3[i, :])

        return lambda_out

    # Random input state and output gradient
    np.random.seed(42)
    U_in = np.random.randn(nx, ny)
    S_scaled = np.random.randn(nx, ny) * 0.01
    grad_out = np.random.randn(nx, ny)

    # Forward pass
    U_out = adi_forward_step(U_in, S_scaled)

    # Loss: L = sum(U_out * grad_out)
    loss = np.sum(U_out * grad_out)

    # Adjoint pass
    lambda_in = adi_adjoint_step(grad_out)

    # Adjoint gradient w.r.t. U_in: should be lambda_in
    # Adjoint gradient w.r.t. S_scaled: needs separate computation

    # Finite diff gradient w.r.t. U_in[5,5]
    eps = 1e-5
    i, j = 5, 5

    U_in_plus = U_in.copy()
    U_in_plus[i, j] += eps
    U_out_plus = adi_forward_step(U_in_plus, S_scaled)
    loss_plus = np.sum(U_out_plus * grad_out)

    U_in_minus = U_in.copy()
    U_in_minus[i, j] -= eps
    U_out_minus = adi_forward_step(U_in_minus, S_scaled)
    loss_minus = np.sum(U_out_minus * grad_out)

    fd_grad_U = (loss_plus - loss_minus) / (2 * eps)
    adj_grad_U = lambda_in[i, j]

    print(f"\nGradient w.r.t. U_in[{i},{j}]:")
    print(f"  Adjoint: {adj_grad_U:.8f}")
    print(f"  Finite diff: {fd_grad_U:.8f}")
    print(f"  Ratio: {adj_grad_U/fd_grad_U if abs(fd_grad_U) > 1e-10 else float('nan'):.6f}")

    # Also test gradient w.r.t. S_scaled
    # The adjoint of S_scaled is the sum of contributions from both ADI half-steps
    # From Step 1: Ax^{-1} * (RHS_x + S_scaled) -> d/dS = Ax^{-T} * grad_after_step1
    # From Step 2: Ay^{-1} * (RHS_y + S_scaled) -> d/dS = Ay^{-T} * grad_out

    # More simply: the source appears in both RHS_x and RHS_y
    # Total d(L)/d(S_scaled) = d(L)/d(RHS_x) + d(L)/d(RHS_y)

    # Let's compute this numerically
    S_plus = S_scaled.copy()
    S_plus[i, j] += eps
    U_out_Splus = adi_forward_step(U_in, S_plus)
    loss_Splus = np.sum(U_out_Splus * grad_out)

    S_minus = S_scaled.copy()
    S_minus[i, j] -= eps
    U_out_Sminus = adi_forward_step(U_in, S_minus)
    loss_Sminus = np.sum(U_out_Sminus * grad_out)

    fd_grad_S = (loss_Splus - loss_Sminus) / (2 * eps)

    # For adjoint w.r.t. S, we need to compute:
    # d(U_next)/d(S) = Ay^{-1} + Ay^{-1} @ B_y @ Ax^{-1}
    # where B_y = (I + r*Lx)
    #
    # So: d(L)/d(S) = grad_out^T @ [Ay^{-1} + Ay^{-1} @ B_y @ Ax^{-1}]
    #              = (Ay^{-1} @ grad_out) + (Ax^{-1} @ B_y @ Ay^{-1} @ grad_out)
    #              = (Ay^{-1} @ grad_out) + adi_adjoint_step(grad_out)

    # First term: Ay^{-1} @ grad_out
    adj_S_term1 = np.zeros_like(grad_out)
    for i_grid in range(nx):
        adj_S_term1[i_grid, :] = Ay_lu.solve(grad_out[i_grid, :])

    # Second term: adi_adjoint_step(grad_out) = (I + r*Ly) @ Ax^{-1} @ (I + r*Lx) @ Ay^{-1} @ grad_out
    adj_S_term2 = adi_adjoint_step(grad_out)

    adj_grad_S_total = adj_S_term1 + adj_S_term2
    adj_grad_S = adj_grad_S_total[i, j]

    print(f"\nGradient w.r.t. S_scaled[{i},{j}]:")
    print(f"  Adjoint: {adj_grad_S:.8f}")
    print(f"  Finite diff: {fd_grad_S:.8f}")
    print(f"  Ratio: {adj_grad_S/fd_grad_S if abs(fd_grad_S) > 1e-10 else float('nan'):.6f}")

    # Check if adjoint for U is correct
    rel_err_U = abs(adj_grad_U - fd_grad_U) / max(abs(fd_grad_U), 1e-10)
    rel_err_S = abs(adj_grad_S - fd_grad_S) / max(abs(fd_grad_S), 1e-10)

    success = True
    if rel_err_U < 0.01:
        print(f"\n✓ Adjoint w.r.t. U is CORRECT (error: {rel_err_U:.2%})")
    else:
        print(f"\n✗ Adjoint w.r.t. U is WRONG (error: {rel_err_U:.2%})")
        success = False

    if rel_err_S < 0.01:
        print(f"✓ Adjoint w.r.t. S is CORRECT (error: {rel_err_S:.2%})")
    else:
        print(f"✗ Adjoint w.r.t. S is WRONG (error: {rel_err_S:.2%})")
        success = False

    return success


if __name__ == "__main__":
    # First test single timestep adjoint
    single_ok = test_single_timestep_adjoint()

    print("\n" + "="*60)

    # Then test full gradient
    full_ok = test_gradient_accuracy()

    print(f"\n=== Final Results ===")
    print(f"Single timestep adjoint: {'SUCCESS' if single_ok else 'FAILED'}")
    print(f"Full gradient: {'SUCCESS' if full_ok else 'FAILED'}")

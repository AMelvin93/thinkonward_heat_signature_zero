"""
Visualization module using Plotly for heat source estimation.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'data', 'Heat_Signature_zero-starter_notebook'))
from simulator import Heat2D


def plot_sample_overview(
    sample: Dict,
    meta: Dict,
    estimated_sources: Optional[List[Tuple[float, float, float]]] = None,
    true_sources: Optional[List[Dict]] = None,
    Lx: float = 2.0,
    Ly: float = 1.0,
) -> go.Figure:
    """
    Create an overview plot of a sample showing domain, sensors, and sources.

    Args:
        sample: Sample dict
        meta: Meta dict with dt
        estimated_sources: List of (x, y, q) tuples for estimated sources
        true_sources: List of dicts with 'x', 'y', 'q' for ground truth
        Lx, Ly: Domain dimensions

    Returns:
        Plotly Figure
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Spatial Layout', 'Sensor Temperature History'),
        column_widths=[0.4, 0.6],
    )

    sensors_xy = sample['sensors_xy']
    Y_noisy = sample['Y_noisy']
    dt = meta['dt']
    n_sensors = sensors_xy.shape[0]

    # Color palette for sensors
    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    # === Left plot: Spatial layout ===

    # Domain boundary
    fig.add_trace(
        go.Scatter(
            x=[0, Lx, Lx, 0, 0],
            y=[0, 0, Ly, Ly, 0],
            mode='lines',
            line=dict(color='black', width=2),
            name='Domain',
            showlegend=False,
        ),
        row=1, col=1
    )

    # Sensors
    fig.add_trace(
        go.Scatter(
            x=sensors_xy[:, 0],
            y=sensors_xy[:, 1],
            mode='markers+text',
            marker=dict(size=12, color=[colors[i % len(colors)] for i in range(n_sensors)]),
            text=[f'S{i}' for i in range(n_sensors)],
            textposition='top center',
            name='Sensors',
        ),
        row=1, col=1
    )

    # True sources (if available)
    if true_sources:
        fig.add_trace(
            go.Scatter(
                x=[s['x'] for s in true_sources],
                y=[s['y'] for s in true_sources],
                mode='markers+text',
                marker=dict(size=15, color='red', symbol='star'),
                text=[f"q={s['q']:.2f}" for s in true_sources],
                textposition='bottom center',
                name='True Sources',
            ),
            row=1, col=1
        )

    # Estimated sources (if available)
    if estimated_sources:
        fig.add_trace(
            go.Scatter(
                x=[s[0] for s in estimated_sources],
                y=[s[1] for s in estimated_sources],
                mode='markers+text',
                marker=dict(size=15, color='green', symbol='star-triangle-up'),
                text=[f"q={s[2]:.2f}" for s in estimated_sources],
                textposition='top center',
                name='Estimated Sources',
            ),
            row=1, col=1
        )

    # === Right plot: Temperature history ===
    time = np.arange(Y_noisy.shape[0]) * dt

    for i in range(n_sensors):
        fig.add_trace(
            go.Scatter(
                x=time,
                y=Y_noisy[:, i],
                mode='lines',
                line=dict(color=colors[i % len(colors)], width=2),
                name=f'Sensor {i}',
            ),
            row=1, col=2
        )

    # Layout
    metadata = sample['sample_metadata']
    title = (
        f"Sample: {sample['sample_id']} | "
        f"Sources: {sample['n_sources']} | "
        f"BC: {metadata['bc']} | "
        f"κ={metadata['kappa']} | "
        f"noise={metadata['noise_std']}"
    )

    fig.update_layout(
        title=title,
        height=500,
        showlegend=True,
    )

    fig.update_xaxes(title_text='x', row=1, col=1, range=[0, Lx])
    fig.update_yaxes(title_text='y', row=1, col=1, range=[0, Ly], scaleanchor='x', scaleratio=1)
    fig.update_xaxes(title_text='Time (s)', row=1, col=2)
    fig.update_yaxes(title_text='Temperature', row=1, col=2)

    return fig


def plot_estimation_comparison(
    sample: Dict,
    meta: Dict,
    estimated_sources: List[Tuple[float, float, float]],
    true_sources: Optional[List[Dict]] = None,
    Lx: float = 2.0,
    Ly: float = 1.0,
    nx: int = 100,
    ny: int = 50,
) -> go.Figure:
    """
    Compare observed vs simulated sensor readings.
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Observed vs Predicted', 'Residuals'),
    )

    sensors_xy = sample['sensors_xy']
    Y_observed = sample['Y_noisy']
    dt = meta['dt']
    n_sensors = sensors_xy.shape[0]

    # Simulate with estimated sources
    metadata = sample['sample_metadata']
    solver = Heat2D(Lx, Ly, nx, ny, metadata['kappa'], bc=metadata['bc'])
    sources = [{'x': s[0], 'y': s[1], 'q': s[2]} for s in estimated_sources]
    times, Us = solver.solve(dt=dt, nt=metadata['nt'], T0=metadata['T0'], sources=sources)
    Y_pred = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])

    time = np.arange(Y_observed.shape[0]) * dt
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # === Left: Observed vs Predicted ===
    for i in range(n_sensors):
        # Observed (solid)
        fig.add_trace(
            go.Scatter(
                x=time, y=Y_observed[:, i],
                mode='lines',
                line=dict(color=colors[i % len(colors)], width=2),
                name=f'S{i} Observed',
                legendgroup=f'sensor{i}',
            ),
            row=1, col=1
        )
        # Predicted (dashed)
        fig.add_trace(
            go.Scatter(
                x=time, y=Y_pred[:, i],
                mode='lines',
                line=dict(color=colors[i % len(colors)], width=2, dash='dash'),
                name=f'S{i} Predicted',
                legendgroup=f'sensor{i}',
            ),
            row=1, col=1
        )

    # === Right: Residuals ===
    residuals = Y_pred - Y_observed
    for i in range(n_sensors):
        fig.add_trace(
            go.Scatter(
                x=time, y=residuals[:, i],
                mode='lines',
                line=dict(color=colors[i % len(colors)], width=2),
                name=f'S{i} Residual',
                showlegend=False,
            ),
            row=1, col=2
        )

    # Add zero line
    fig.add_hline(y=0, line_dash='dash', line_color='gray', row=1, col=2)

    # Compute RMSE
    rmse = np.sqrt(np.mean(residuals**2))

    fig.update_layout(
        title=f"Estimation Result | RMSE: {rmse:.4f}",
        height=450,
    )

    fig.update_xaxes(title_text='Time (s)', row=1, col=1)
    fig.update_yaxes(title_text='Temperature', row=1, col=1)
    fig.update_xaxes(title_text='Time (s)', row=1, col=2)
    fig.update_yaxes(title_text='Residual', row=1, col=2)

    return fig


def plot_optimization_history(history: List[Dict]) -> go.Figure:
    """
    Plot optimization convergence history.
    """
    iterations = list(range(len(history)))
    rmse_values = [h['rmse'] for h in history]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=rmse_values,
            mode='lines+markers',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4),
            name='RMSE',
        )
    )

    fig.update_layout(
        title='Optimization Convergence',
        xaxis_title='Iteration',
        yaxis_title='RMSE',
        height=400,
    )

    return fig


def plot_source_search_space(
    sample: Dict,
    meta: Dict,
    estimated_sources: List[Tuple[float, float, float]],
    true_sources: Optional[List[Dict]] = None,
    Lx: float = 2.0,
    Ly: float = 1.0,
    nx: int = 100,
    ny: int = 50,
    grid_resolution: int = 20,
) -> go.Figure:
    """
    Visualize the loss landscape for source position (for single-source case).
    Shows a heatmap of RMSE as a function of source position.
    """
    if sample['n_sources'] != 1:
        print("Loss landscape visualization only supported for single-source samples")
        return None

    sensors_xy = sample['sensors_xy']
    Y_observed = sample['Y_noisy']
    metadata = sample['sample_metadata']
    dt = meta['dt']

    solver = Heat2D(Lx, Ly, nx, ny, metadata['kappa'], bc=metadata['bc'])

    # Create grid
    x_grid = np.linspace(0.05 * Lx, 0.95 * Lx, grid_resolution)
    y_grid = np.linspace(0.05 * Ly, 0.95 * Ly, grid_resolution)

    # Use the estimated q value
    q_est = estimated_sources[0][2] if estimated_sources else 1.0

    # Compute RMSE for each grid point
    rmse_grid = np.zeros((grid_resolution, grid_resolution))

    for i, x in enumerate(x_grid):
        for j, y in enumerate(y_grid):
            sources = [{'x': x, 'y': y, 'q': q_est}]
            times, Us = solver.solve(dt=dt, nt=metadata['nt'], T0=metadata['T0'], sources=sources)
            Y_pred = np.array([solver.sample_sensors(U, sensors_xy) for U in Us])
            rmse_grid[j, i] = np.sqrt(np.mean((Y_pred - Y_observed) ** 2))

    fig = go.Figure()

    # Heatmap
    fig.add_trace(
        go.Heatmap(
            x=x_grid,
            y=y_grid,
            z=rmse_grid,
            colorscale='Viridis_r',
            colorbar=dict(title='RMSE'),
        )
    )

    # Sensors
    fig.add_trace(
        go.Scatter(
            x=sensors_xy[:, 0],
            y=sensors_xy[:, 1],
            mode='markers',
            marker=dict(size=12, color='white', symbol='circle', line=dict(width=2, color='black')),
            name='Sensors',
        )
    )

    # Estimated source
    if estimated_sources:
        fig.add_trace(
            go.Scatter(
                x=[estimated_sources[0][0]],
                y=[estimated_sources[0][1]],
                mode='markers',
                marker=dict(size=15, color='lime', symbol='star', line=dict(width=2, color='black')),
                name='Estimated',
            )
        )

    # True source
    if true_sources:
        fig.add_trace(
            go.Scatter(
                x=[true_sources[0]['x']],
                y=[true_sources[0]['y']],
                mode='markers',
                marker=dict(size=15, color='red', symbol='star', line=dict(width=2, color='white')),
                name='True',
            )
        )

    fig.update_layout(
        title=f'Loss Landscape (q={q_est:.2f})',
        xaxis_title='x',
        yaxis_title='y',
        height=500,
        width=700,
    )

    fig.update_xaxes(range=[0, Lx])
    fig.update_yaxes(range=[0, Ly], scaleanchor='x', scaleratio=1)

    return fig


def plot_dataset_summary(test_dataset: Dict) -> go.Figure:
    """
    Create summary visualizations of the test dataset.
    """
    samples = test_dataset['samples']

    # Collect statistics
    n_sources_list = [s['n_sources'] for s in samples]
    n_sensors_list = [s['sensors_xy'].shape[0] for s in samples]
    bc_list = [s['sample_metadata']['bc'] for s in samples]
    kappa_list = [s['sample_metadata']['kappa'] for s in samples]
    noise_list = [s['sample_metadata']['noise_std'] for s in samples]

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Number of Sources', 'Number of Sensors', 'Boundary Conditions',
            'Thermal Diffusivity (κ)', 'Noise Level (σ)', 'Sources vs Sensors'
        ),
        specs=[
            [{'type': 'histogram'}, {'type': 'histogram'}, {'type': 'pie'}],
            [{'type': 'pie'}, {'type': 'histogram'}, {'type': 'histogram2d'}],
        ]
    )

    # 1. Number of sources
    fig.add_trace(
        go.Histogram(x=n_sources_list, name='Sources', marker_color='#1f77b4'),
        row=1, col=1
    )

    # 2. Number of sensors
    fig.add_trace(
        go.Histogram(x=n_sensors_list, name='Sensors', marker_color='#ff7f0e'),
        row=1, col=2
    )

    # 3. Boundary conditions (pie)
    bc_counts = {bc: bc_list.count(bc) for bc in set(bc_list)}
    fig.add_trace(
        go.Pie(labels=list(bc_counts.keys()), values=list(bc_counts.values()), name='BC'),
        row=1, col=3
    )

    # 4. Kappa (pie)
    kappa_counts = {str(k): kappa_list.count(k) for k in set(kappa_list)}
    fig.add_trace(
        go.Pie(labels=list(kappa_counts.keys()), values=list(kappa_counts.values()), name='κ'),
        row=2, col=1
    )

    # 5. Noise level
    fig.add_trace(
        go.Histogram(x=noise_list, name='Noise', marker_color='#2ca02c'),
        row=2, col=2
    )

    # 6. Sources vs Sensors heatmap
    fig.add_trace(
        go.Histogram2d(x=n_sources_list, y=n_sensors_list, colorscale='Blues'),
        row=2, col=3
    )

    fig.update_layout(
        title='Test Dataset Summary',
        height=700,
        showlegend=False,
    )

    return fig

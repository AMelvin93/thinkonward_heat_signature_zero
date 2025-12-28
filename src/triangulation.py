"""
Triangulation-based initialization for heat source localization.

Uses heat diffusion physics to estimate source positions from sensor readings:
1. Detect onset/characteristic times at each sensor
2. Estimate distances using diffusion time scales
3. Trilaterate source position from distance estimates

References:
- Heat equation fundamental solution: T(r,t) ~ exp(-r²/(4κt)) / t
- Trilateration: minimize Σ(||p - sensor_i|| - d_i)²
"""

import numpy as np
from scipy.optimize import minimize
from typing import List, Tuple, Optional, Dict
import warnings


def detect_onset_time(
    signal: np.ndarray,
    dt: float,
    threshold_fraction: float = 0.1,
    noise_floor: float = 0.1,
) -> float:
    """
    Detect when a sensor signal starts rising above noise.

    Args:
        signal: Temperature time series for one sensor
        dt: Time step
        threshold_fraction: Fraction of max signal to consider as "onset"
        noise_floor: Expected noise standard deviation

    Returns:
        Onset time in seconds (or inf if no clear onset)
    """
    # Smooth the signal to reduce noise effects
    kernel_size = min(11, len(signal) // 10)
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size >= 3:
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(signal, kernel, mode='same')
    else:
        smoothed = signal

    # Find the maximum temperature
    max_temp = np.max(smoothed)

    # If signal never rises significantly above noise, return inf
    if max_temp < 3 * noise_floor:
        return float('inf')

    # Threshold for onset detection
    threshold = threshold_fraction * max_temp

    # Find first time signal exceeds threshold
    above_threshold = smoothed > threshold
    if not np.any(above_threshold):
        return float('inf')

    onset_idx = np.argmax(above_threshold)
    return onset_idx * dt


def detect_characteristic_time(
    signal: np.ndarray,
    dt: float,
    fraction: float = 0.5,
    noise_floor: float = 0.1,
) -> float:
    """
    Detect when sensor reaches a fraction of its maximum temperature.

    This is more robust than onset detection for noisy signals.

    Args:
        signal: Temperature time series for one sensor
        dt: Time step
        fraction: Fraction of max (e.g., 0.5 for half-max time)
        noise_floor: Expected noise standard deviation

    Returns:
        Characteristic time in seconds (or inf if no clear signal)
    """
    # Smooth the signal
    kernel_size = min(21, len(signal) // 5)
    if kernel_size % 2 == 0:
        kernel_size += 1
    if kernel_size >= 3:
        kernel = np.ones(kernel_size) / kernel_size
        smoothed = np.convolve(signal, kernel, mode='same')
    else:
        smoothed = signal

    max_temp = np.max(smoothed)

    # If signal is weak, return inf
    if max_temp < 3 * noise_floor:
        return float('inf')

    threshold = fraction * max_temp
    above = smoothed > threshold

    if not np.any(above):
        return float('inf')

    return np.argmax(above) * dt


def estimate_distances_from_times(
    times: np.ndarray,
    kappa: float,
    scale_factor: float = 2.0,
) -> np.ndarray:
    """
    Estimate distances from characteristic times using diffusion scaling.

    For heat diffusion: r ~ sqrt(4 * kappa * t)

    Args:
        times: Characteristic times for each sensor
        kappa: Thermal diffusivity
        scale_factor: Calibration factor (default 2.0 for sqrt(4))

    Returns:
        Estimated distances (inf for sensors with no clear signal)
    """
    distances = np.zeros_like(times)
    valid = np.isfinite(times) & (times > 0)

    # r ~ sqrt(4 * kappa * t) for diffusion
    distances[valid] = scale_factor * np.sqrt(kappa * times[valid])
    distances[~valid] = float('inf')

    return distances


def trilaterate_2d(
    sensors_xy: np.ndarray,
    distances: np.ndarray,
    weights: Optional[np.ndarray] = None,
    bounds: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None,
) -> Tuple[float, float]:
    """
    Find position that minimizes weighted sum of squared distance errors.

    Args:
        sensors_xy: Sensor positions (n_sensors, 2)
        distances: Estimated distances to source (n_sensors,)
        weights: Optional weights for each sensor (higher = more trusted)
        bounds: Optional ((x_min, x_max), (y_min, y_max))

    Returns:
        Estimated (x, y) position
    """
    valid = np.isfinite(distances)
    if np.sum(valid) < 2:
        # Not enough valid sensors, fall back to weighted centroid
        if weights is not None:
            w = weights.copy()
            w[~valid] = 0
            if np.sum(w) > 0:
                return tuple(np.average(sensors_xy, weights=w, axis=0))
        return tuple(np.mean(sensors_xy, axis=0))

    sensors_valid = sensors_xy[valid]
    distances_valid = distances[valid]

    if weights is not None:
        weights_valid = weights[valid]
    else:
        # Weight by inverse distance (closer sensors are more reliable)
        weights_valid = 1.0 / (distances_valid + 0.1)

    weights_valid = weights_valid / np.sum(weights_valid)

    def objective(p):
        dists_to_p = np.linalg.norm(sensors_valid - p, axis=1)
        errors = (dists_to_p - distances_valid) ** 2
        return np.sum(weights_valid * errors)

    # Initial guess: weighted centroid
    x0 = np.average(sensors_valid, weights=weights_valid, axis=0)

    if bounds is not None:
        scipy_bounds = [bounds[0], bounds[1]]
    else:
        scipy_bounds = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(objective, x0, method='L-BFGS-B', bounds=scipy_bounds)

    return tuple(result.x)


def estimate_intensity_from_temperature(
    max_temps: np.ndarray,
    distances: np.ndarray,
    kappa: float,
    sigma: float = 0.05,  # Gaussian source width
    q_range: Tuple[float, float] = (0.5, 2.0),
) -> float:
    """
    Estimate source intensity from sensor temperatures and distances.

    Uses steady-state approximation for relative scaling.

    Args:
        max_temps: Maximum temperature at each sensor
        distances: Distances from source to each sensor
        kappa: Thermal diffusivity
        sigma: Gaussian source width
        q_range: Valid intensity range

    Returns:
        Estimated intensity q
    """
    valid = np.isfinite(distances) & (distances > 0.01) & (max_temps > 0.01)

    if np.sum(valid) == 0:
        return np.mean(q_range)

    # For a Gaussian source, steady-state temp ~ q / (2*pi*kappa) * something
    # Use empirical scaling based on hottest sensor
    max_temp_valid = np.max(max_temps[valid])

    # Empirical: q ~ max_temp * scaling_factor
    # This is approximate; the simulator will refine it
    q_est = max_temp_valid * 0.5  # Empirical scaling

    return np.clip(q_est, q_range[0], q_range[1])


def triangulate_single_source(
    sample: Dict,
    meta: Dict,
    q_range: Tuple[float, float] = (0.5, 2.0),
    Lx: float = 2.0,
    Ly: float = 1.0,
    time_fraction: float = 0.3,
) -> Tuple[float, float, float]:
    """
    Triangulate a single source position from sensor readings.

    Args:
        sample: Sample dictionary with Y_noisy, sensors_xy, sample_metadata
        meta: Metadata with dt
        q_range: Valid intensity range
        Lx, Ly: Domain dimensions
        time_fraction: Fraction of max temp for characteristic time

    Returns:
        (x, y, q) estimated source parameters
    """
    Y_noisy = sample['Y_noisy']
    sensors_xy = sample['sensors_xy']
    sample_meta = sample['sample_metadata']

    dt = meta['dt']
    kappa = sample_meta['kappa']
    noise_std = sample_meta.get('noise_std', 0.1)

    n_sensors = sensors_xy.shape[0]

    # Detect characteristic times for each sensor
    char_times = np.zeros(n_sensors)
    max_temps = np.zeros(n_sensors)

    for i in range(n_sensors):
        signal = Y_noisy[:, i]
        char_times[i] = detect_characteristic_time(
            signal, dt, fraction=time_fraction, noise_floor=noise_std
        )
        max_temps[i] = np.max(signal)

    # Estimate distances from characteristic times
    distances = estimate_distances_from_times(char_times, kappa)

    # Weight sensors by their max temperature (hotter = closer = more reliable)
    weights = max_temps.copy()
    weights[weights < 0] = 0
    weights = weights / (np.sum(weights) + 1e-8)

    # Trilaterate position
    bounds = ((0.05 * Lx, 0.95 * Lx), (0.05 * Ly, 0.95 * Ly))
    x, y = trilaterate_2d(sensors_xy, distances, weights, bounds)

    # Estimate intensity
    q = estimate_intensity_from_temperature(max_temps, distances, kappa, q_range=q_range)

    return (x, y, q)


def triangulate_multiple_sources(
    sample: Dict,
    meta: Dict,
    n_sources: int,
    q_range: Tuple[float, float] = (0.5, 2.0),
    Lx: float = 2.0,
    Ly: float = 1.0,
    min_separation: float = 0.3,
) -> List[Tuple[float, float, float]]:
    """
    Triangulate multiple source positions.

    Strategy:
    1. Cluster sensors by response pattern
    2. Triangulate within each cluster
    3. Fall back to hottest-sensor heuristic if clustering fails

    Args:
        sample: Sample dictionary
        meta: Metadata
        n_sources: Number of sources to find
        q_range: Valid intensity range
        Lx, Ly: Domain dimensions
        min_separation: Minimum distance between sources

    Returns:
        List of (x, y, q) tuples
    """
    if n_sources == 1:
        return [triangulate_single_source(sample, meta, q_range, Lx, Ly)]

    Y_noisy = sample['Y_noisy']
    sensors_xy = sample['sensors_xy']
    sample_meta = sample['sample_metadata']

    dt = meta['dt']
    kappa = sample_meta['kappa']
    noise_std = sample_meta.get('noise_std', 0.1)
    n_sensors = sensors_xy.shape[0]

    # For multi-source: use k-means on sensor response patterns
    # Normalize responses by their max to focus on timing patterns
    responses = Y_noisy.T  # (n_sensors, n_timesteps)
    max_per_sensor = np.max(np.abs(responses), axis=1, keepdims=True) + 1e-8
    normalized = responses / max_per_sensor

    # Simple clustering: assign sensors to sources based on when they peak
    peak_times = np.argmax(responses, axis=1)
    max_temps = np.max(responses, axis=1)

    # Sort sensors by peak time
    time_order = np.argsort(peak_times)

    # Assign sensors to clusters based on spatial proximity and timing
    clusters = [[] for _ in range(n_sources)]

    # Start with hottest sensors as cluster seeds
    hottest = np.argsort(max_temps)[::-1]
    seeds = []
    for idx in hottest:
        if len(seeds) >= n_sources:
            break
        # Check separation from existing seeds
        is_sep = True
        for s in seeds:
            if np.linalg.norm(sensors_xy[idx] - sensors_xy[s]) < min_separation:
                is_sep = False
                break
        if is_sep:
            seeds.append(idx)

    # Fill seeds if needed
    for idx in hottest:
        if len(seeds) >= n_sources:
            break
        if idx not in seeds:
            seeds.append(idx)

    # Assign sensors to nearest seed
    for i in range(n_sensors):
        min_dist = float('inf')
        best_cluster = 0
        for c, seed in enumerate(seeds):
            d = np.linalg.norm(sensors_xy[i] - sensors_xy[seed])
            if d < min_dist:
                min_dist = d
                best_cluster = c
        clusters[best_cluster].append(i)

    # Triangulate within each cluster
    sources = []
    for c in range(n_sources):
        if len(clusters[c]) == 0:
            # Empty cluster - use seed position
            seed_idx = seeds[c] if c < len(seeds) else hottest[c]
            x, y = sensors_xy[seed_idx]
            q = estimate_intensity_from_temperature(
                np.array([max_temps[seed_idx]]),
                np.array([0.1]),
                kappa,
                q_range=q_range
            )
            sources.append((x, y, q))
            continue

        cluster_sensors = sensors_xy[clusters[c]]
        cluster_signals = Y_noisy[:, clusters[c]]

        # Create mini-sample for this cluster
        mini_sample = {
            'Y_noisy': cluster_signals,
            'sensors_xy': cluster_sensors,
            'sample_metadata': sample_meta,
        }

        try:
            source = triangulate_single_source(mini_sample, meta, q_range, Lx, Ly)
            sources.append(source)
        except Exception:
            # Fallback to hottest sensor in cluster
            cluster_temps = max_temps[clusters[c]]
            hottest_in_cluster = clusters[c][np.argmax(cluster_temps)]
            x, y = sensors_xy[hottest_in_cluster]
            q = np.mean(q_range)
            sources.append((x, y, q))

    return sources


def triangulation_init(
    sample: Dict,
    meta: Dict,
    n_sources: int,
    q_range: Tuple[float, float] = (0.5, 2.0),
    Lx: float = 2.0,
    Ly: float = 1.0,
) -> np.ndarray:
    """
    Generate initial parameters using triangulation.

    This is the main interface for use with optimizers.

    Args:
        sample: Sample dictionary
        meta: Metadata
        n_sources: Number of sources
        q_range: Valid intensity range
        Lx, Ly: Domain dimensions

    Returns:
        Flat array [x1, y1, q1, x2, y2, q2, ...] of initial parameters
    """
    sources = triangulate_multiple_sources(
        sample, meta, n_sources, q_range, Lx, Ly
    )

    params = []
    for x, y, q in sources:
        params.extend([x, y, q])

    return np.array(params)


if __name__ == "__main__":
    import pickle
    from pathlib import Path

    # Test triangulation on real samples
    data_path = Path(__file__).parent.parent / "data" / "heat-signature-zero-test-data.pkl"

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    samples = data['samples']
    meta = data['meta']

    print("Testing triangulation initialization...")
    print("=" * 60)

    for i, sample in enumerate(samples[:10]):
        n_sources = sample['n_sources']
        sample_id = sample['sample_id']

        params = triangulation_init(sample, meta, n_sources)

        print(f"\n[{i+1}] {sample_id} ({n_sources} sources)")
        for j in range(n_sources):
            x, y, q = params[j*3:(j+1)*3]
            print(f"  Source {j+1}: x={x:.3f}, y={y:.3f}, q={q:.3f}")

        # Compare with hottest sensor (simple baseline)
        sensors = sample['sensors_xy']
        temps = np.mean(sample['Y_noisy'], axis=0)
        hottest_idx = np.argmax(temps)
        print(f"  Hottest sensor: {sensors[hottest_idx]}")

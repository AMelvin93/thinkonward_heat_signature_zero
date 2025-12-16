# Heat Signature Zero Solution Package
from .optimizer import HeatSourceOptimizer, estimate_all_samples
from .visualize import (
    plot_sample_overview,
    plot_estimation_comparison,
    plot_optimization_history,
    plot_source_search_space,
    plot_dataset_summary,
)

# JAX-based components (optional, requires JAX)
try:
    from .jax_simulator import JAXHeatSimulator, check_gpu
    from .jax_optimizer import JAXOptimizer, benchmark_jax_vs_numpy
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

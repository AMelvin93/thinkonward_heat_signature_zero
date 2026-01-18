# Exploration-Focused Orchestration Strategy

## Problem with Current Approach

The current orchestrator has **converged prematurely** on CMA-ES hyperparameter tuning:
- 32+ experiments all within the same algorithm family
- Workers locked into narrow parameter ranges
- No mechanism to try fundamentally different approaches
- "EXHAUSTED" conclusion reached without exploring alternatives

## Algorithmic Landscape

### Tested (with results)
| Algorithm | Best Score | Time | Verdict |
|-----------|------------|------|---------|
| CMA-ES + Multi-fidelity | 1.1362 | 69 min | Best accuracy, over budget |
| CMA-ES + Baseline sigma | 1.1247 | 57 min | In-budget best |
| Differential Evolution | 0.95 | 60 min | Lacks covariance adaptation |
| Basin Hopping | 1.06 | 1105 min | L-BFGS-B gradient overhead |
| Nelder-Mead (primary) | 1.03 | 240 min | Too many fevals |
| L-BFGS-B Hybrid | 1.15 | 352 min | Gradient overhead |
| ICA Decomposition | **1.04** | 87 min | Best-ever score, over budget |

### NOT TESTED (high potential)
| Approach | Why Promising | Effort | References |
|----------|---------------|--------|------------|
| **Surrogate-Assisted CMA-ES** | Use NN to pre-filter bad candidates | Medium | SACMA-ES papers |
| **Neural Network Surrogate** | 1000x faster predictions | High | PINNs, DeepONet |
| **Particle Swarm + Local** | Different global search than CMA-ES | Low | PSO literature |
| **Trust Region Methods** | Better than L-BFGS-B for noisy | Medium | BOBYQA, COBYLA |
| **Multi-Objective** | Optimize time AND accuracy jointly | Medium | NSGA-II, MOEA/D |
| **Ensemble Methods** | Combine multiple weak optimizers | Low | Voting, stacking |
| **Reinforcement Learning** | Learn search policy | High | RL for optimization |
| **Genetic Programming** | Evolve the optimizer itself | High | GP literature |

### PARTIALLY TESTED (revisit with modifications)
| Approach | Previous Issue | Potential Fix |
|----------|---------------|---------------|
| ICA Decomposition | 87 min (over budget) | Faster ICA, fewer iterations |
| Adjoint Gradients | 157s/sample | Use ONLY for refinement (1-2 iters) |
| JAX Differentiable | GPU overhead | Try on larger batches, or CPU JAX |
| Coarse-to-Fine | Accuracy loss | Better interpolation scheme |

## New Orchestrator Design

### Philosophy: Explore-Exploit Balance
```
EXPLORATION (60% of effort):
  - Try fundamentally different algorithm families
  - Implement approaches from recent papers
  - Cross-pollinate ideas between workers

EXPLOITATION (40% of effort):
  - Refine promising discoveries
  - Combine best elements from different approaches
  - Optimize for time budget
```

### Worker Roles (Redesigned)

**W1: SURROGATE EXPLORER**
- Focus: Fast approximation methods
- Tasks:
  1. Implement neural network surrogate for simulator
  2. Try surrogate-assisted CMA-ES (pre-filter candidates)
  3. Gaussian Process surrogate with adaptive sampling
  4. Test if surrogate can replace coarse grid

**W2: ALGORITHM HUNTER**
- Focus: Alternative optimization algorithms
- Tasks:
  1. Particle Swarm Optimization + local refinement
  2. Trust-region methods (BOBYQA, COBYLA)
  3. Pattern search methods
  4. Simulated annealing variants

**W3: HYBRID ARCHITECT**
- Focus: Combining approaches
- Tasks:
  1. Ensemble of fast weak optimizers
  2. Multi-stage pipelines (different algo per stage)
  3. Portfolio optimization (run multiple, pick best)
  4. Adaptive algorithm selection per sample

**W4: RESEARCH SCOUT** (NEW)
- Focus: Finding new ideas from literature
- Tasks:
  1. Web search for recent inverse problem papers
  2. Find implementations of promising methods
  3. Prototype ideas from papers
  4. Share findings with other workers

### Coordination Protocol (Redesigned)

```json
{
  "algorithm_families": {
    "evolutionary": {
      "tested": ["cma-es", "differential_evolution"],
      "untested": ["pso", "genetic_algorithm", "evolution_strategy"],
      "best_result": {"algo": "cma-es", "score": 1.1362, "time": 69}
    },
    "gradient_based": {
      "tested": ["l-bfgs-b", "basin_hopping"],
      "untested": ["trust_region", "levenberg_marquardt"],
      "best_result": {"algo": "l-bfgs-b", "score": 1.15, "time": 352}
    },
    "surrogate": {
      "tested": [],
      "untested": ["nn_surrogate", "gp_surrogate", "rbf_surrogate"],
      "best_result": null
    },
    "hybrid": {
      "tested": ["ica_decomposition"],
      "untested": ["ensemble", "portfolio", "meta_learning"],
      "best_result": {"algo": "ica", "score": 1.04, "time": 87}
    }
  },
  "exploration_budget": {
    "total_experiments": 100,
    "per_family_minimum": 5,
    "exploitation_threshold": 1.20
  }
}
```

### Experiment Generation Rules

1. **Diversity Requirement**: No algorithm family gets >30% of experiments until all families have ≥5 experiments

2. **Promising Signal**: If an algorithm achieves >1.15 score (even over budget), allocate 3 more experiments to optimize it

3. **Failure Learning**: When algorithm fails, document WHY and share with all workers to avoid repeating

4. **Cross-Pollination**: Every 10 experiments, workers share their best ideas for combination

### Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Algorithm families tested | 2 | 6+ |
| Unique approaches tried | 8 | 20+ |
| Best in-budget score | 1.1247 | 1.20+ |
| Research papers referenced | ~5 | 15+ |

## Implementation Plan

### Phase 1: Expand Algorithm Coverage (Days 1-2)
- [ ] Implement PSO with local refinement
- [ ] Implement BOBYQA/COBYLA trust region
- [ ] Create neural network surrogate prototype
- [ ] Test ensemble of existing optimizers

### Phase 2: Optimize Promising Directions (Days 3-4)
- [ ] Refine best new algorithm
- [ ] Hybrid combinations of top performers
- [ ] Time budget optimization

### Phase 3: Final Push (Days 5-7)
- [ ] Combine all learnings
- [ ] Final submission preparation
- [ ] Documentation for innovation score

## Key Insight

**ICA Decomposition achieved 1.04 score** - proving the accuracy headroom exists.
The challenge is getting there within 60 minutes, not finding better accuracy.

Focus areas:
1. **Faster ICA** - Can we do ICA in <30 min?
2. **Surrogate speedup** - Can NN replace simulator calls?
3. **Smarter early stopping** - Detect when we're close enough

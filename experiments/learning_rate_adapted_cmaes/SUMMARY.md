# LRA-CMA-ES (Learning Rate Adapted CMA-ES)

## Experiment ID: EXP_LRA_CMAES_001

## Status: FAILED

## Hypothesis
LRA-CMA-ES maintains constant signal-to-noise ratio in learning. May improve convergence on difficult samples where standard CMA-ES struggles.

## Results

### Run 1: cmaes library with lr_adapt=True
| Metric | Value |
|--------|-------|
| Score | 1.1440 |
| RMSE | 0.1748 |
| RMSE (1-src) | 0.1342 |
| RMSE (2-src) | 0.2018 |
| Projected (400) | 53.0 min |
| Budget Status | In budget |

### Run 2: cmaes library with lr_adapt=False (Control)
| Metric | Value |
|--------|-------|
| Score | 1.1457 |
| RMSE | 0.1642 |
| RMSE (1-src) | 0.1284 |
| RMSE (2-src) | 0.1880 |
| Projected (400) | 52.0 min |
| Budget Status | In budget |

### Comparison to Baseline
| Configuration | Score | Time | Delta Score | Delta Time |
|--------------|-------|------|-------------|------------|
| Baseline (pycma) | 1.1688 | 58.4 min | - | - |
| cmaes lr_adapt=True | 1.1440 | 53.0 min | **-0.0248** | -5.4 min |
| cmaes lr_adapt=False | 1.1457 | 52.0 min | **-0.0231** | -6.4 min |

## Key Findings

### 1. LRA Provides No Benefit
- lr_adapt=True: 1.1440
- lr_adapt=False: 1.1457
- LRA actually **HURTS** slightly (-0.0017)

The Learning Rate Adaptation feature doesn't help for this problem. The RMSE landscape is smooth enough that standard CMA-ES learning rates work well.

### 2. cmaes Library Underperforms pycma
Both cmaes library configurations are worse than the pycma baseline:
- cmaes (best): 1.1457 @ 52.0 min
- pycma baseline: 1.1688 @ 58.4 min
- Delta: -0.0231 score

The cmaes library runs faster but achieves worse accuracy. This suggests different default parameters or convergence behavior.

### 3. Speed vs Accuracy Trade-off
| Library | Time | Accuracy |
|---------|------|----------|
| pycma | 58.4 min | Better |
| cmaes | 52.0 min | Worse |

The cmaes library converges faster (fewer generations?) but doesn't reach the same quality solutions.

## Root Cause Analysis

The cmaes library and pycma have different implementations:
1. **Different defaults**: cmaes may use different population size, learning rates, or stopping criteria
2. **Different convergence**: cmaes seems to converge earlier with worse solutions
3. **API differences**: cmaes uses batch tell() vs pycma's individual tell()

The LRA feature specifically is not the issue - the base cmaes library is simply less effective for this problem than pycma.

## Recommendation

**STAY with pycma (cma library).** The cmaes library's implementation does not match pycma's performance for this problem. The baseline using pycma remains optimal.

Do not pursue further cmaes library variants (LRA, SepCMA from cmaes, etc.) as the underlying library is less suitable.

## MLflow Runs
- Run 1 (lr_adapt=True): `30d0fa9276c74c5db638e0a0176980ad`
- Run 2 (lr_adapt=False): `42f306a6803f47d4b5427b50a186446c`

## Conclusion

LRA-CMA-ES fails to improve on the baseline. The cmaes library underperforms pycma regardless of the lr_adapt setting. The pycma library with standard CMA-ES remains the best choice for this problem.

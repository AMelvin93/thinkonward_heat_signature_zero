# Experiment Summary: higher_population_density

## Metadata
- **Experiment ID**: EXP_HIGHER_POPULATION_DENSITY_001
- **Worker**: W1
- **Date**: 2026-01-26
- **Algorithm Family**: cmaes_tuning_v3

## Status: FAILED

## Objective
Test whether increasing CMA-ES population size (2x default) improves candidate diversity.

## Results Summary
- **Score**: 1.1115 @ 33.3 min
- **Baseline**: 1.1246 @ 32.6 min
- **Delta**: -0.0131 (-1.0%)
- **Status**: FAILED

## RMSE Breakdown
- 1-source: 0.1721 (vs baseline ~0.15)
- 2-source: 0.2479 (vs baseline ~0.19)

## Why It Failed

With a fixed fevals budget (20 for 1-src, 36 for 2-src):
- 2x population = 2x samples per generation
- Same total budget = fewer generations
- Fewer generations = worse convergence

CMA-ES needs sufficient generations to adapt its covariance matrix. Doubling population without increasing budget just halves the number of generations, degrading convergence quality.

## Recommendation

DO NOT increase population size without proportionally increasing fevals budget. The default population is already well-tuned for our fevals budget.

Mark cmaes_tuning_v3 (population tuning) as EXHAUSTED.

## Raw Data
- MLflow run ID: 4a59f78232f148e98c00e06404c9d17d

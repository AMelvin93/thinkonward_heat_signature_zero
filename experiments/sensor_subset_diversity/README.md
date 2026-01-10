# Sensor Subset Diversity Experiment

## Research Basis

From recent 2025 SIAM paper on "Sparse Boundary Measurements":
- **Two boundary points can uniquely determine a heat source**
- Our samples have 8-12 sensors (4-6x more than needed!)

## Key Insight

If 2 sensors are theoretically sufficient, then:
- Different sensor subsets should give different but valid solutions
- This creates NATURAL diversity without artificial perturbation
- Each subset optimizes on slightly different information

## Implementation

1. Divide sensors into 2-3 overlapping subsets
2. Run CMA-ES on each subset independently
3. Each gives a candidate with different "view" of the problem
4. Diversity comes naturally from different sensor information

## Expected Benefits

- Natural diversity (not forced perturbation)
- Each candidate uses different subset → different solution
- All candidates should be high quality (each uses sufficient info)
- Captures diversity bonus (0.3 points) effectively

## References

- [Sparse Boundary Measurements (2025)](https://arxiv.org/html/2502.03018v1)
- [Two-Point Measurement Uniqueness (SIAM)](https://epubs.siam.org/doi/abs/10.1137/25M1725620)

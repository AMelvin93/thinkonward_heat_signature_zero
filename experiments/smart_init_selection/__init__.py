"""
Smart Init Selection Optimizer for Heat Source Identification.

Key Innovation: Instead of splitting fevals across multiple initializations
(which wastes compute on inits that don't win), we:

1. Quickly evaluate ALL inits with 1 simulation each
2. Pick the best init (lowest RMSE)
3. Give ALL remaining fevals to that winning init

Problem Identified:
    With 3 inits (triangulation, smart, transfer) and 24 fevals:
    - Old approach: Each init gets 8 fevals (3 separate CMA-ES runs)
    - But one init usually wins (smart 55%, tri 35%, transfer 10%)
    - We waste ~65% of compute on losing inits!

New Approach:
    - Init selection: 3 fevals (one per init)
    - Focused optimization: 21 fevals on best init
    - Single CMA-ES run with full budget

Expected Benefits:
    - ~40% time reduction on 2-source samples
    - Better convergence (more fevals per CMA-ES run)
    - Could enable 12/24 config within 60-min budget
"""

from .optimizer import SmartInitOptimizer

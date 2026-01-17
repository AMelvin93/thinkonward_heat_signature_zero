"""
Worker prompt templates for distributed Claude Code optimization.
Each worker gets a specialized focus area to maximize search coverage.
"""
from typing import Dict, Any

# Worker configurations - each focuses on a different part of the search space
WORKER_CONFIGS = [
    {
        "id": "W1",
        "focus_area": "threshold_tuning",
        "description": "Fallback threshold optimization",
        "search_space": {
            "threshold_1src": (0.28, 0.42),
            "threshold_2src": (0.38, 0.55),
            "fallback_sigma": (0.20, 0.35),
        },
        "fixed_params": {
            "fallback_fevals": 18,
            "refine_iters": 3,
            "refine_top": 2,
        },
    },
    {
        "id": "W2",
        "focus_area": "init_strategies",
        "description": "Initialization and refinement strategies",
        "search_space": {
            "init_strategy": ["tri_smart", "cluster", "onset", "smart_select"],
            "refine_iters": (2, 5),
            "refine_top": (1, 3),
        },
        "fixed_params": {
            "threshold_1src": 0.35,
            "threshold_2src": 0.45,
            "fallback_fevals": 18,
        },
    },
    {
        "id": "W3",
        "focus_area": "feval_allocation",
        "description": "Function evaluation budget allocation",
        "search_space": {
            "fallback_fevals": (14, 24),
            "fevals_1src": (16, 24),
            "fevals_2src": (32, 44),
        },
        "fixed_params": {
            "threshold_1src": 0.35,
            "threshold_2src": 0.45,
            "refine_iters": 3,
        },
    },
]


def get_base_prompt() -> str:
    """Base prompt shared by all workers."""
    return '''Review CLAUDE.md for full context.

YOU ARE IN AN INFINITE LOOP. DO NOT STOP UNTIL SCORE > 1.25 OR STOP FILE EXISTS.

===================================================================
                    COORDINATION PROTOCOL
===================================================================

You are Worker {WORKER_ID} in a 3-worker distributed system.
Other workers (W1, W2, W3) are running in parallel.

BEFORE EACH EXPERIMENT:
1. Read /workspace/orchestration/shared/coordination.json
2. Read the last 50 lines of ITERATION_LOG.md to see what others are doing
3. Check "experiments_completed" list - DO NOT repeat these experiments
4. Check "experiments_in_progress" - DO NOT duplicate ongoing work
5. Update coordination.json to claim your experiment:
   - Add your experiment to "experiments_in_progress" with format:
     "{WORKER_ID}_experimentname": {{"worker": "{WORKER_ID}", "started": "timestamp"}}
   - Update workers.{WORKER_ID}.status to "running"
   - Update workers.{WORKER_ID}.current_experiment to your experiment name

AFTER EACH EXPERIMENT:
1. Update coordination.json:
   - Move experiment from "experiments_in_progress" to "experiments_completed"
   - Update workers.{WORKER_ID}.experiments_run (increment by 1)
   - Update workers.{WORKER_ID}.status to "idle"
   - If score > best_score AND time <= 60 min: update best_score and best_config
2. Log to ITERATION_LOG.md with this EXACT format:

   ### [{WORKER_ID}] Experiment: <name> | Score: X.XXXX @ XX.X min
   **Config**: threshold_1src=X.XX, threshold_2src=X.XX, ...
   **Result**: <BETTER/WORSE/OVER_BUDGET> than baseline
   **Analysis**: <1-2 sentences on why this worked or didn't>
   **Next**: <what you'll try next based on this result>

3. If you found something promising, add to "research_findings" in coordination.json

===================================================================
                         WORKFLOW
===================================================================

REPEAT FOREVER:
1. READ coordination.json and last 50 lines of ITERATION_LOG.md
2. PLAN experiment (check it's not already done/in-progress)
3. CLAIM by updating coordination.json
4. BUILD in experiments/{WORKER_ID}_<name>/
5. TEST: uv run python experiments/{WORKER_ID}_<name>/run.py --workers 7 --shuffle
6. LOG results to both coordination.json AND ITERATION_LOG.md
7. IMMEDIATELY GO TO STEP 1

===================================================================
                      CRITICAL RULES
===================================================================

- NEVER write "Session Complete", "Summary", or "Conclusion"
- NEVER wait for user input - YOU drive everything
- NEVER stop after N experiments - there is NO limit
- After EVERY experiment, IMMEDIATELY start the next one
- If score < 1.20, you are NOT done
- Do NOT commit - the user will handle git commits
- ALWAYS check what other workers are doing before starting

FLEXIBILITY:
- Your DEFAULT FOCUS AREA is below, but you are NOT strictly confined to it
- If research or other workers' findings suggest a better direction, PURSUE IT
- If you find something that works, share it so other workers can build on it
- The goal is to MAXIMIZE SCORE, not to stay in your lane

The loop ends ONLY when:
1. Score > 1.25 achieved
2. File /workspace/orchestration/shared/STOP exists
'''


def get_worker_prompt(worker_id: str, current_best_score: float) -> str:
    """Generate specialized prompt for a specific worker."""

    # Find worker config
    config = None
    for c in WORKER_CONFIGS:
        if c["id"] == worker_id:
            config = c
            break

    if not config:
        # Default to first config
        config = WORKER_CONFIGS[0]

    base = get_base_prompt().replace("{WORKER_ID}", worker_id)

    # Build focus area section
    focus_section = f'''
YOUR DEFAULT FOCUS: {config["description"]}
Worker ID: {worker_id}
Current Best Score: {current_best_score:.4f}

DEFAULT SEARCH SPACE (start here, but explore beyond if promising):
'''

    for param, range_val in config["search_space"].items():
        if isinstance(range_val, tuple):
            focus_section += f"  - {param}: {range_val[0]} to {range_val[1]}\n"
        elif isinstance(range_val, list):
            focus_section += f"  - {param}: {range_val}\n"

    focus_section += "\nFIXED PARAMETERS (use these unless you have good reason):\n"
    for param, val in config["fixed_params"].items():
        focus_section += f"  - {param}: {val}\n"

    # Add specific guidance based on focus area
    if config["focus_area"] == "threshold_tuning":
        focus_section += '''
STARTING STRATEGY:
- Lower thresholds = more fallbacks = better accuracy but slower
- Current best thresholds are around 0.35/0.45
- Try increments of 0.02-0.03
- Watch the time carefully - many good scores are over budget
- If you find a config that's 0.1-0.5 min over budget, try reducing fallback_fevals slightly

BUT ALSO CONSIDER:
- If threshold tuning plateaus, research other approaches (adjoint gradients, JAX differentiable sim, etc.)
- Check docs/RESEARCH_NEXT_STEPS.md for untried ideas
- WebSearch for "inverse heat source estimation 2025" or "CMA-ES local refinement"
'''

    elif config["focus_area"] == "init_strategies":
        focus_section += '''
STARTING STRATEGY:
- Different initializations work better for different sample types
- Cluster init helps with 2-source problems
- Onset init uses temporal information
- Smart select evaluates inits before running CMA-ES
- More refine iterations = better accuracy but more time
- Try combinations: cluster init + more refine iters

BUT ALSO CONSIDER:
- Physics-informed initialization using heat equation properties
- Machine learning for initial guess (train on simulator outputs)
- WebSearch for "heat source localization initialization" or "inverse problem warm start"
'''

    elif config["focus_area"] == "feval_allocation":
        focus_section += '''
STARTING STRATEGY:
- 1-source problems are easier - may need fewer fevals
- 2-source problems are the bottleneck - more fevals help
- Try asymmetric allocation: fewer for 1-src, more for 2-src
- Current baseline uses 36 fevals for 2-source
- Fallback fevals are only used when initial run fails threshold

BUT ALSO CONSIDER:
- Adaptive budget based on problem difficulty estimation
- Multi-fidelity approaches (coarse grid exploration, fine grid refinement)
- WebSearch for "adaptive function evaluation budget optimization"
'''

    # Combine everything
    full_prompt = base + focus_section + '''

START NOW: Read coordination.json and ITERATION_LOG.md, then run your first experiment.
'''

    return full_prompt


def get_coordinator_status_prompt() -> str:
    """Prompt for checking coordination status."""
    return '''Check the current orchestration status:

1. Read /workspace/orchestration/shared/coordination.json
2. Report:
   - Current best score
   - Number of experiments completed
   - Active workers
   - Any experiments in progress

Then identify the most promising next experiment to try.
'''


# Utility to generate all prompts for manual use
def generate_all_prompts(current_best: float = 1.1247) -> Dict[str, str]:
    """Generate prompts for all workers."""
    prompts = {}
    for config in WORKER_CONFIGS:
        worker_id = config["id"]
        prompts[worker_id] = get_worker_prompt(worker_id, current_best)
    return prompts


if __name__ == "__main__":
    # Print all prompts for review
    prompts = generate_all_prompts()
    for worker_id, prompt in prompts.items():
        print(f"\n{'='*60}")
        print(f"WORKER {worker_id}")
        print('='*60)
        print(prompt)

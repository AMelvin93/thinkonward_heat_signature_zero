# Heat Signature Zero - Autonomous Experiment Loop

## The Loop (Execute Repeatedly)

```
┌─────────────────────────────────────────────────────────────────┐
│  1. READ: docs/RESEARCH_NEXT_STEPS.md                           │
│     → Get current best score, priority queue, past learnings    │
│                                                                 │
│  2. SELECT: Pick next experiment from priority queue            │
│     → Consider potential gain vs effort                         │
│                                                                 │
│  3. IMPLEMENT: Create/modify experiment code                    │
│     → experiments/<approach_name>/optimizer.py, run.py          │
│                                                                 │
│  4. RUN: Execute experiment (--workers 7 --shuffle)             │
│     → Target: score > 1.0116 AND time < 60 min                  │
│     → Try parameter variations to optimize                      │
│                                                                 │
│  5. LOG: Results automatically pushed to MLflow                 │
│     → submission_score, rmse, projected_400_samples_min         │
│                                                                 │
│  6. RESEARCH: Think deeply about results                        │
│     → Root cause analysis: WHY did it succeed/fail?             │
│     → Generate new hypotheses based on learnings                │
│     → Identify patterns across all experiments                  │
│                                                                 │
│  7. UPDATE: Modify docs/RESEARCH_NEXT_STEPS.md                  │
│     → Add result to experiment history table                    │
│     → Document key learnings                                    │
│     → Add new approach ideas with hypotheses                    │
│     → Re-prioritize queue based on insights                     │
│                                                                 │
│  8. COMMIT: Push to git                                         │
│     → git add && git commit -m "[SCORE: X.XX] approach: result" │
│                                                                 │
│  9. REPEAT: Go back to step 1                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Objective
Maximize the competition score for heat source identification. Current best: **1.0116 @ 58.6 min** (SmartInitOptimizer).

**Theoretical maximum score: 1.3** (we're at 77.8%)

## Convergence Optimization Strategies

### 1. Fast Validation First
Before running full 80-sample tests, do quick validation:
```bash
# Quick 10-sample test (2 min) to check if approach is viable
uv run python experiments/<name>/run.py --workers 7 --shuffle --max-samples 10
```
Only run full 80-sample test if quick test shows promise.

**Note:** If `--max-samples` isn't supported, add it to the run.py:
```python
parser.add_argument('--max-samples', type=int, default=None,
                    help='Limit samples for quick validation')
# Then after loading samples and shuffling:
if args.max_samples and args.max_samples < len(samples):
    samples = samples[:args.max_samples]
```

### 2. Parallel Variations (Quick Exploration vs Accurate Timing)

**For quick exploration (score comparison, not timing):**
Use `--max-samples 10` to test variations quickly in sequence:
```bash
# Quick tests to find best config (~2 min each)
uv run python experiments/<name>/run.py --max-fevals-2src 25 --workers 7 --shuffle --max-samples 10
uv run python experiments/<name>/run.py --max-fevals-2src 30 --workers 7 --shuffle --max-samples 10
uv run python experiments/<name>/run.py --subsample-factor 4 --workers 7 --shuffle --max-samples 10
```

**For accurate timing (final validation):**
Run ONE full experiment at a time with 7 workers:
```bash
# Full 80-sample run - accurate G4dn timing
uv run python experiments/<name>/run.py --max-fevals-2src 28 --workers 7 --shuffle
```

**WARNING:** Do NOT run multiple 7-worker experiments in parallel inside the container.
The container is limited to 8 CPUs (G4dn simulation). Parallel runs will compete
for resources and give inflated, inaccurate timing.

### 3. Track Component Contributions
For each experiment, isolate what helped:
- Did accuracy improve? → Which component? (init? fevals? polish?)
- Did speed improve? → Which phase got faster?
- Use ablation: disable one component at a time to measure its contribution

### 4. Exploit vs Explore Balance
- **70% Exploit**: Variations on approaches scoring > 1.0
- **30% Explore**: New untested approaches
- If stuck (3 iterations no improvement) → increase explore to 50%

### 5. Combine Winning Components
After 3-5 experiments, look for combinable wins:
- "Smart init improved by X%" + "Timestep subsampling saved Y min"
- → Try combining them in one optimizer

### 6. Focus on the Bottleneck
Current bottleneck: **2-source RMSE (0.316)**
- Prioritize approaches that specifically target 2-source accuracy
- 1-source is nearly solved (RMSE 0.190)

## Critical Constraints
- **Time Budget**: 400 samples must complete in <60 minutes on G4dn.2xlarge
- **For 80-sample tests**: Target <11 min (projects to ~55 min for 400)
- **Must use simulator at inference** - no static ML or grid search
- **Workers**: Always use `--workers 7` for timing accuracy

## Key Files
- `docs/RESEARCH_NEXT_STEPS.md` - **Source of truth** for all experiments, results, and next steps
- `CLAUDE.md` - Project rules and constraints
- `experiments/` - All experimental code
- `src/` - Production-ready optimizers (only after validation)

---

## Autonomous Workflow Loop

### Phase 1: Research & Select Next Experiment

**FIRST: Safety checks before each iteration:**
```bash
# 1. Check for emergency stop
if [ -f /workspace/STOP ]; then exit; fi

# 2. Check iteration count in ITERATION_LOG.md
# If >= 10 iterations, STOP and write final summary

# 3. Record iteration start
echo "## Iteration N - $(date)" >> ITERATION_LOG.md
```

**THEN: Read current state:**
```bash
cat docs/RESEARCH_NEXT_STEPS.md
cat PROMPT.md  # Check for any updates to the workflow
```

**Check for user directives:** Look for sections like:
- `## URGENT NOTE` - User wants to redirect focus
- `## PAUSE` - User wants to stop the loop
- `## PRIORITY OVERRIDE` - User changed the priority queue

**Decision criteria for next experiment:**
1. Check "Remaining Untested Approaches" section for priority order
2. Consider potential score gain vs implementation effort
3. Prefer approaches that address the **2-source RMSE bottleneck** (currently 0.316)

**Current priority queue (from RESEARCH_NEXT_STEPS.md):**
| Priority | Approach | Status |
|----------|----------|--------|
| 5 | Timestep Subsampling | Ready to test |
| 6 | Early CMA-ES Termination | Untested |
| 7 | Bayesian Optimization | Untested |

### Phase 2: Implement Experiment
```bash
# Create experiment folder if new approach
mkdir -p experiments/<approach_name>

# Required files:
# - optimizer.py: The optimizer implementation
# - run.py: Run script with MLflow logging
# - __init__.py: Module exports
```

**Implementation checklist:**
- [ ] Uses simulator at inference (required)
- [ ] Supports `--workers 7` for G4dn simulation
- [ ] Logs to MLflow with key metrics
- [ ] Supports `--shuffle` for balanced history
- [ ] Prints projected 400-sample time

### Phase 3: Run Experiment
```bash
# Standard test command (80 samples, 7 workers)
uv run python experiments/<approach_name>/run.py --workers 7 --shuffle

# Alternative configs to try if base works:
# --max-fevals-2src 25  # More 2-source fevals
# --subsample-factor 4   # For timestep subsampling
```

**Expected output format:**
```
RESULTS
======================================================================
Config:           <config description>
Best RMSE:        X.XXXXXX +/- X.XXXXXX
Avg Candidates:   X.X
Submission Score: X.XXXX
Projected (400):  XX.X min
```

### Phase 4: Analyze Results
**Success criteria:**
- Score > 1.0116 (current best) AND projected time < 60 min → **SUCCESS**
- Score > 1.0 AND projected time < 55 min → **VIABLE** (safe margin)
- Score improved but time > 60 min → **OVER BUDGET** (need optimization)
- Score not improved → **FAILED** (document learnings)

**Key metrics to track:**
- `submission_score` - Primary metric
- `rmse_2src` - Main bottleneck (currently 0.316)
- `projected_400_samples_min` - Must be < 60

### Phase 5: Commit & Document
```bash
# Commit experiment results to git
git add experiments/<approach_name>/ docs/RESEARCH_NEXT_STEPS.md
git commit -m "[SCORE: X.XXXX @ XX.X min] <approach>: <brief result>"

# Commit message examples:
# [SCORE: 1.0234 @ 54.2 min] timestep_subsampling: 2x factor beats baseline
# [SCORE: 0.9812 @ 48.1 min] early_termination: faster but accuracy loss
# [FAIL: 72.3 min] bayesian_opt: over budget, needs optimization

# Push after significant milestones (new best, or every 3-5 experiments)
# git push origin main

# Update RESEARCH_NEXT_STEPS.md with:
# 1. Add result to "Complete Experiment History" table
# 2. Mark approach as tested in priority queue
# 3. Add "Key Learning" from experiment
# 4. Update "Current Best" if improved
# 5. Add any new approach ideas discovered
```

**Documentation template for new result:**
```markdown
### <Approach Name> (Priority X) - TESTED ✅/❌

**Result**: Score X.XXXX @ XX.X min

**Test Results**:
| Config | Score | RMSE | 2-src RMSE | Time | Status |
|--------|-------|------|------------|------|--------|
| <config> | X.XXXX | X.XXX | X.XXX | XX.X min | ✅/❌ |

**Key Learning**: <What we learned from this experiment>

**Recommendation**: <Keep/abandon this approach>
```

### Phase 6: Research & Adapt Based on Results

**IMPORTANT: Use extended thinking for this phase.**
Before analyzing, say: "Let me think deeply about these results..."

**After EVERY experiment, perform this analysis:**

1. **Root Cause Analysis** - Why did it succeed/fail?
   - If over budget: What dominated runtime? (timesteps? fevals? polish?)
   - If accuracy loss: Which metric suffered? (1-src vs 2-src RMSE?)
   - If succeeded: What was the key insight that made it work?

2. **Generate New Hypotheses** - Think harder about what the results imply:
   - "Timestep subsampling worked because..." → Could we push further?
   - "Early termination failed because..." → What alternative addresses this?
   - Look for patterns across experiments (e.g., "all fast methods lose 2-src accuracy")
   - Consider: "What would a domain expert notice that I might be missing?"

3. **Update RESEARCH_NEXT_STEPS.md with NEW IDEAS:**
   ```markdown
   ### NEW: <Approach Name> (Priority X) - Untested

   **Hypothesis**: Based on <previous experiment>, we believe...

   **Why This Could Work**: <reasoning from results>

   **Implementation Plan**: <concrete steps>
   ```

4. **Re-prioritize the Queue** - Based on learnings:
   - If timestep subsampling shows promise → prioritize variations
   - If a whole category fails (e.g., "speed optimizations hurt accuracy") → deprioritize similar approaches
   - Add new approaches discovered during analysis

**Decision tree for next experiment:**
```
Results Analysis
├── Score improved AND time < 60 min?
│   ├── YES → Try parameter variations to optimize further
│   │         (e.g., --subsample-factor 4, --max-fevals-2src 30)
│   └── NO → Analyze WHY, then:
│       ├── Time issue? → Look for speedup opportunities
│       ├── Accuracy issue? → What component caused loss?
│       └── Generate new hypothesis → Add to priority queue
│
└── After optimization or 2-3 failed variations:
    └── Move to next priority in RESEARCH_NEXT_STEPS.md
```

**If experiment succeeded:**
1. Run 2-3 more times to verify consistency
2. Try parameter variations to optimize further
3. If consistently better → promote to `src/`
4. Update `src/OPTIMIZER_HISTORY.md`

**If experiment failed:**
1. Document WHY it failed (not just that it failed)
2. Extract any useful sub-components or insights
3. Generate new approach ideas based on the failure
4. Update priority queue and move to next

**If hitting diminishing returns:**
1. Document final best result for this approach
2. **Think deeply**: "What would need to change to break through?"
3. Research external resources (papers, similar problems)
4. **Think step by step** to generate 2-3 new approach ideas
5. Add to RESEARCH_NEXT_STEPS.md with clear hypotheses
6. Continue to next priority

---

## Quick Reference

### Current Best Model
```bash
uv run python experiments/smart_init_selection/run.py \
    --max-fevals-1src 12 --max-fevals-2src 22 --workers 7 --shuffle
```
- Score: 1.0116 @ 58.6 min

### Key Technical Insights
1. **Heat equation is LINEAR in q**: `T(x,t) = q × T_unit(x,t)`
2. **Heat2D time ∝ nt (timesteps)**, not grid size
3. **2-source RMSE (0.316) is 66% worse than 1-source (0.190)** - main bottleneck
4. **Smart Init Selection** eliminates wasted compute on losing initializations

### Scoring Formula
```
P = (1/N_valid) * Σ(1/(1 + RMSE)) + 0.3 * (N_valid/3)
```
- Accuracy term: max 1.0 (when RMSE = 0)
- Diversity term: max 0.3 (when 3 valid candidates)
- **Total max: 1.3**

### What Works
- Smart Init Selection (+1.4%)
- Analytical Intensity (+14.8%)
- Enhanced Features (+3.3%)
- Transfer Learning (+12.1%)
- Shuffle (balanced history)

### What Doesn't Work
- Coarse-to-Fine Grid (grid size isn't bottleneck)
- ICA Decomposition (too slow within budget)
- Adaptive k in Transfer (dilutes fevals)

---

## Stop Conditions & Token Protection

### Hard Limits (MUST OBEY)
```
MAX_ITERATIONS = 10          # Stop after 10 experiment iterations
MAX_VARIATIONS_PER_APPROACH = 3   # Max variations before moving on
MAX_CONSECUTIVE_FAILURES = 3      # Move on after 3 failures
MAX_SESSION_HOURS = 8            # Stop after 8 hours
```

### Iteration Tracking
At the START of each iteration, update `ITERATION_LOG.md`:
```markdown
## Iteration X - [timestamp]
- Approach: <name>
- Status: RUNNING/COMPLETED/FAILED
- Score: X.XXXX (if completed)
- Time: XX min (if completed)
```

**Check iteration count** - If >= MAX_ITERATIONS, STOP and summarize.

### Emergency Stop File
**Before EVERY iteration**, check for stop signal:
```bash
if [ -f /workspace/STOP ]; then
    echo "STOP file detected - halting loop"
    # Document current state and exit
fi
```

**To stop the loop manually**, create the file:
```bash
touch /workspace/STOP
```

### Token Conservation Rules
1. **Don't re-read entire files** - Only read sections you need
2. **Keep responses concise** - No verbose explanations during loop
3. **Skip analysis if clear failure** - Over budget by 20+ min? Just log and move on
4. **Limit deep thinking** - Only use extended thinking for ambiguous results
5. **Batch commits** - Commit every 2-3 experiments, not every one

### Stop the loop when:
1. **Score > 1.2** achieved within budget (excellent result)
2. **MAX_ITERATIONS reached** (10 experiments)
3. **3+ consecutive experiments** show no improvement
4. **All priority approaches** have been tested
5. **STOP file exists** in /workspace/
6. **8 hours elapsed** since session start

### When stopping, ensure:
- [ ] RESEARCH_NEXT_STEPS.md is fully updated
- [ ] ITERATION_LOG.md has complete history
- [ ] Best model is documented with exact command
- [ ] All learnings are captured
- [ ] Final summary written

---

## Example Iteration

```
Iteration 1: Timestep Subsampling
├── Read RESEARCH_NEXT_STEPS.md → Priority 5 is next
├── Check experiments/timestep_subsampling/ exists
├── Run: uv run python experiments/timestep_subsampling/run.py --workers 7 --shuffle
├── Result: Score 1.02, Time 52 min → SUCCESS (if better than 1.0116)
│   ├── Update docs with result
│   ├── Try variations: --subsample-factor 4, --max-fevals-2src 30
│   └── If consistent → promote to src/
└── Or: Score 0.98, Time 45 min → FAILED (document learning, move to Priority 6)

Iteration 2: Early CMA-ES Termination (if needed)
├── Read updated RESEARCH_NEXT_STEPS.md
├── Create experiments/early_termination/
├── Implement early stopping logic
├── Run and analyze...
└── Continue loop...
```

---

## Overnight Robustness

### Error Handling
If an experiment crashes or errors:
1. **Log the error** - Copy the full traceback
2. **Document in RESEARCH_NEXT_STEPS.md** under the approach:
   ```markdown
   **Error**: <error type>
   **Cause**: <likely cause>
   **Fix attempted**: <what you tried>
   ```
3. **Try ONE fix** - If obvious (import error, typo), fix and retry
4. **If still failing** - Mark as BLOCKED, move to next priority
5. **Never retry same error more than twice**

### Git Error Handling
```bash
# If "nothing to commit":
# → This is OK, just continue to next iteration

# If merge conflict or other git error:
git status  # Check what's wrong
git stash   # Save work if needed
# Document the issue and continue
```

### Context Management
After **every 3-5 iterations**, summarize progress:
```markdown
## SESSION SUMMARY (Iteration X)
- Experiments run: [list]
- Best result so far: Score X.XXXX @ XX min
- Key learnings: [bullets]
- Next priorities: [list]
```
This helps maintain context if the session gets long.

### Stuck Detection
**You are stuck if:**
- 3 consecutive experiments show no improvement
- Same error occurs twice
- No new ideas can be generated

**When stuck:**
1. **Think deeply**: "What fundamental assumption might be wrong?"
2. Review ALL past experiments in RESEARCH_NEXT_STEPS.md for patterns
3. Consider radically different approaches (not just parameter tweaks)
4. If truly stuck after 30 min of analysis → document state and PAUSE

### Resource Checks
Periodically verify:
```bash
# Check disk space (MLflow runs can grow)
df -h /workspace

# Check if MLflow tracking is working
ls -la mlruns/
```

### Recovery
If session crashes or needs to resume:
1. Read `docs/RESEARCH_NEXT_STEPS.md` - it has full state
2. Check `git log --oneline -10` for recent progress
3. Check `mlruns/` for recent experiment results
4. Continue from last documented state

### Guardrails
**DO NOT:**
- Run more than 3 variations of a failed approach
- Spend more than 2 hours on a single approach without results
- Make changes to `src/` without a validated improvement
- Delete or overwrite existing experiment code that worked
- Ignore the 60-min time budget constraint

**ALWAYS:**
- Commit after each experiment (even failures)
- Update RESEARCH_NEXT_STEPS.md with every result
- Use `--workers 7` for timing accuracy
- Check for user directives at the start of each iteration

---

*This prompt guides autonomous experimentation for Heat Signature Zero.*
*Primary goal: Beat 1.0116 score while staying under 60-min budget.*
*Source of truth: docs/RESEARCH_NEXT_STEPS.md*

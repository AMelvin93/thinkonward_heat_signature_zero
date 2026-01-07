# Heat Signature Zero - Autonomous Experiment Loop

## Objective
Maximize the competition score for heat source identification. Current best: **1.0116 @ 58.6 min** (SmartInitOptimizer).

**Theoretical maximum score: 1.3** (we're at 77.8%)

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
```bash
# Read current state
cat docs/RESEARCH_NEXT_STEPS.md
```

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

### Phase 6: Iterate or Promote
**If experiment succeeded:**
1. Run 2-3 more times to verify consistency
2. Try parameter variations to optimize further
3. If consistently better → promote to `src/`
4. Update `src/OPTIMIZER_HISTORY.md`

**If experiment failed:**
1. Document why it failed in RESEARCH_NEXT_STEPS.md
2. Identify if any sub-component is salvageable
3. Move to next priority in queue

**If minimal returns on current approach:**
1. Document final best result
2. Generate new approach ideas based on learnings
3. Update priority queue with new approaches
4. Start next iteration

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

## Stop Conditions
Stop the loop when:
1. **Score > 1.2** achieved within budget (excellent result)
2. **3+ consecutive experiments** show no improvement
3. **All priority approaches** have been tested
4. **Time constraint**: Competition deadline approaching

When stopping, ensure:
- [ ] RESEARCH_NEXT_STEPS.md is fully updated
- [ ] Best model is documented with exact command
- [ ] All learnings are captured
- [ ] src/OPTIMIZER_HISTORY.md is updated if new best

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

*This prompt guides autonomous experimentation for Heat Signature Zero.*
*Primary goal: Beat 1.0116 score while staying under 60-min budget.*
*Source of truth: docs/RESEARCH_NEXT_STEPS.md*

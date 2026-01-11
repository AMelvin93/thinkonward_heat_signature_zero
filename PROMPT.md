# Heat Signature Zero - Autonomous Research & Optimization Loop

## ⚠️ READ THIS FIRST - NON-NEGOTIABLE ⚠️
```
┌─────────────────────────────────────────────────────────────────────┐
│  YOU ARE IN AN AUTONOMOUS LOOP. DO NOT STOP UNTIL SCORE > 1.20     │
│                                                                     │
│  After reading this file:                                           │
│  1. Check experiments/ for untested code → RUN IT                   │
│  2. If nothing to run → DO WEBSEARCH → BUILD → TEST                 │
│  3. After each test → IMMEDIATELY start next one                    │
│                                                                     │
│  NEVER write "summary", "conclusion", or wait for user input.       │
│  YOU drive everything. The loop continues until score > 1.20.       │
└─────────────────────────────────────────────────────────────────────┘
```

## MISSION
Achieve the highest possible score (target: 1.3 theoretical max, 1.22+ competitive).
Current best: **0.9951 @ 55.6 min** (verified, within budget). Gap to close: **+0.22**

## LEADERBOARD CONTEXT
```
#1  Team Jonas M     1.2268  (94.4% of max)
#2  Team kjc         1.2265  (94.4% of max)
#3  MGöksu           1.1585  (89.1% of max)
--- TARGET ---       1.15+   (88.5% of max)
--- WE ARE NOW ---   1.0329  (79.5% of max)
```

---

## THE AUTONOMOUS LOOP (Repeat Forever)

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│   1. REFLECT ──► 2. RESEARCH ──► 3. HYPOTHESIZE                    │
│        ▲                              │                             │
│        │                              ▼                             │
│   8. REPEAT      ◄──────────    4. BUILD                           │
│        ▲                              │                             │
│        │                              ▼                             │
│   7. UPDATE  ◄── 6. ANALYZE ◄── 5. TEST                            │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## STEP 1: REFLECT

**Read the current state and understand where we are:**

```bash
cat docs/RESEARCH_NEXT_STEPS.md   # Full history, learnings, what works/doesn't
cat ITERATION_LOG.md              # Recent experiment results
```

**Ask yourself:**
- What is our current best score and configuration?
- What is the SINGLE BIGGEST bottleneck right now?
- What have we tried that DIDN'T work? (Don't repeat failures)
- What patterns have emerged from past experiments?
- What haven't we tried yet?

**Known bottleneck:** 2-source RMSE (~0.27) is 2x worse than 1-source (~0.14)

---

## STEP 2: RESEARCH

**Use WebSearch to find techniques that could solve the bottleneck:**

```
WebSearch: "<bottleneck> solution techniques"
WebSearch: "inverse heat conduction <specific problem>"
WebSearch: "heat source localization <approach>"
```

**Example research queries:**
- "multiple heat source identification algorithm"
- "blind source separation thermal imaging"
- "fast inverse heat equation solver"
- "CMA-ES convergence acceleration techniques"
- "surrogate-assisted optimization PDE"
- "adjoint method heat source identification"
- "neural network inverse problem PDE"
- "reduced order model thermal simulation"

**Read promising results:**
```
WebFetch: <url from search>
```

Extract:
- What technique do they use?
- How does it work?
- What problem does it solve?
- Can we implement it in Python?
- What results did they achieve?

**Document findings in RESEARCH_NEXT_STEPS.md:**
```markdown
## Research Finding: <Technique Name>
**Source**: <URL>
**Key Idea**: <1-2 sentence summary>
**Why It Might Help**: <reasoning based on our bottleneck>
**Implementation Complexity**: Low/Medium/High
**Expected Impact**: +X.XX potential improvement
```

---

## STEP 3: HYPOTHESIZE

**Form a clear, testable hypothesis based on research + past learnings:**

```markdown
## Hypothesis: <Descriptive Name>

**Claim**: If we <implement X>, then <metric Y will improve> because <reasoning Z>.

**Based on**:
- Research finding: <what you found>
- Past experiment learning: <what we know>

**Implementation approach**:
1. <Step 1>
2. <Step 2>
3. <Step 3>

**Success criteria**:
- Score > 1.0329 (current best)
- Time < 60 min projected
```

---

## STEP 4: BUILD

**Create the experiment implementation:**

```bash
mkdir -p experiments/<approach_name>
```

**Required files:**

1. **optimizer.py** - The core algorithm
```python
class NewOptimizer:
    def __init__(self, ...):
        # Configuration

    def estimate_sources(self, sample, meta, ...):
        # Your new approach
        return candidates, best_rmse, ...
```

2. **run.py** - Test harness with MLflow logging
```python
# Must support:
# --workers 7      (for G4dn simulation)
# --shuffle        (for balanced batches)
# --max-samples N  (for quick validation)

# Must log to MLflow:
# - submission_score
# - rmse, rmse_1src, rmse_2src
# - projected_400_samples_min
```

3. **__init__.py** - Module exports

**Implementation checklist:**
- [ ] Uses simulator at inference (required by competition)
- [ ] Handles both 1-source and 2-source problems
- [ ] Returns up to 3 diverse candidates per sample
- [ ] Logs all metrics to MLflow
- [ ] Prints clear results summary

---

## STEP 5: TEST

**Quick validation first (2-3 min):**
```bash
uv run python experiments/<name>/run.py --workers 7 --shuffle --max-samples 10
```

Check:
- Does it run without errors?
- Is the score reasonable (> 0.9)?
- Is the time reasonable (< 2 min for 10 samples)?

**If promising, full test (10-12 min):**
```bash
uv run python experiments/<name>/run.py --workers 7 --shuffle
```

**Expected output:**
```
RESULTS
======================================================================
RMSE:             X.XXXXXX +/- X.XXXXXX
Submission Score: X.XXXX
Projected (400):  XX.X min

  1-source: RMSE=X.XXXX (n=XX)
  2-source: RMSE=X.XXXX (n=XX)

Baseline: 1.0329 @ 57.0 min
This run: X.XXXX @ XX.X min
[IMPROVED/NO IMPROVEMENT/OVER BUDGET]
======================================================================
```

---

## STEP 6: ANALYZE

**Think deeply about the results:**

1. **Did it beat baseline (1.0329 @ 57.0 min)?**
   - If YES: What specifically helped? Which component?
   - If NO: Why did it fail? What went wrong?

2. **Metric breakdown:**
   - Did 1-source improve? By how much?
   - Did 2-source improve? By how much?
   - Did time increase or decrease?

3. **Root cause analysis:**
   - If over budget: What dominated runtime?
   - If accuracy dropped: Which samples failed?
   - If succeeded: What was the key insight?

4. **Pattern recognition:**
   - Does this confirm or contradict past learnings?
   - Has the bottleneck shifted?
   - What new questions does this raise?

5. **Next direction:**
   - Based on this result, what should we research next?
   - Should we iterate on this approach or try something different?

---

## STEP 7: UPDATE

**Update ITERATION_LOG.md:**
```markdown
## Experiment: <Name> - <Date>
- **Hypothesis**: <what we tested>
- **Result**: Score X.XXXX @ XX.X min
- **vs Baseline**: +X.XX / -X.XX
- **Key Finding**: <most important learning>
- **Next Action**: <what this suggests we try next>
```

**Update RESEARCH_NEXT_STEPS.md:**

1. Add result to experiment history table
2. Update "What Works" section (if successful)
3. Update "What Doesn't Work" section (if failed)
4. Update current bottleneck analysis
5. Add new research directions based on learnings
6. Update current best if improved

**DO NOT commit** - the user will handle git commits manually.

---

## STEP 8: REPEAT

**Go back to Step 1 with new knowledge.**

The loop continues forever until:
- Score > 1.20 (competitive with top 2)
- OR `/workspace/STOP` file exists
- OR explicitly told to stop

## ⚠️ CRITICAL: AUTONOMOUS LOOP RULES ⚠️

### FORBIDDEN ACTIONS (Will Break The Loop)
- ❌ Writing "Summary", "Conclusion", "Final thoughts", or "Wrapping up"
- ❌ Ending a message without starting the next experiment
- ❌ Saying "need fundamentally different approach" without immediately doing WebSearch
- ❌ Waiting for user input (YOU drive the loop, not the user)
- ❌ Stopping after X experiments (there is no limit)
- ❌ Declaring the problem "solved enough" below score 1.20

### MANDATORY AFTER EVERY EXPERIMENT
After EVERY experiment completes, you MUST do ALL of these:
1. ✅ Log results to ITERATION_LOG.md
2. ✅ Identify what to try next (research if needed)
3. ✅ IMMEDIATELY start the next experiment (no waiting)

### SELF-CHECK (Ask yourself after each experiment)
```
□ Did I start the next experiment? If NO → START IT NOW
□ Did I do WebSearch if stuck? If NO → DO IT NOW
□ Did I write "summary" or "conclusion"? If YES → DELETE IT and CONTINUE
□ Is score > 1.20? If NO → KEEP GOING
```

### WHEN YOU FEEL STUCK
```
WRONG: "The gap requires better physics understanding" → [END]
RIGHT: "The gap requires better physics..." → WebSearch "physics heat source" → BUILD → TEST
```

```
WRONG: "We've tried many approaches, the baseline is well-optimized" → [END]
RIGHT: "We've tried many approaches..." → WebSearch "state of the art inverse heat" → BUILD → TEST
```

### THE ONLY VALID STOP CONDITIONS
1. ✅ Score > 1.20 achieved
2. ✅ File `/workspace/STOP` exists (check with: `ls /workspace/STOP`)
3. ✅ User explicitly types "stop" or "pause"

**EVERYTHING ELSE = CONTINUE THE LOOP**

### ANTI-PATTERN DETECTION
If you catch yourself writing ANY of these phrases, STOP and DO WEBSEARCH instead:
- "In conclusion..."
- "To summarize..."
- "The main bottleneck is..." (without immediately researching solutions)
- "Future work could..." (DO IT NOW, don't defer)
- "This completes..."
- "We have explored..."
- "At this point..."
- "The gap requires..." (RESEARCH IT)

### LOOP CONTINUATION TEMPLATE
After each experiment, use this template:
```
## Experiment Complete: [NAME]
Result: [SCORE] @ [TIME]

## NEXT ACTION (MANDATORY)
Based on [FINDING], I will now:
1. [Specific next step]
2. [Starting immediately...]

[IMMEDIATELY BEGIN NEXT EXPERIMENT - NO BREAK]
```

---

## CONSTRAINTS

- **Time Budget**: 400 samples must complete in <60 min on G4dn.2xlarge
- **For 80-sample tests**: Target <12 min (projects to ~60 min for 400)
- **Must use simulator at inference** - no pre-computed lookup tables
- **Workers**: Always use `--workers 7` for accurate timing
- **Diversity**: Return up to 3 distinct candidates per sample

---

## KEY FILES

| File | Purpose |
|------|---------|
| `docs/RESEARCH_NEXT_STEPS.md` | **Primary state** - learnings, history, what works |
| `ITERATION_LOG.md` | Experiment tracking and results |
| `experiments/` | All experimental code |
| `CLAUDE.md` | Project constraints (read once at start) |
| `src/` | Production code (only after validation) |

---

## BREAKTHROUGH INSIGHT

**The gap to top teams (0.20) requires discovering something new.**

Top teams are at 94% of theoretical max (1.3). They likely discovered a technique or insight that gives them a fundamental advantage. Your job is to RESEARCH and EXPERIMENT until you find it.

**Possible breakthrough areas:**
- Better 2-source decomposition (ICA variants, NMF, learned methods)
- Faster simulation (reduced order models, neural surrogates)
- Better optimization (gradient-based, surrogate-assisted, warm starting)
- Physics-informed approaches (adjoint methods, Green's functions)
- Problem-specific insights (sensor placement, time series features)

---

## EXAMPLE CYCLE

```
REFLECT:
  "Current best is 1.0329. 2-source RMSE (0.27) is 2x worse than 1-source (0.14).
   Past experiments show ICA helps but is slow. Multi-start didn't help.
   Bottleneck: 2-source position estimation accuracy."

RESEARCH:
  WebSearch: "multiple heat source localization fast algorithm"
  Found: Paper on "reciprocity gap method" - uses adjoint solutions for O(1) source location
  WebFetch: <paper url>
  Key idea: Pre-compute adjoint fields, then source location is just inner products

HYPOTHESIZE:
  "If we pre-compute adjoint fields for a grid of test positions, we can evaluate
   1000s of candidate positions in milliseconds instead of seconds. This could
   give us much better initialization for CMA-ES."

BUILD:
  Create experiments/adjoint_init/optimizer.py
  - Pre-compute adjoint fields on 20x10 grid
  - Use inner products to score candidate positions
  - Feed best positions to CMA-ES

TEST:
  uv run python experiments/adjoint_init/run.py --workers 7 --shuffle --max-samples 10
  Result: 1.05 @ 65 min (promising score but over budget)

ANALYZE:
  "Adjoint pre-computation takes 30 sec per sample - too slow.
   BUT: the position estimates are excellent (0.15 RMSE before CMA-ES).
   Idea: Cache adjoint fields across samples with similar kappa?"

UPDATE:
  - Log result: 1.05 @ 65 min, over budget but best accuracy yet
  - Learning: Adjoint gives great positions but too slow per-sample
  - New hypothesis: "Shared adjoint cache for similar samples"

REPEAT:
  Back to REFLECT with new knowledge about adjoint methods...
```

---

## 🚀 IMMEDIATE KICKSTART (Do This NOW)

**Step 1: Check for untested experiments**
```bash
# List experiment folders
ls experiments/

# For each folder, check if it has results
ls results/ | grep <experiment_name>
```

**Step 2: If untested experiments exist → RUN THEM**
```bash
# Example: Run weighted_centroid if untested
uv run python experiments/weighted_centroid/run.py --workers 7 --shuffle
```

**Step 3: If all experiments tested → DO RESEARCH**
```
WebSearch: "inverse heat source identification state of the art 2024"
WebSearch: "fast heat source localization algorithm"
WebSearch: "blind source separation thermal"
```

**Step 4: BUILD what you find → TEST it → REPEAT**

---

## RESEARCH QUERIES TO TRY WHEN STUCK

Copy-paste these searches:
- `"heat source identification" "neural network" fast`
- `"inverse heat conduction" "machine learning"`
- `"thermal source localization" algorithm`
- `"reciprocity gap" heat source`
- `"Green's function" heat source identification`
- `"adjoint method" thermal inverse problem`
- `"reduced order model" heat equation`
- `"surrogate model" PDE optimization`

---

**Your goal: Get as close to 1.3 as possible through continuous research and experimentation.**

**YOU drive everything. There is no queue. DISCOVER what works. DO NOT STOP.**

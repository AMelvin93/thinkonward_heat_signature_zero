# Heat Signature Zero - Autonomous Research & Optimization Loop

## MISSION
Achieve the highest possible score (target: 1.3 theoretical max, 1.22+ competitive).
Current best: **1.0329 @ 57.0 min**. Gap to close: **+0.20**

## LEADERBOARD CONTEXT
```
#1  Team Jonas M     1.2268  (94.4% of max)
#2  Team kjc         1.2265  (94.4% of max)
#3  MGöksu           1.1585  (89.1% of max)
--- TARGET ---       1.15+   (88.5% of max)
--- WE ARE NOW ---   1.0329  (79.5% of max)
```

## THE AUTONOMOUS LOOP

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CONTINUOUS RESEARCH CYCLE                        │
│                                                                     │
│  ┌─────────────┐                                                   │
│  │   REFLECT   │ ← What do we know? What's the bottleneck?         │
│  └──────┬──────┘                                                   │
│         ▼                                                          │
│  ┌─────────────┐                                                   │
│  │  RESEARCH   │ ← WebSearch for techniques to solve bottleneck    │
│  └──────┬──────┘                                                   │
│         ▼                                                          │
│  ┌─────────────┐                                                   │
│  │ HYPOTHESIZE │ ← Form hypothesis: "X could work because Y"       │
│  └──────┬──────┘                                                   │
│         ▼                                                          │
│  ┌─────────────┐                                                   │
│  │ EXPERIMENT  │ ← Implement and test the hypothesis               │
│  └──────┬──────┘                                                   │
│         ▼                                                          │
│  ┌─────────────┐                                                   │
│  │   ANALYZE   │ ← Did it work? Why or why not?                    │
│  └──────┬──────┘                                                   │
│         ▼                                                          │
│  ┌─────────────┐                                                   │
│  │   UPDATE    │ ← Update docs with learnings, loop back           │
│  └──────┬──────┘                                                   │
│         │                                                          │
│         └──────────────────► REPEAT FOREVER                        │
└─────────────────────────────────────────────────────────────────────┘
```

## PHASE 1: REFLECT (Before Every Experiment)

Read the current state and identify the bottleneck:

```bash
cat docs/RESEARCH_NEXT_STEPS.md   # Full history and learnings
cat ITERATION_LOG.md              # Recent experiments
```

**Ask yourself:**
1. What is our current best score and why?
2. What is the SINGLE BIGGEST bottleneck right now?
3. What have we tried that DIDN'T work? (Don't repeat failures)
4. What patterns have emerged from past experiments?

**Current known bottleneck:** 2-source RMSE (~0.27) is 2x worse than 1-source (~0.14)

## PHASE 2: RESEARCH (Web Search for Solutions)

**Based on the bottleneck, search for solutions:**

```
WebSearch: "<bottleneck> solution techniques"
WebSearch: "inverse heat conduction <specific problem>"
WebSearch: "heat source localization <approach>"
```

**Example research queries:**
- "multiple heat source identification algorithm"
- "blind source separation thermal imaging"
- "fast inverse heat equation solver"
- "CMA-ES convergence acceleration"
- "surrogate-assisted optimization thermal"
- "adjoint method heat source"
- "neural network PDE inverse problem"

**Read promising results:**
```
WebFetch: <url>
→ What technique do they use?
→ How does it work?
→ Can we implement it?
→ What results did they achieve?
```

**Add findings to RESEARCH_NEXT_STEPS.md:**
```markdown
## Research Finding: <Technique Name>
**Source**: <URL>
**Key Idea**: <1-2 sentence summary>
**Why It Might Help**: <reasoning>
**Implementation Complexity**: Low/Medium/High
**Expected Impact**: +X.XX potential improvement
```

## PHASE 3: HYPOTHESIZE (Form a Testable Hypothesis)

Based on research + past learnings, form a clear hypothesis:

```markdown
## Hypothesis: <Name>

**Claim**: If we <do X>, then <Y will improve> because <reasoning>.

**Based on**:
- Research finding: <source>
- Past experiment: <what we learned>

**Test plan**:
1. Implement <specific change>
2. Run with --max-samples 10 for quick validation
3. If promising, run full 80-sample test

**Success criteria**: Score > 1.0329 AND time < 60 min
```

## PHASE 4: EXPERIMENT (Implement and Test)

**Quick validation first (2-3 min):**
```bash
uv run python experiments/<name>/run.py --workers 7 --shuffle --max-samples 10
```

**If promising, full test (10-12 min):**
```bash
uv run python experiments/<name>/run.py --workers 7 --shuffle
```

**Always log to MLflow and track:**
- submission_score
- rmse_1src, rmse_2src
- projected_400_samples_min

## PHASE 5: ANALYZE (Deep Analysis)

**After every experiment, think deeply:**

1. **Did it work?** Compare to baseline (1.0329 @ 57.0 min)
2. **If YES**: What specifically helped? Can we push further?
3. **If NO**: Why did it fail? What does this teach us?
4. **What's the NEW bottleneck?** (It may have shifted)

**Update Meta-Patterns:**
```markdown
## What We've Learned

### Techniques that WORK:
- <technique>: +X% improvement because <why>

### Techniques that DON'T WORK:
- <technique>: failed because <why>

### Current Bottleneck:
- <specific bottleneck> is limiting our score

### Next Research Direction:
- Based on above, we should research <topic>
```

## PHASE 6: UPDATE (Document Everything)

**Update RESEARCH_NEXT_STEPS.md with:**
1. Experiment result (score, time, key metrics)
2. What we learned (why it worked/failed)
3. New hypotheses generated
4. Updated bottleneck analysis
5. Next research direction

**Commit to git:**
```bash
git add -A
git commit -m "[SCORE: X.XXXX] <approach>: <key finding>"
```

## CONSTRAINTS

- **Time Budget**: Must complete 400 samples in <60 min on G4dn.2xlarge
- **For 80-sample tests**: Target <12 min (projects to ~60 min for 400)
- **Must use simulator at inference** - no pre-computed solutions
- **Workers**: Always use `--workers 7` for accurate timing

## STOP CONDITIONS

**Keep going until:**
- Score > 1.20 (competitive with top 2)
- OR STOP file exists at /workspace/STOP
- OR explicitly told to stop

**Do NOT stop just because:**
- An experiment failed
- You feel stuck (do more research!)
- You've done "enough" experiments

## KEY INSIGHT

**The gap to top teams (0.20) requires discovering something new.**

Top teams are at 94% of theoretical max. They likely discovered a technique
or insight that we haven't found yet. Your job is to RESEARCH until you find it.

Possible breakthrough areas:
- Better 2-source decomposition (ICA variants, NMF, learned decomposition)
- Faster simulation (reduced order models, neural surrogates)
- Better optimization (warm starting, surrogate-assisted)
- Problem-specific insights (physics-informed approaches)

## EXAMPLE CYCLE

```
REFLECT: "2-source RMSE is 0.27, twice as bad as 1-source. This is the bottleneck."

RESEARCH: WebSearch "multiple heat source separation algorithm"
          Found paper on "adjoint-based source inversion" - O(1) gradient computation

HYPOTHESIZE: "If we use adjoint gradients instead of finite differences in CMA-ES,
             we could do 10x more iterations in the same time budget."

EXPERIMENT: Implement adjoint_optimizer.py, test with --max-samples 10
            Result: 1.05 score but 80 min runtime - too slow

ANALYZE: "Adjoint computation itself is expensive. But the GRADIENT INFORMATION
         is valuable. What if we use it just for initialization?"

UPDATE: Add finding to docs. New hypothesis: "Adjoint for init only"

REPEAT: Research "warm start optimization gradient" ...
```

## FILES

- `docs/RESEARCH_NEXT_STEPS.md` - **Primary state** (learnings, hypotheses, history)
- `ITERATION_LOG.md` - Experiment tracking
- `experiments/` - All experimental code
- `CLAUDE.md` - Project constraints (read once)

---

## START NOW

1. Read `docs/RESEARCH_NEXT_STEPS.md` to understand current state
2. Identify the current bottleneck
3. Do web research to find potential solutions
4. Form a hypothesis and test it
5. Analyze, learn, and repeat

**Your goal: Get as close to 1.3 as possible through continuous research and experimentation.**

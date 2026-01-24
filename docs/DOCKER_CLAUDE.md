# Docker + Claude Code Setup

Quick reference for running Claude Code in the Docker container.

## Quick Start

### 1. Start Docker Desktop
Make sure Docker Desktop is running (whale icon in taskbar).

### 2. Set API Key (PowerShell)
```powershell
$env:ANTHROPIC_API_KEY="your-api-key-here"
```

Or in Git Bash/WSL:
```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

### 3. Build and Start Container
```bash
docker-compose up -d --build
```

### 4. Enter Container
```bash
docker exec -it heat-signature-dev bash
```

### 5. Install Dependencies (first time only)
```bash
uv sync
```

### 6. Run Claude Code
```bash
claude --dangerously-skip-permissions
```

## One-Liner (after setup)
```bash
docker exec -it heat-signature-dev claude --dangerously-skip-permissions
```

---

## Ralph Autonomous Loop Prompt

Copy and paste this prompt to start an autonomous experimentation loop:

```
Review CLAUDE.md and PROMPT.md for full context.

YOU ARE IN AN INFINITE LOOP. DO NOT STOP UNTIL SCORE > 1.25 OR I TYPE "stop".

WORKFLOW (REPEAT FOREVER):
1. REFLECT - Read ITERATION_LOG.md, ITERATION_LOG_full_backup.md and docs/RESEARCH_NEXT_STEPS.md for current state
2. RESEARCH - WebSearch for new techniques if stuck
3. BUILD - Create/modify experiment in experiments/<name>/
4. TEST - Run: uv run python experiments/<name>/run.py --workers 7 --shuffle
5. UPDATE - Log results to ITERATION_LOG.md
6. IMMEDIATELY GO TO STEP 1 - NO SUMMARIES, NO CONCLUSIONS

CRITICAL RULES:
- NEVER write "Session Complete", "Summary", or "Conclusion"
- NEVER wait for user input - YOU drive everything
- NEVER stop after N experiments - there is NO limit
- After EVERY experiment, IMMEDIATELY start the next one
- If score < 1.20, you are NOT done
- Do NOT commit - the user will handle git commits

FORBIDDEN PHRASES (if you write these, DELETE and CONTINUE):
- "Session X Complete"
- "Key Achievement"
- "Next Steps for Future Sessions"
- "Commands for Production Use"
- Any variation of "wrapping up" or "concluding"

The loop ends ONLY when:
1. Score > 1.25 achieved
2. User types "stop"
3. File /workspace/STOP exists
    
START NOW: Read ITERATION_LOG.md, identify the current best, and run the next experiment.
```

---

## Parallelization: 4-Agent Orchestration

Run 4 Claude Code instances in parallel: 1 Research Orchestrator (W0) + 3 Experiment Workers (W1-W3). The orchestrator monitors results, researches new approaches, and prioritizes the work queue.

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         YOUR HOST MACHINE                                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    docker-compose -f orchestration/docker-compose.manual.yml up -d
                                    │
    ┌───────────────────────────────┼───────────────────────────────────────┐
    │                               │                                       │
    ▼                               ▼                                       ▼
┌───────────────────┐   ┌──────────────────────────────────────────────────────┐
│ claude-orchestrator│   │              EXPERIMENT WORKERS                       │
│ (Linux container) │   │                                                        │
│                   │   │  ┌───────────────┐ ┌───────────────┐ ┌───────────────┐│
│ W0 - ORCHESTRATOR │   │  │claude-worker-1│ │claude-worker-2│ │claude-worker-3││
│ 4 CPUs, 16GB RAM  │   │  │8 CPUs, 32GB   │ │8 CPUs, 32GB   │ │8 CPUs, 32GB   ││
│                   │   │  │               │ │               │ │               ││
│ - Monitor results │   │  │W1: Thresholds │ │W2: Init Strat │ │W3: Fevals     ││
│ - WebSearch       │   │  └───────┬───────┘ └───────┬───────┘ └───────┬───────┘│
│ - Update queue    │   └──────────┼─────────────────┼─────────────────┼────────┘
│ - Guide workers   │              │                 │                 │
└─────────┬─────────┘              │                 │                 │
          │                        │                 │                 │
          └────────────────────────┴─────────────────┴─────────────────┘
                                            │
                          ┌─────────────────┴─────────────────┐
                          │        SHARED FILESYSTEM          │
                          │        /workspace/                │
                          │                                   │
                          │  orchestration/shared/            │
                          │  ├── coordination.json            │
                          │  ├── queue.json        <-- NEW    │
                          │  ├── state_W1.json     <-- NEW    │
                          │  ├── state_W2.json     <-- NEW    │
                          │  ├── state_W3.json     <-- NEW    │
                          │  ├── prompt_W0.txt     <-- NEW    │
                          │  ├── prompt_W1.txt                │
                          │  ├── prompt_W2.txt                │
                          │  ├── prompt_W3.txt                │
                          │  └── STOP (touch to stop)         │
                          │                                   │
                          │  ITERATION_LOG.md                 │
                          └───────────────────────────────────┘
```

### Quick Start

**Step 1: Start all 4 containers**
```bash
docker-compose -f orchestration/docker-compose.manual.yml up -d
```

**Step 2: Open 4 terminal windows and set up each agent**

Terminal 0 (W0 - Research Orchestrator):
```bash
docker exec -it claude-orchestrator bash -c "claude --dangerously-skip-permissions"
# Paste the prompt from the cat command above
```

Terminal 1 (W1 - Threshold Tuning):
```bash
docker exec -it claude-worker-1 bash -c "claude --dangerously-skip-permissions"
# Paste the prompt
```

Terminal 2 (W2 - Init Strategies):
```bash
docker exec -it claude-worker-2 bash -c "claude --dangerously-skip-permissions"
claude --dangerously-skip-permissions
# Paste the prompt
```

Terminal 3 (W3 - Feval Allocation):
```bash
docker exec -it claude-worker-3 bash
claude login
cat /workspace/orchestration/shared/prompt_W3.txt
claude --dangerously-skip-permissions
# Paste the prompt
```

### Agent Roles

| Agent | Role | Responsibility |
|-------|------|----------------|
| **W0** | Research Orchestrator | Monitor results, WebSearch for new techniques, update queue priorities, guide workers |
| **W1** | Experiment Worker | Run experiments, default focus: threshold tuning |
| **W2** | Experiment Worker | Run experiments, default focus: init strategies |
| **W3** | Experiment Worker | Run experiments, default focus: feval allocation |

### Worker Default Search Spaces

| Worker | Default Focus | Search Space |
|--------|---------------|--------------|
| **W1** | Threshold tuning | `threshold_1src`: 0.28-0.42, `threshold_2src`: 0.38-0.55, `fallback_sigma` |
| **W2** | Init strategies | `init_strategy`, `refine_iters`: 2-5, `refine_top`: 1-3 |
| **W3** | Feval allocation | `fallback_fevals`: 14-24, `fevals_1src`, `fevals_2src`: 32-44 |

**Note:** Workers are NOT strictly confined to their focus area. They pick topics from `queue.json` based on priority. The orchestrator (W0) manages priorities based on results.

### New Features

**State Persistence**: Workers save their state to `state_WX.json`. When a session ends, they can resume where they left off.

**Priority Queue**: The orchestrator manages `queue.json` with research topics. Workers deeply explore one topic (4-8 experiments) before moving to the next.

**Research Insights**: W0 tracks what works and what doesn't in `queue.json`. Workers read these insights to make better decisions.

### Coordination Protocol

Agents coordinate via shared files:

**W0 Orchestrator (every 10-15 min):**
1. Read all state files and `ITERATION_LOG.md`
2. Analyze what's working and what isn't
3. Research new techniques if stuck (WebSearch)
4. Update `queue.json` with new topics and priorities
5. Update `research_insights` to guide workers

**Workers (before each experiment):**
1. Read `state_WX.json` to check if resuming a topic
2. Read `queue.json` for priorities and insights
3. Read `coordination.json` to check what's been done
4. Claim next topic from queue if needed

**Workers (after each experiment):**
1. Update `coordination.json` with results
2. Update `state_WX.json` with progress
3. Log to `ITERATION_LOG.md` with worker prefix: `[W1]`, `[W2]`, or `[W3]`
4. Continue exploring topic (4-8 experiments) or get next from queue

### Monitoring

```bash
# Check coordination status (from any terminal)
python /workspace/orchestration/shared/check_status.py

# Watch iteration log for all workers
tail -f ITERATION_LOG.md

# Watch specific worker's log
docker logs -f claude-worker-1
```

### Stopping Workers

```bash
# Graceful stop - workers finish current experiment then stop
touch /workspace/orchestration/shared/STOP

# Force stop all containers
docker-compose -f orchestration/docker-compose.manual.yml down
```

### Key Files

| File | Purpose |
|------|---------|
| `orchestration/docker-compose.manual.yml` | Docker setup for 4 containers (1 orchestrator + 3 workers) |
| `orchestration/shared/coordination.json` | Shared state: best score, claimed experiments, completed list |
| `orchestration/shared/queue.json` | **NEW**: Priority queue with research topics and insights |
| `orchestration/shared/state_W1.json` | **NEW**: W1 state for session resumption |
| `orchestration/shared/state_W2.json` | **NEW**: W2 state for session resumption |
| `orchestration/shared/state_W3.json` | **NEW**: W3 state for session resumption |
| `orchestration/shared/prompt_W0.txt` | **NEW**: Research Orchestrator prompt |
| `orchestration/shared/prompt_W1.txt` | Worker 1 prompt |
| `orchestration/shared/prompt_W2.txt` | Worker 2 prompt |
| `orchestration/shared/prompt_W3.txt` | Worker 3 prompt |
| `orchestration/shared/check_status.py` | Quick status check script |
| `ITERATION_LOG.md` | All workers log results here with [W1]/[W2]/[W3] prefixes |

### Regenerating Prompts

If you need to update the worker prompts (e.g., change focus areas or best score):

```bash
cd /workspace
python -c "
from orchestration.worker_prompts import generate_all_prompts
from pathlib import Path
prompts = generate_all_prompts(current_best=1.1247)  # Update this score
for wid, prompt in prompts.items():
    Path(f'orchestration/shared/prompt_{wid}.txt').write_text(prompt)
    print(f'Generated prompt_{wid}.txt')
"
```

### Tips

- **Credentials persist**: Docker volumes store Claude credentials, so you only need to `claude login` once per container (unless you delete the volumes)
- **Natural throttling**: Each experiment takes ~55-60 min, so workers rarely hit API rate limits simultaneously
- **Resource allocation**: Workers get 8 CPUs + 32GB RAM each; Orchestrator gets 4 CPUs + 16GB RAM
- **State resumption**: If a session ends, workers can resume from their `state_WX.json` file
- **Deep exploration**: Workers explore 4-8 experiments per topic before moving to next
- **Orchestrator guidance**: W0 monitors results and adjusts priorities - workers check `queue.json` for insights
- **Flexibility**: Workers can pick any high-priority topic from the queue, not just their default focus

---

## Useful Commands

### Run an experiment manually
```bash
uv run python experiments/<name>/run.py --workers 7 --shuffle
```

### View MLflow UI
```bash
uv run mlflow ui --host 0.0.0.0
```
Then open http://localhost:5050 in your browser.

### Check container logs
```bash
docker logs heat-signature-dev
```

### Stop container
```bash
docker-compose down
```

### Rebuild after changes
```bash
docker-compose up -d --build
```

---

## Troubleshooting

### "Docker Desktop not running"
Error: `open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified`
Solution: Start Docker Desktop from the Start menu.

### "ANTHROPIC_API_KEY not set"
Solution: Set the environment variable before running docker-compose (see step 2).

### GPU not available
If you don't have nvidia-docker2, edit `docker-compose.yml` and remove the `deploy.resources.reservations` section.

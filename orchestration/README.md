# Multi-Worker Orchestration for Heat Signature Zero

Run 3 parallel Claude Code instances to 3x your experiment throughput.

## Context-Clearing Mode (v4)

Workers now **exit after each experiment** instead of running an infinite loop. This clears the context window between experiments, preventing context accumulation. The orchestrator automatically restarts workers, and the resume logic ensures no work is lost.

**How it works:**
1. Worker picks an experiment, executes it, writes STATE.json and SUMMARY.md
2. Worker **exits cleanly** (not a crash - intentional exit)
3. Orchestrator detects exit and restarts Claude Code
4. Fresh Claude Code instance starts with empty context
5. Worker checks for incomplete work (resume logic), finds none
6. Worker picks next experiment from queue
7. Loop continues...

## Quick Start (Recommended)

### Option 1: Interactive Mode (Easiest)

This starts 3 terminal windows, each with a worker. You paste the prompt manually.

**PowerShell (Windows):**
```powershell
# Set API key first
$env:ANTHROPIC_API_KEY="your-key-here"

# Generate prompts and start terminals
.\orchestration\run_orchestration.ps1 -Interactive
```

**Then in each terminal:**
1. Run: `claude --dangerously-skip-permissions`
2. Paste the prompt that was displayed

### Option 2: Using Existing Container

If you already have `heat-signature-dev` running:

```bash
# Terminal 1
docker exec -it heat-signature-dev bash
cat /workspace/orchestration/shared/prompt_W1.txt
claude --dangerously-skip-permissions
# Paste prompt

# Terminal 2 (new window)
docker exec -it heat-signature-dev bash
cat /workspace/orchestration/shared/prompt_W2.txt
claude --dangerously-skip-permissions
# Paste prompt

# Terminal 3 (new window)
docker exec -it heat-signature-dev bash
cat /workspace/orchestration/shared/prompt_W3.txt
claude --dangerously-skip-permissions
# Paste prompt
```

### Option 3: Docker Compose (3 Separate Containers)

```bash
# Set API key
export ANTHROPIC_API_KEY="your-key-here"

# Start all workers
docker-compose -f orchestration/docker-compose.workers.yml up -d

# Attach to each worker
docker attach claude-worker-1  # Terminal 1
docker attach claude-worker-2  # Terminal 2
docker attach claude-worker-3  # Terminal 3
```

## How It Works

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    SHARED FILE SYSTEM                        │
│  orchestration/shared/                                       │
│  ├── coordination.json  (current best, completed exps)      │
│  ├── prompt_W1.txt      (worker 1 prompt)                   │
│  ├── prompt_W2.txt      (worker 2 prompt)                   │
│  ├── prompt_W3.txt      (worker 3 prompt)                   │
│  └── all_results.jsonl  (full history)                      │
└─────────────────────────────────────────────────────────────┘
        ▲                    ▲                    ▲
        │                    │                    │
┌───────┴───────┐   ┌───────┴───────┐   ┌───────┴───────┐
│   WORKER W1   │   │   WORKER W2   │   │   WORKER W3   │
│               │   │               │   │               │
│  Threshold    │   │  Init         │   │  Feval        │
│  Tuning       │   │  Strategies   │   │  Allocation   │
│               │   │               │   │               │
│  experiments/ │   │  experiments/ │   │  experiments/ │
│  W1_*/        │   │  W2_*/        │   │  W3_*/        │
└───────────────┘   └───────────────┘   └───────────────┘
```

### Worker Focus Areas

| Worker | Focus Area | Parameters Explored |
|--------|------------|---------------------|
| **W1** | Threshold Tuning | `threshold_1src`, `threshold_2src`, `fallback_sigma` |
| **W2** | Init Strategies | `init_strategy`, `refine_iters`, `refine_top` |
| **W3** | Feval Allocation | `fallback_fevals`, `fevals_1src`, `fevals_2src` |

### Coordination Protocol

1. **Before each experiment:**
   - Worker reads `coordination.json`
   - Checks if config already tried (by hash)
   - Skips if duplicate

2. **After each experiment:**
   - Worker logs to `ITERATION_LOG.md` with `[W1]`, `[W2]`, or `[W3]` prefix
   - Results automatically visible to other workers

3. **New best found:**
   - Worker notes it in the log
   - Other workers see updated baseline on next iteration

## Files

| File | Purpose |
|------|---------|
| `coordinator.py` | Shared state management with file locking |
| `orchestrator.py` | Main Python orchestrator (auto mode) |
| `worker_prompts.py` | Prompt templates for each worker |
| `docker-compose.workers.yml` | Docker setup for 3 containers |
| `run_orchestration.ps1` | PowerShell launcher (Windows) |
| `run_orchestration.sh` | Bash launcher (WSL/Linux) |

## Monitoring

### Check Status

```bash
# View coordination state
cat orchestration/shared/coordination.json | python -m json.tool

# Watch results in real-time
tail -f orchestration/shared/all_results.jsonl

# Check iteration log
tail -100 ITERATION_LOG.md
```

### Stop All Workers

Create a stop file:
```bash
touch orchestration/shared/STOP
```

Workers will stop at their next checkpoint.

## Troubleshooting

### "Workers are duplicating experiments"

The coordination file may not be syncing properly. Check:
1. All workers are reading from the same path
2. File locking is working (no permission issues)

### "One worker is faster than others"

This is normal - different parameter spaces have different evaluation times.
The threshold tuning worker may run more experiments because its changes are smaller.

### "Claude Code crashed with stack overflow"

Run workers sequentially instead of in parallel, or increase Node.js stack:
```bash
NODE_OPTIONS="--stack-size=8192" claude --dangerously-skip-permissions
```

### "Results not being logged"

Check that workers are writing to `ITERATION_LOG.md` with their worker ID prefix.
The coordination system tracks by config hash, not by log entries.

## Advanced Usage

### Custom Worker Configuration

Edit `worker_prompts.py` to change:
- Focus areas for each worker
- Search space bounds
- Fixed parameters

### Adding More Workers

1. Add configuration to `WORKER_CONFIGS` in `worker_prompts.py`
2. Add service to `docker-compose.workers.yml`
3. Run with `--workers N` flag

### Combining with Optuna

For smarter search, integrate Optuna:
```python
# In each worker's experiment loop
import optuna
study = optuna.load_study(
    study_name="heat_signature",
    storage="sqlite:////workspace/orchestration/shared/optuna.db"
)
config = study.ask()  # Get next config to try
# ... run experiment ...
study.tell(trial, score)  # Report result
```

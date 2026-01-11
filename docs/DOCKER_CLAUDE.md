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

YOU ARE IN AN INFINITE LOOP. DO NOT STOP UNTIL SCORE > 1.20 OR I TYPE "stop".

WORKFLOW (REPEAT FOREVER):
1. REFLECT - Read ITERATION_LOG.md and docs/RESEARCH_NEXT_STEPS.md for current state
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
1. Score > 1.20 achieved
2. User types "stop"
3. File /workspace/STOP exists

START NOW: Read ITERATION_LOG.md, identify the current best, and run the next experiment.
```

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

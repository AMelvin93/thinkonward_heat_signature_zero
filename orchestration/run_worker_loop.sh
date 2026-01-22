#!/bin/bash
# Worker loop wrapper - restarts Claude Code after each experiment
# This implements the "Ralph Wiggum loop" with context clearing
#
# Usage:
#   ./orchestration/run_worker_loop.sh W1
#   ./orchestration/run_worker_loop.sh W2
#   etc.

WORKER_ID="${1:-W1}"
PROMPT_FILE="/workspace/orchestration/shared/prompt_${WORKER_ID}.txt"
STOP_FILE="/workspace/orchestration/shared/STOP"

echo "=============================================="
echo "Worker Loop: ${WORKER_ID} (Context-Clearing Mode)"
echo "=============================================="
echo "Each experiment runs in a fresh Claude Code instance."
echo "Context is cleared between experiments."
echo "Create $STOP_FILE to stop the loop."
echo "=============================================="

# Main loop - runs forever until STOP file created
while true; do
    # Check for stop file
    if [ -f "$STOP_FILE" ]; then
        echo "[${WORKER_ID}] STOP file detected. Exiting loop."
        exit 0
    fi

    # Check if prompt file exists
    if [ ! -f "$PROMPT_FILE" ]; then
        echo "[${WORKER_ID}] Prompt file not found: $PROMPT_FILE"
        echo "[${WORKER_ID}] Run: uv run python orchestration/worker_prompts_v4.py"
        exit 1
    fi

    echo ""
    echo "[${WORKER_ID}] Starting Claude Code instance... ($(date))"
    echo "[${WORKER_ID}] Context will be cleared after this experiment."
    echo ""

    # Read prompt and pass to Claude Code
    PROMPT=$(cat "$PROMPT_FILE")

    # Run Claude Code with the prompt
    # --dangerously-skip-permissions: Auto-approve all tool calls
    # --print: Print responses (not interactive mode)
    # The worker will exit after completing one experiment
    claude --dangerously-skip-permissions --print "$PROMPT"

    EXIT_CODE=$?

    if [ $EXIT_CODE -eq 0 ]; then
        echo ""
        echo "[${WORKER_ID}] Claude Code exited cleanly (experiment completed)"
        echo "[${WORKER_ID}] Restarting with fresh context in 5 seconds..."
        sleep 5
    else
        echo ""
        echo "[${WORKER_ID}] Claude Code exited with code $EXIT_CODE"
        echo "[${WORKER_ID}] Waiting 30 seconds before retry..."
        sleep 30
    fi
done

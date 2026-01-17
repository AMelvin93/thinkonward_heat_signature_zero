#!/bin/bash
# Worker runner script - runs Claude Code in a loop with auto-restart
# Usage: ./orchestration/worker_runner.sh W1
#
# This script:
# 1. Reads the worker prompt
# 2. Starts Claude Code with --print mode
# 3. If Claude exits (crash, completion, etc.), restarts automatically
# 4. Stops only when STOP file exists or score > 1.25

set -e

WORKER_ID="${1:-W1}"
PROMPT_FILE="/workspace/orchestration/shared/prompt_${WORKER_ID}.txt"
STOP_FILE="/workspace/orchestration/shared/STOP"
COORD_FILE="/workspace/orchestration/shared/coordination.json"
LOG_FILE="/workspace/orchestration/shared/worker_${WORKER_ID}.log"

echo "========================================"
echo "Worker $WORKER_ID starting"
echo "========================================"
echo "Prompt: $PROMPT_FILE"
echo "Stop file: $STOP_FILE"
echo ""

# Check prompt exists
if [ ! -f "$PROMPT_FILE" ]; then
    echo "ERROR: Prompt file not found: $PROMPT_FILE"
    echo "Run: python orchestration/worker_prompts.py"
    exit 1
fi

# Function to check if we should stop
should_stop() {
    # Check stop file
    if [ -f "$STOP_FILE" ]; then
        echo "STOP file detected"
        return 0
    fi

    # Check if target score reached
    if [ -f "$COORD_FILE" ]; then
        BEST_SCORE=$(python3 -c "
import json
data = json.loads(open('$COORD_FILE').read())
print(data.get('best_score', 0))
" 2>/dev/null || echo "0")

        if (( $(echo "$BEST_SCORE >= 1.25" | bc -l) )); then
            echo "Target score reached: $BEST_SCORE"
            return 0
        fi
    fi

    return 1
}

# Function to get current best score for prompt update
get_best_score() {
    if [ -f "$COORD_FILE" ]; then
        python3 -c "
import json
data = json.loads(open('$COORD_FILE').read())
print(data.get('best_score', 1.1247))
" 2>/dev/null || echo "1.1247"
    else
        echo "1.1247"
    fi
}

# Main loop
ITERATION=0
MAX_RETRIES=100  # Prevent infinite crash loop

while [ $ITERATION -lt $MAX_RETRIES ]; do
    ITERATION=$((ITERATION + 1))

    # Check if we should stop
    if should_stop; then
        echo ""
        echo "========================================"
        echo "Worker $WORKER_ID stopping"
        echo "========================================"
        exit 0
    fi

    BEST_SCORE=$(get_best_score)
    echo ""
    echo "========================================"
    echo "Worker $WORKER_ID - Iteration $ITERATION"
    echo "Current best: $BEST_SCORE"
    echo "========================================"
    echo ""

    # Update prompt with current best score
    PROMPT=$(cat "$PROMPT_FILE" | sed "s/Current Best Score: [0-9.]*/Current Best Score: $BEST_SCORE/")

    # Run Claude Code
    # Using --print for non-interactive mode
    # The prompt tells Claude to run experiments in a loop
    START_TIME=$(date +%s)

    echo "$PROMPT" | claude --dangerously-skip-permissions --print 2>&1 | tee -a "$LOG_FILE" || true

    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))

    echo ""
    echo "Claude exited after ${DURATION}s"

    # Brief pause before restart (back off if very quick exit = likely error)
    if [ $DURATION -lt 60 ]; then
        echo "Quick exit detected, waiting 30s before retry..."
        sleep 30
    else
        echo "Restarting in 5s..."
        sleep 5
    fi
done

echo "Max retries reached, stopping worker $WORKER_ID"

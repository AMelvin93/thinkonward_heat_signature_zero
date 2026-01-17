#!/bin/bash
# Bash script to start orchestration from WSL or Linux
# Usage: ./orchestration/run_orchestration.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

WORKERS=${1:-3}
MODE=${2:-interactive}  # interactive or auto

echo "========================================"
echo "HEAT SIGNATURE ZERO - ORCHESTRATION"
echo "========================================"
echo "Project: $PROJECT_DIR"
echo "Workers: $WORKERS"
echo "Mode: $MODE"
echo ""

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "ERROR: ANTHROPIC_API_KEY not set"
    echo "Set it with: export ANTHROPIC_API_KEY='your-key'"
    exit 1
fi

# Create shared directory
SHARED_DIR="$SCRIPT_DIR/shared"
mkdir -p "$SHARED_DIR"

# Generate prompts
echo "Generating worker prompts..."
cd "$PROJECT_DIR"
python3 -c "
import sys
sys.path.insert(0, '.')
from orchestration.worker_prompts import generate_all_prompts
from pathlib import Path

prompts = generate_all_prompts()
shared_dir = Path('orchestration/shared')
shared_dir.mkdir(exist_ok=True)

for worker_id, prompt in prompts.items():
    prompt_file = shared_dir / f'prompt_{worker_id}.txt'
    prompt_file.write_text(prompt)
    print(f'  Generated: {prompt_file}')
"

if [ "$MODE" == "interactive" ]; then
    echo ""
    echo "INTERACTIVE MODE"
    echo "================"
    echo ""
    echo "To start each worker, open a new terminal and run:"
    echo ""

    for i in $(seq 1 $WORKERS); do
        WORKER_ID="W$i"
        echo "--- Worker $WORKER_ID ---"
        echo "docker exec -it heat-signature-dev bash"
        echo "cat /workspace/orchestration/shared/prompt_$WORKER_ID.txt"
        echo "claude --dangerously-skip-permissions"
        echo "(paste the prompt)"
        echo ""
    done

    echo ""
    echo "Or use tmux to run all in one terminal:"
    echo ""
    echo "tmux new-session -d -s worker1 'docker exec -it heat-signature-dev bash'"
    echo "tmux split-window -h 'docker exec -it heat-signature-dev bash'"
    echo "tmux split-window -v 'docker exec -it heat-signature-dev bash'"
    echo "tmux attach"

else
    echo ""
    echo "AUTO MODE - Starting Python orchestrator..."
    python3 orchestration/orchestrator.py --workers "$WORKERS"
fi

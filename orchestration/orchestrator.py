"""
Main orchestrator for multi-worker Claude Code experiment optimization.
Runs on Windows host, manages Docker containers.

Usage:
    python orchestration/orchestrator.py --workers 3
    python orchestration/orchestrator.py --workers 3 --dry-run
"""
import subprocess
import json
import time
import threading
import argparse
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.coordinator import Coordinator, format_status_report
# Use v5 enhanced tuning prompts (minimum 3 runs, time budget awareness, pivot strategy)
from orchestration.worker_prompts_v5 import get_worker_prompt_v5 as get_worker_prompt


# Worker configurations - defines focus areas for each worker
WORKER_CONFIGS = [
    {"focus_area": "Experiment Executor", "worker_id": "W1"},
    {"focus_area": "Experiment Executor", "worker_id": "W2"},
    {"focus_area": "Experiment Executor", "worker_id": "W3"},
    {"focus_area": "Experiment Executor", "worker_id": "W4"},
]


@dataclass
class WorkerState:
    """Track state of a single worker container."""
    worker_id: str
    container_name: str
    focus_area: str
    status: str = "stopped"  # stopped, starting, running, error
    process: Optional[subprocess.Popen] = None
    experiments_run: int = 0
    last_output: str = ""
    started_at: Optional[float] = None
    error_count: int = 0


class Orchestrator:
    """
    Manages multiple Claude Code workers in Docker containers.
    Coordinates via shared filesystem.
    """

    def __init__(
        self,
        num_workers: int = 3,
        project_dir: Optional[Path] = None,
        dry_run: bool = False,
        image_name: str = "heat-signature-zero:latest",
    ):
        self.num_workers = num_workers
        self.project_dir = Path(project_dir) if project_dir else Path.cwd()
        self.dry_run = dry_run
        self.image_name = image_name

        # Shared directory for coordination (inside project, mounted to containers)
        self.shared_dir = self.project_dir / "orchestration" / "shared"
        self.shared_dir.mkdir(parents=True, exist_ok=True)

        # Initialize coordinator
        self.coordinator = Coordinator(shared_dir=self.shared_dir)

        # Worker states
        self.workers: Dict[str, WorkerState] = {}

        # Control flags
        self.running = False
        self.stop_requested = False

        # Initialize workers
        self._init_workers()

    def _init_workers(self):
        """Initialize worker configurations."""
        for i in range(self.num_workers):
            worker_id = f"W{i + 1}"
            config = WORKER_CONFIGS[i % len(WORKER_CONFIGS)]

            self.workers[worker_id] = WorkerState(
                worker_id=worker_id,
                container_name=f"claude-worker-{i + 1}",
                focus_area=config["focus_area"],
            )

            # Register with coordinator
            self.coordinator.register_worker(worker_id, config["focus_area"])

    def _build_docker_command(self, worker: WorkerState) -> List[str]:
        """Build docker run command for a worker."""
        # Get the prompt for this worker
        prompt = get_worker_prompt(
            worker.worker_id,
            self.coordinator.get_current_best()["score"],
        )

        # Escape the prompt for shell
        escaped_prompt = prompt.replace('"', '\\"').replace('$', '\\$')

        # Build docker command
        cmd = [
            "docker", "run",
            "--rm",  # Remove container when done
            "--name", worker.container_name,
            "-v", f"{self.project_dir}:/workspace",
            "-e", f"ANTHROPIC_API_KEY={os.environ.get('ANTHROPIC_API_KEY', '')}",
            "-e", f"WORKER_ID={worker.worker_id}",
            "-e", "PYTHONUNBUFFERED=1",
            "--cpus", "8",
            "--memory", "32g",
            "-w", "/workspace",
            self.image_name,
            "bash", "-c",
            f'claude --dangerously-skip-permissions --print "{escaped_prompt}"'
        ]

        return cmd

    def _start_worker(self, worker: WorkerState) -> bool:
        """Start a worker container."""
        if self.dry_run:
            print(f"[DRY RUN] Would start {worker.worker_id}: {worker.focus_area}")
            worker.status = "running"
            return True

        try:
            # First, ensure no existing container with same name
            subprocess.run(
                ["docker", "rm", "-f", worker.container_name],
                capture_output=True,
                timeout=30,
            )

            # Build and start the command
            cmd = self._build_docker_command(worker)
            print(f"Starting {worker.worker_id}...")

            worker.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            worker.status = "running"
            worker.started_at = time.time()
            worker.error_count = 0

            return True

        except Exception as e:
            print(f"ERROR starting {worker.worker_id}: {e}")
            worker.status = "error"
            worker.error_count += 1
            return False

    def _monitor_worker(self, worker: WorkerState):
        """Monitor a running worker, capture output."""
        if not worker.process:
            return

        output_lines = []
        try:
            for line in iter(worker.process.stdout.readline, ''):
                if not line:
                    break

                output_lines.append(line.rstrip())
                worker.last_output = line.rstrip()

                # Check for experiment completion markers
                if "Score:" in line or "score:" in line:
                    print(f"[{worker.worker_id}] {line.rstrip()}")

                # Check for errors
                if "Error" in line or "ERROR" in line:
                    print(f"[{worker.worker_id}] {line.rstrip()}")

                # Heartbeat
                self.coordinator.heartbeat(worker.worker_id)

            # Process finished
            return_code = worker.process.wait()

            if return_code == 0:
                print(f"[{worker.worker_id}] Completed successfully")
                worker.experiments_run += 1
            else:
                print(f"[{worker.worker_id}] Exited with code {return_code}")
                worker.error_count += 1

            worker.status = "stopped"
            worker.process = None

        except Exception as e:
            print(f"[{worker.worker_id}] Monitor error: {e}")
            worker.status = "error"

    def _run_worker_loop(self, worker: WorkerState):
        """Main loop for a single worker - keeps restarting until stopped.

        Context-clearing behavior (v4):
        - Worker executes ONE experiment then exits cleanly
        - This loop restarts Claude Code, which clears context
        - Resume logic in worker prompt ensures continuity
        - Result: Fresh context for each experiment, no accumulation
        """
        while self.running and not self.stop_requested:
            # Check if target reached
            if self.coordinator.is_target_reached():
                print(f"[{worker.worker_id}] Target reached, stopping")
                break

            # Check for stop file
            if self.coordinator.should_stop():
                print(f"[{worker.worker_id}] Stop signal received")
                break

            # Start worker
            if self._start_worker(worker):
                self._monitor_worker(worker)

            # Brief pause before restart
            if self.running and not self.stop_requested:
                # Back off if errors
                wait_time = min(5 * (worker.error_count + 1), 60)
                print(f"[{worker.worker_id}] Waiting {wait_time}s before next run...")
                time.sleep(wait_time)

    def run(self):
        """Run the orchestrator - manages all workers."""
        print("=" * 60)
        print("HEAT SIGNATURE ZERO - PARALLEL ORCHESTRATOR")
        print("=" * 60)
        print(f"Workers: {self.num_workers}")
        print(f"Project: {self.project_dir}")
        print(f"Shared: {self.shared_dir}")
        print(f"Target: {self.coordinator.get_status()['target_score']}")
        print("=" * 60)

        # Clear any previous stop signal
        self.coordinator.clear_stop_file()

        self.running = True

        # Start status reporter thread
        status_thread = threading.Thread(target=self._status_reporter, daemon=True)
        status_thread.start()

        try:
            if self.dry_run:
                print("\n[DRY RUN MODE - No containers will be started]\n")
                for worker_id, worker in self.workers.items():
                    prompt = get_worker_prompt(worker_id, 1.1247)
                    print(f"\n{'='*40}")
                    print(f"Worker {worker_id}: {worker.focus_area}")
                    print(f"{'='*40}")
                    print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
                return

            # Run workers in parallel threads
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {
                    executor.submit(self._run_worker_loop, worker): worker_id
                    for worker_id, worker in self.workers.items()
                }

                # Wait for all workers (or until stopped)
                for future in as_completed(futures):
                    worker_id = futures[future]
                    try:
                        future.result()
                    except Exception as e:
                        print(f"[{worker_id}] Worker failed: {e}")

        except KeyboardInterrupt:
            print("\n\nInterrupt received, stopping workers...")
            self.stop()

        finally:
            self.running = False
            print("\n" + format_status_report(self.coordinator))

    def stop(self):
        """Signal all workers to stop."""
        self.stop_requested = True
        self.coordinator.create_stop_file()

        # Kill any running containers
        for worker_id, worker in self.workers.items():
            if worker.process:
                try:
                    subprocess.run(
                        ["docker", "kill", worker.container_name],
                        capture_output=True,
                        timeout=10,
                    )
                except:
                    pass

    def _status_reporter(self):
        """Periodically print status."""
        while self.running:
            time.sleep(60)  # Every minute
            if self.running:
                print("\n" + format_status_report(self.coordinator) + "\n")


class InteractiveOrchestrator(Orchestrator):
    """
    Interactive version that uses separate terminal windows for each worker.
    Better for debugging and watching progress.
    """

    def _start_worker(self, worker: WorkerState) -> bool:
        """Start worker in a new terminal window."""
        if self.dry_run:
            print(f"[DRY RUN] Would start {worker.worker_id}: {worker.focus_area}")
            return True

        prompt = get_worker_prompt(
            worker.worker_id,
            self.coordinator.get_current_best()["score"],
        )

        # Write prompt to file (easier to pass to container)
        prompt_file = self.shared_dir / f"prompt_{worker.worker_id}.txt"
        prompt_file.write_text(prompt)

        # Build command to run in new terminal
        docker_cmd = (
            f'docker run --rm -it '
            f'--name {worker.container_name} '
            f'-v "{self.project_dir}:/workspace" '
            f'-e ANTHROPIC_API_KEY=%ANTHROPIC_API_KEY% '
            f'-e WORKER_ID={worker.worker_id} '
            f'--cpus 8 --memory 32g '
            f'-w /workspace '
            f'{self.image_name} '
            f'bash -c "claude --dangerously-skip-permissions"'
        )

        # On Windows, open new cmd window
        cmd = f'start cmd /k "{docker_cmd}"'

        try:
            subprocess.Popen(cmd, shell=True)
            worker.status = "running"
            worker.started_at = time.time()
            print(f"Started {worker.worker_id} in new terminal")
            return True
        except Exception as e:
            print(f"ERROR starting {worker.worker_id}: {e}")
            worker.status = "error"
            return False

    def run(self):
        """Start workers in separate terminals."""
        print("=" * 60)
        print("INTERACTIVE ORCHESTRATOR")
        print("=" * 60)
        print(f"Starting {self.num_workers} workers in separate terminals...")
        print("Each worker will need the Ralph prompt pasted manually.")
        print("=" * 60)

        # Write prompts to files
        for worker_id, worker in self.workers.items():
            prompt = get_worker_prompt(worker_id, 1.1247)
            prompt_file = self.shared_dir / f"prompt_{worker_id}.txt"
            prompt_file.write_text(prompt)
            print(f"\nPrompt for {worker_id} saved to: {prompt_file}")

        print("\n" + "=" * 60)
        print("TO START EACH WORKER:")
        print("=" * 60)
        for i, (worker_id, worker) in enumerate(self.workers.items()):
            print(f"""
Worker {worker_id} ({worker.focus_area}):
  1. Open new terminal
  2. Run: docker exec -it heat-signature-dev bash
  3. Run: cat /workspace/orchestration/shared/prompt_{worker_id}.txt
  4. Copy the prompt
  5. Run: claude --dangerously-skip-permissions
  6. Paste the prompt
""")


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrate multiple Claude Code workers for heat signature optimization"
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=3,
        help="Number of parallel workers (default: 3)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without starting containers"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Start workers in separate terminal windows"
    )
    parser.add_argument(
        "--project-dir",
        type=str,
        default=None,
        help="Project directory (default: current directory)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="heat-signature-zero:latest",
        help="Docker image name"
    )

    args = parser.parse_args()

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY") and not args.dry_run:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        print("Set it with: $env:ANTHROPIC_API_KEY='your-key'")
        sys.exit(1)

    # Create orchestrator
    if args.interactive:
        orchestrator = InteractiveOrchestrator(
            num_workers=args.workers,
            project_dir=args.project_dir,
            dry_run=args.dry_run,
            image_name=args.image,
        )
    else:
        orchestrator = Orchestrator(
            num_workers=args.workers,
            project_dir=args.project_dir,
            dry_run=args.dry_run,
            image_name=args.image,
        )

    orchestrator.run()


if __name__ == "__main__":
    main()

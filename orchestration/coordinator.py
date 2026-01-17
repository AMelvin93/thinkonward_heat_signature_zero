"""
Shared coordination module for multi-worker Claude Code orchestration.
Uses file-based locking for cross-container coordination.
"""
import json
import hashlib
import time
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from filelock import FileLock, Timeout

# Default paths - can be overridden
DEFAULT_SHARED_DIR = Path("/workspace/orchestration/shared")
COORD_FILE_NAME = "coordination.json"
LOCK_FILE_NAME = "coordination.lock"
RESULTS_LOG_NAME = "all_results.jsonl"


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    threshold_1src: float
    threshold_2src: float
    fallback_fevals: int
    fallback_sigma: float
    refine_iters: int
    refine_top: int
    experiment_name: str
    extra_params: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if d['extra_params'] is None:
            del d['extra_params']
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'ExperimentConfig':
        return cls(**d)


@dataclass
class ExperimentResult:
    """Result from a completed experiment."""
    config: ExperimentConfig
    score: float
    time_min: float
    worker_id: str
    timestamp: float
    rmse_1src: Optional[float] = None
    rmse_2src: Optional[float] = None
    within_budget: bool = True
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d['config'] = self.config.to_dict()
        return d


class Coordinator:
    """
    Coordinates multiple Claude Code workers via shared file state.
    Thread-safe through file locking.
    """

    def __init__(self, shared_dir: Optional[Path] = None):
        self.shared_dir = Path(shared_dir) if shared_dir else DEFAULT_SHARED_DIR
        self.shared_dir.mkdir(parents=True, exist_ok=True)

        self.coord_file = self.shared_dir / COORD_FILE_NAME
        self.lock_file = self.shared_dir / LOCK_FILE_NAME
        self.results_log = self.shared_dir / RESULTS_LOG_NAME

        self._init_state()

    def _init_state(self):
        """Initialize coordination state if not exists."""
        if not self.coord_file.exists():
            initial_state = {
                "best_score": 1.1247,  # Current known best
                "best_config": None,
                "best_worker": None,
                "best_time_min": None,
                "target_score": 1.25,
                "experiments_claimed": {},  # hash -> {worker, timestamp}
                "experiments_completed": [],  # list of hashes
                "workers_registered": {},  # worker_id -> {status, last_seen}
                "created_at": time.time(),
                "updated_at": time.time(),
            }
            self._write_state(initial_state)

    def _read_state(self) -> Dict[str, Any]:
        """Read current state with lock."""
        try:
            with FileLock(self.lock_file, timeout=10):
                if self.coord_file.exists():
                    return json.loads(self.coord_file.read_text())
                return {}
        except Timeout:
            print("WARNING: Could not acquire lock, returning empty state")
            return {}

    def _write_state(self, state: Dict[str, Any]):
        """Write state with lock."""
        state["updated_at"] = time.time()
        try:
            with FileLock(self.lock_file, timeout=10):
                self.coord_file.write_text(json.dumps(state, indent=2))
        except Timeout:
            print("WARNING: Could not acquire lock for writing")

    @staticmethod
    def config_hash(config: ExperimentConfig) -> str:
        """Generate deterministic hash of config."""
        config_str = json.dumps(config.to_dict(), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def register_worker(self, worker_id: str, focus_area: str) -> bool:
        """Register a worker as active."""
        state = self._read_state()
        state["workers_registered"][worker_id] = {
            "focus_area": focus_area,
            "status": "active",
            "registered_at": time.time(),
            "last_seen": time.time(),
            "experiments_run": 0,
        }
        self._write_state(state)
        return True

    def heartbeat(self, worker_id: str):
        """Update worker's last seen timestamp."""
        state = self._read_state()
        if worker_id in state["workers_registered"]:
            state["workers_registered"][worker_id]["last_seen"] = time.time()
            self._write_state(state)

    def claim_experiment(self, worker_id: str, config: ExperimentConfig) -> bool:
        """
        Try to claim an experiment. Returns True if successful.
        Prevents duplicate work across workers.
        """
        config_hash = self.config_hash(config)

        try:
            with FileLock(self.lock_file, timeout=10):
                state = json.loads(self.coord_file.read_text())

                # Check if already completed
                if config_hash in state["experiments_completed"]:
                    return False

                # Check if already claimed (and not stale)
                if config_hash in state["experiments_claimed"]:
                    claimed = state["experiments_claimed"][config_hash]
                    # Consider stale after 2 hours
                    if time.time() - claimed["timestamp"] < 7200:
                        return False

                # Claim it
                state["experiments_claimed"][config_hash] = {
                    "worker": worker_id,
                    "timestamp": time.time(),
                    "config": config.to_dict(),
                }
                state["updated_at"] = time.time()
                self.coord_file.write_text(json.dumps(state, indent=2))
                return True

        except Timeout:
            print(f"WARNING: {worker_id} could not acquire lock to claim experiment")
            return False

    def report_result(self, result: ExperimentResult) -> bool:
        """
        Report experiment result. Returns True if this is a new best.
        """
        config_hash = self.config_hash(result.config)
        is_new_best = False

        try:
            with FileLock(self.lock_file, timeout=10):
                state = json.loads(self.coord_file.read_text())

                # Move from claimed to completed
                if config_hash in state["experiments_claimed"]:
                    del state["experiments_claimed"][config_hash]

                if config_hash not in state["experiments_completed"]:
                    state["experiments_completed"].append(config_hash)

                # Update worker stats
                if result.worker_id in state["workers_registered"]:
                    state["workers_registered"][result.worker_id]["experiments_run"] += 1
                    state["workers_registered"][result.worker_id]["last_seen"] = time.time()

                # Check if new best (must be within budget)
                if result.score > state["best_score"] and result.within_budget:
                    state["best_score"] = result.score
                    state["best_config"] = result.config.to_dict()
                    state["best_worker"] = result.worker_id
                    state["best_time_min"] = result.time_min
                    is_new_best = True

                state["updated_at"] = time.time()
                self.coord_file.write_text(json.dumps(state, indent=2))

        except Timeout:
            print(f"WARNING: Could not acquire lock to report result")

        # Also append to results log (for history)
        self._append_result_log(result)

        return is_new_best

    def _append_result_log(self, result: ExperimentResult):
        """Append result to JSONL log file."""
        try:
            with open(self.results_log, "a") as f:
                f.write(json.dumps(result.to_dict()) + "\n")
        except Exception as e:
            print(f"WARNING: Could not write to results log: {e}")

    def get_current_best(self) -> Dict[str, Any]:
        """Get current best score and config."""
        state = self._read_state()
        return {
            "score": state.get("best_score", 0),
            "config": state.get("best_config"),
            "worker": state.get("best_worker"),
            "time_min": state.get("best_time_min"),
        }

    def get_status(self) -> Dict[str, Any]:
        """Get full coordinator status."""
        state = self._read_state()
        return {
            "best_score": state.get("best_score", 0),
            "target_score": state.get("target_score", 1.25),
            "experiments_completed": len(state.get("experiments_completed", [])),
            "experiments_in_progress": len(state.get("experiments_claimed", {})),
            "workers": state.get("workers_registered", {}),
            "updated_at": state.get("updated_at"),
        }

    def is_target_reached(self) -> bool:
        """Check if target score has been reached."""
        state = self._read_state()
        return state.get("best_score", 0) >= state.get("target_score", 1.25)

    def get_completed_hashes(self) -> List[str]:
        """Get list of completed experiment hashes."""
        state = self._read_state()
        return state.get("experiments_completed", [])

    def get_all_results(self) -> List[Dict[str, Any]]:
        """Read all results from the log file."""
        results = []
        if self.results_log.exists():
            with open(self.results_log, "r") as f:
                for line in f:
                    try:
                        results.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        continue
        return results

    def create_stop_file(self):
        """Create stop file to signal all workers to stop."""
        (self.shared_dir / "STOP").touch()

    def should_stop(self) -> bool:
        """Check if stop signal exists."""
        return (self.shared_dir / "STOP").exists()

    def clear_stop_file(self):
        """Remove stop file."""
        stop_file = self.shared_dir / "STOP"
        if stop_file.exists():
            stop_file.unlink()


def format_status_report(coordinator: Coordinator) -> str:
    """Generate human-readable status report."""
    status = coordinator.get_status()
    best = coordinator.get_current_best()

    lines = [
        "=" * 60,
        "ORCHESTRATION STATUS",
        "=" * 60,
        f"Best Score: {status['best_score']:.4f} (target: {status['target_score']})",
        f"Best Worker: {best.get('worker', 'N/A')}",
        f"Best Time: {best.get('time_min', 'N/A')} min",
        f"Experiments Completed: {status['experiments_completed']}",
        f"Experiments In Progress: {status['experiments_in_progress']}",
        "",
        "Workers:",
    ]

    for worker_id, info in status.get("workers", {}).items():
        last_seen = info.get("last_seen", 0)
        ago = time.time() - last_seen if last_seen else float('inf')
        status_str = "active" if ago < 300 else "stale"
        lines.append(f"  {worker_id}: {info.get('focus_area', 'unknown')} "
                    f"({info.get('experiments_run', 0)} runs, {status_str})")

    lines.append("=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    # Test the coordinator
    coord = Coordinator()
    print(format_status_report(coord))

#!/usr/bin/env python3
"""
Quick status check - run from any worker or host.
Usage: python /workspace/orchestration/shared/check_status.py
"""
import json
from pathlib import Path

SHARED_DIR = Path(__file__).parent
COORD_FILE = SHARED_DIR / "coordination.json"
QUEUE_FILE = SHARED_DIR / "experiment_queue.json"

def main():
    print("\n" + "=" * 60)
    print("        ORCHESTRATION STATUS")
    print("=" * 60)

    # Coordination status
    if COORD_FILE.exists():
        data = json.loads(COORD_FILE.read_text())
        best = data.get("best_score", 0)
        target = data.get("target_score", 1.25)
        gap = target - best

        print(f"\nBEST SCORE:  {best:.4f} @ {data.get('best_time_min', '?')} min")
        print(f"TARGET:      {target:.4f}")
        print(f"GAP:         {gap:.4f} ({gap/target*100:.1f}%)")
    else:
        print("\nNo coordination.json found!")

    # Experiment queue status
    if QUEUE_FILE.exists():
        queue = json.loads(QUEUE_FILE.read_text())

        ready = queue.get("ready_experiments", [])
        completed = queue.get("completed_experiments", [])

        available = [e for e in ready if e.get("status") == "available"]
        running = [e for e in ready if e.get("status") == "running"]

        print("\n" + "-" * 60)
        print("EXPERIMENT QUEUE")
        print("-" * 60)
        print(f"  Available:  {len(available)}")
        print(f"  Running:    {len(running)}")
        print(f"  Completed:  {len(completed)}")

        if running:
            print("\n  Currently Running:")
            for e in running:
                print(f"    - {e.get('name', '?')} (by {e.get('assigned_to', '?')})")

        if available:
            print("\n  Next Up:")
            for e in sorted(available, key=lambda x: x.get("priority", 99))[:3]:
                print(f"    P{e.get('priority', '?')}: {e.get('name', '?')}")
                if e.get("hypothesis"):
                    print(f"        {e.get('hypothesis')[:50]}...")

        # Analysis
        analysis = queue.get("analysis", {})
        if analysis:
            print("\n" + "-" * 60)
            print("ANALYSIS")
            print("-" * 60)

            promising = analysis.get("promising_directions", [])
            blocked = analysis.get("blocked_directions", [])

            if promising:
                print("  Promising:")
                for p in promising[-2:]:
                    print(f"    + {p[:55]}...")

            if blocked:
                print("  Blocked:")
                for b in blocked[-2:]:
                    print(f"    - {b[:55]}...")
    else:
        print("\nNo experiment_queue.json found!")

    # Workers
    if COORD_FILE.exists():
        data = json.loads(COORD_FILE.read_text())
        workers = data.get("workers", {})

        print("\n" + "-" * 60)
        print("WORKERS")
        print("-" * 60)

        for wid in ["W1", "W2"]:
            w = workers.get(wid, {})
            status = w.get("status", "unknown")
            runs = w.get("experiments_run", 0)
            status_icon = "[R]" if status == "running" else "[ ]"
            print(f"  {status_icon} {wid}: {status:8} | {runs} experiments completed")

    print("\n" + "=" * 60)
    print()

if __name__ == "__main__":
    main()

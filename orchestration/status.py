#!/usr/bin/env python3
"""
Quick status checker for orchestration system.
Usage: python orchestration/status.py
"""
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def main():
    shared_dir = Path(__file__).parent / "shared"
    coord_file = shared_dir / "coordination.json"
    results_log = shared_dir / "all_results.jsonl"

    print("=" * 60)
    print("ORCHESTRATION STATUS")
    print("=" * 60)
    print()

    if not coord_file.exists():
        print("No coordination file found.")
        print(f"Expected at: {coord_file}")
        print()
        print("Run this first to initialize:")
        print("  python orchestration/orchestrator.py --dry-run")
        return

    # Load coordination state
    data = json.loads(coord_file.read_text())

    # Best score
    print(f"BEST SCORE: {data.get('best_score', 'N/A'):.4f}")
    print(f"Target:     {data.get('target_score', 1.25):.4f}")
    print(f"Gap:        {data.get('target_score', 1.25) - data.get('best_score', 0):.4f}")
    print()

    if data.get('best_config'):
        print("Best Config:")
        for k, v in data['best_config'].items():
            if k != 'extra_params':
                print(f"  {k}: {v}")
    print()

    # Progress
    completed = len(data.get('experiments_completed', []))
    in_progress = len(data.get('experiments_claimed', {}))
    print(f"EXPERIMENTS")
    print(f"  Completed:   {completed}")
    print(f"  In Progress: {in_progress}")
    print()

    # Workers
    workers = data.get('workers_registered', {})
    if workers:
        print("WORKERS")
        for wid, info in workers.items():
            last_seen = info.get('last_seen', 0)
            if last_seen:
                ago = datetime.now().timestamp() - last_seen
                if ago < 60:
                    status = f"active ({int(ago)}s ago)"
                elif ago < 300:
                    status = f"active ({int(ago/60)}m ago)"
                else:
                    status = f"stale ({int(ago/60)}m ago)"
            else:
                status = "unknown"

            print(f"  {wid}: {info.get('focus_area', 'unknown')}")
            print(f"       Runs: {info.get('experiments_run', 0)}, Status: {status}")
        print()

    # Recent results
    if results_log.exists():
        results = []
        with open(results_log) as f:
            for line in f:
                try:
                    results.append(json.loads(line.strip()))
                except:
                    pass

        if results:
            print("RECENT RESULTS (last 5)")
            for r in results[-5:]:
                score = r.get('score', 0)
                time_min = r.get('time_min', 0)
                worker = r.get('worker_id', '?')
                budget_status = "✓" if r.get('within_budget', True) else "✗"
                print(f"  [{worker}] {score:.4f} @ {time_min:.1f}min {budget_status}")
            print()

            # Best results overall
            valid_results = [r for r in results if r.get('within_budget', True)]
            if valid_results:
                best = max(valid_results, key=lambda r: r.get('score', 0))
                print("BEST RESULT (in budget)")
                print(f"  Score: {best.get('score', 0):.4f}")
                print(f"  Time:  {best.get('time_min', 0):.1f} min")
                print(f"  Worker: {best.get('worker_id', '?')}")

    print()
    print("=" * 60)

    # Check for stop file
    stop_file = shared_dir / "STOP"
    if stop_file.exists():
        print("⚠️  STOP FILE EXISTS - Workers will stop at next checkpoint")
        print(f"   Remove with: rm {stop_file}")
        print("=" * 60)


if __name__ == "__main__":
    main()

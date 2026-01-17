#!/usr/bin/env python3
"""
Quick status check - run from any worker or host.
Usage: python /workspace/orchestration/shared/check_status.py
"""
import json
from pathlib import Path
from datetime import datetime

SHARED_DIR = Path(__file__).parent
COORD_FILE = SHARED_DIR / "coordination.json"
QUEUE_FILE = SHARED_DIR / "queue.json"

def main():
    print("\n" + "=" * 60)
    print("        ORCHESTRATION STATUS")
    print("=" * 60)

    if not COORD_FILE.exists():
        print("No coordination.json found!")
        return

    data = json.loads(COORD_FILE.read_text())

    # Best score
    best = data.get("best_score", 0)
    target = data.get("target_score", 1.25)
    gap = target - best

    print(f"\nBEST SCORE:  {best:.4f}")
    print(f"TARGET:      {target:.4f}")
    print(f"GAP:         {gap:.4f} ({gap/target*100:.1f}%)")

    if data.get("best_config"):
        print(f"\nBest config: {data.get('best_config', {}).get('experiment', 'unknown')}")

    # Workers
    print("\n" + "-" * 60)
    print("AGENTS")
    print("-" * 60)

    # Check W0 state
    print("  [W0] Research Orchestrator - monitors & prioritizes queue")

    workers = data.get("workers", {})
    for wid in ["W1", "W2", "W3"]:
        w = workers.get(wid, {})
        status = w.get("status", "unknown")
        focus = w.get("focus", "unknown")
        runs = w.get("experiments_run", 0)
        current = w.get("current_experiment", "-")

        status_icon = "[R]" if status == "running" else "[ ]"
        print(f"  {status_icon} {wid}: {status:8} | {runs} runs | Focus: {focus}")

        # Check state file for current topic
        state_file = SHARED_DIR / f"state_{wid}.json"
        if state_file.exists():
            state = json.loads(state_file.read_text())
            topic = state.get("current_topic", {}).get("name")
            topic_status = state.get("topic_status", "idle")
            if topic:
                print(f"       Topic: {topic} ({topic_status})")

    # Queue status
    if QUEUE_FILE.exists():
        queue = json.loads(QUEUE_FILE.read_text())
        topics = queue.get("research_topics", [])
        available = [t for t in topics if t.get("status") == "available"]
        assigned = [t for t in topics if t.get("status") == "assigned"]

        print("\n" + "-" * 60)
        print("RESEARCH QUEUE")
        print("-" * 60)
        print(f"  Available topics: {len(available)}")
        print(f"  Assigned topics:  {len(assigned)}")

        if available:
            print("\n  Top priorities:")
            for t in sorted(available, key=lambda x: x.get("priority", 99))[:3]:
                print(f"    P{t.get('priority', '?')}: {t.get('name', 'unknown')}")

        # Research insights
        insights = queue.get("research_insights", {})
        works = insights.get("what_works", [])
        doesnt = insights.get("what_doesnt_work", [])

        if works or doesnt:
            print("\n  Research Insights:")
            if works:
                print(f"    Works: {len(works)} findings")
            if doesnt:
                print(f"    Doesn't work: {len(doesnt)} findings")

    # In progress experiments
    in_progress = data.get("experiments_in_progress", {})
    if in_progress:
        print("\n" + "-" * 60)
        print("IN PROGRESS EXPERIMENTS")
        print("-" * 60)
        for exp, info in in_progress.items():
            print(f"  - {exp} (by {info.get('worker', '?')})")

    # Completed count
    completed = data.get("experiments_completed", [])
    print("\n" + "-" * 60)
    print(f"COMPLETED EXPERIMENTS: {len(completed)}")
    if completed:
        # Handle both list of strings and list of dicts
        recent = []
        for c in completed[-5:]:
            if isinstance(c, dict):
                recent.append(c.get("name", "unknown"))
            else:
                recent.append(str(c))
        print("  Recent: " + ", ".join(recent))

    # Research findings from coordination
    findings = data.get("research_findings", [])
    if findings:
        print("\n" + "-" * 60)
        print("COORDINATION FINDINGS")
        print("-" * 60)
        for f in findings[-3:]:
            print(f"  - {f}")

    print("\n" + "=" * 60)
    print()

if __name__ == "__main__":
    main()

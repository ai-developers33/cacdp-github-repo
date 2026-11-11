"""One-command cross-platform host runner (Windows PowerShell/CMD, macOS, Linux).

Uses ONLY Python 3 standard library on the host machine (all Kafka/NumPy/SciPy packages
run inside Docker containers).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import zipfile

SCENARIOS = [
    "baseline_normal",
    "benign_schema_additive",
    "benign_routing_dual_sub",
    "benign_minor_slowdown",
    "harmful_consumer_slowdown",
    "harmful_partition_hotspot",
    "harmful_dependency_outage",
    "harmful_routing_unsubscribed",
    "harmful_schema_breaking",
]


def run_cmd(cmd: list[str], cwd: str):
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    ap = argparse.ArgumentParser(description="Run Live Study 2 on Dockerized Apache Kafka 3.9.0 + HTTP Gateway")
    ap.add_argument("--reps", type=int, default=5, help="Runs per scenario (default 5 => 45 live runs, ~25 min; use 10 for full 90 runs)")
    ap.add_argument("--duration", type=float, default=30.0, help="Seconds per run (default 30.0; use 10.0 for quick smoke test)")
    ap.add_argument("--seed", type=int, default=20261110)
    args = ap.parse_args()

    if args.reps < 5:
        print("NOTE: Using --reps >= 5 is recommended so 5-fold run-grouped CV has at least 1 run per scenario per fold.")

    here = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(here, "data")
    if os.path.exists(data_dir):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup = os.path.join(here, f"data_backup_{stamp}")
        shutil.move(data_dir, backup)
        print(f"Archived existing data/ -> {backup}")
    os.makedirs(data_dir, exist_ok=True)

    print("\n[1/4] Starting Apache Kafka 3.9.0, HTTP Payment Gateway, 3 Inventory replicas, Shipping, Billing, Fulfillment & 1 Hz Monitor...")
    run_cmd(["docker", "compose", "down", "-v"], cwd=here)
    run_cmd(["docker", "compose", "up", "-d", "--build", "--wait"], cwd=here)
    time.sleep(5.0)  # Let consumers register group assignments

    total_runs = len(SCENARIOS) * args.reps
    done = 0
    print(f"\n[2/4] Executing {total_runs} live runs ({len(SCENARIOS)} scenarios x {args.reps} runs)...")
    for s_idx, sc in enumerate(SCENARIOS):
        for r_idx in range(args.reps):
            done += 1
            seed = args.seed + s_idx * 100 + r_idx
            print(f"\n--- Live Run {done}/{total_runs}: {sc} (rep {r_idx + 1}/{args.reps}) ---", flush=True)
            run_cmd([
                "docker", "compose", "run", "--rm", "--no-deps", "-T", "runner",
                "python", "producer_live.py",
                "--scenario", sc,
                "--rep", str(r_idx),
                "--duration", str(args.duration),
                "--seed", str(seed),
            ], cwd=here)

    print("\n[3/4] Building leak-free 1 Hz causal windows and running 5-fold CV + LOSO evaluation...")
    run_cmd([
        "docker", "compose", "run", "--rm", "--no-deps", "-T", "runner",
        "python", "build_and_evaluate_live.py", "--duration", str(args.duration),
    ], cwd=here)

    zip_path = os.path.join(here, "live_study2_results.zip")
    print(f"\n[4/4] Packaging results into {zip_path} and stopping containers...")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fn in sorted(os.listdir(data_dir)):
            full = os.path.join(data_dir, fn)
            if os.path.isfile(full):
                zf.write(full, os.path.join("live_study2_results", fn))

    run_cmd(["docker", "compose", "down", "-v"], cwd=here)
    print(f"\nCOMPLETED! Live Kafka + HTTP results saved to: {zip_path}")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)

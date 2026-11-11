"""Rebuild the Study 2 analysis from archived logs and check Study 1 summary data."""
import csv
import filecmp
import math
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STUDY2 = ROOT / "study2_live_kafka"
ARCHIVED = STUDY2 / "data"


def check_study1():
    with (ROOT / "study1_summary" / "run_latency.csv").open(newline="") as source:
        rows = list(csv.DictReader(source))
    groups = {}
    for row in rows:
        key = (row["protocol"], row["scenario"])
        group = groups.setdefault(key, {})
        run = int(row["run_number"])
        if run in group:
            raise ValueError(f"Duplicate run {run} for {key}")
        group[run] = float(row["mean_latency_s"])
    protocols = {"Kafka Pilot", "No-Drain", "Drained"}
    expected_groups = {(p, s) for p in protocols for s in ("baseline", "slowdown")}
    if set(groups) != expected_groups or any(set(g) != set(range(1, 6)) for g in groups.values()):
        raise ValueError("Expected five runs per protocol and scenario in the Study 1 CSV")
    observed = [groups[("No-Drain", "slowdown")][i] for i in range(1, 6)]
    predicted = [7.675 + 13.1 * k for k in range(5)]
    residuals = [a - b for a, b in zip(observed, predicted)]
    r_squared = 1 - sum(x * x for x in residuals) / sum((x - statistics.mean(observed)) ** 2 for x in observed)
    if not math.isclose(r_squared, 0.9992, abs_tol=0.00005):
        raise ValueError(f"Study 1 queue model R² changed to {r_squared:.6f}")
    print(f"Study 1: {len(rows)} tabulated rows, no-drain slowdown R² = {r_squared:.4f}", flush=True)


def main():
    check_study1()
    subprocess.run([sys.executable, str(STUDY2 / "verify_live_study2.py")], check=True)
    with tempfile.TemporaryDirectory(prefix="cacdp-replay-") as temporary:
        subprocess.run([sys.executable, str(STUDY2 / "build_and_evaluate_live.py"),
                        "--data-dir", str(ARCHIVED), "--out-dir", temporary], check=True)
        for name in ("study2_windows.csv", "study2_runs_summary.json", "study2_eval_results.json"):
            if not filecmp.cmp(ARCHIVED / name, Path(temporary) / name, shallow=False):
                raise ValueError(f"Rebuilt file differs from the archive: {name}")
            print(f"Exact match: {name}")


if __name__ == "__main__":
    main()

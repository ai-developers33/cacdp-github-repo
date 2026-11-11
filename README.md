# CACDP Kafka replication package

This repository contains run-level latency data from Study 1 and the Study 2 benchmark for Change-Aware Cascading Degradation Prediction (CACDP). A synthetic order-processing workload exercises a three-partition Apache Kafka 3.9.0 broker and an HTTP payment service. There are nine scenarios, each with five recorded runs. Source files for the Study 1 pilot and its wall-clock reimplementation are outside the scope of this package.

## Reproduce the published analysis

From a fresh checkout, run the following on macOS or Linux. The offline replay was tested with CPython 3.12.14. It takes the archived logs as input and writes rebuilt outputs to a temporary directory. It also checks the Study 1 queue model against the tabulated slowdown runs.

```sh
git clone --branch v1.0.1 https://github.com/ai-developers33/cacdp-github-repo.git
cd cacdp-github-repo
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-replay.txt
python reproduce.py
```

A successful replay reports **R² = 0.9992** for the Study 1 no-drain slowdown series, checks the 15 archived SHA-256 digests, and prints an exact match for each of the three Study 2 output files. The experiment uses synthetic order events. All timestamps in the stage logs are Unix seconds. Use the combination of `run_id` and `event_id` to join the producer and processing-stage records. The monitor records samples and occasional polling errors, so its rows do not map one-to-one to orders.

## Dataset inventory

| File | Records | Contents |
|---|---:|---|
| `study1_summary/run_latency.csv` | 30 | Run-level mean and p95 latency in seconds for baseline and slowdown across three protocols. Values are transcribed at the precision of the reported table. |
| `produced.jsonl` | 13,854 | Offered orders, creation and deployment times, run, scenario and partition |
| `inventory.jsonl` | 13,854 | Inventory completion, replica, reservation status and target topic |
| `shipping.jsonl` | 12,038 | Shipping completion and source topic |
| `billing.jsonl` | 12,963 | Billing outcome, completion time and retries |
| `fulfillment.jsonl` | 11,560 | End-to-end join completion |
| `monitor_1hz.jsonl` | 1,756 | Broker and health-monitor samples, including polling errors |
| `study2_windows.csv` | 1,035 | Observation windows, features and target labels; 366 positive windows |
| `study2_runs_summary.json` | 45 runs | Per-run measurements, scenario aggregates and dataset configuration |
| `study2_eval_results.json` | 2 classifiers × 3 feature sets | Grouped cross-validation and held-out-scenario results |

Study 2 files in the table are under `study2_live_kafka/data/`. The replay reanalyzes recorded logs. To generate new observations, use the Docker workflow below. Timing and model scores on fresh traces can differ from the archive.

## Verify the archived files

From the repository root, use Python 3.10 or later. The verifier checks 15 SHA-256 digests and selected values in the stored result files.

```sh
python3 study2_live_kafka/verify_live_study2.py
```

## Reproduce the analysis from the recorded logs

The six JSONL files in `study2_live_kafka/data/` contain the synthetic order events, stage completions and monitor samples. The same folder holds the derived windows, run summaries and reported evaluation results. To rebuild all three outputs in a separate directory:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-replay.txt
python study2_live_kafka/build_and_evaluate_live.py --data-dir study2_live_kafka/data --out-dir recomputed
```

The archive contains 13,854 offered orders and 1,035 observation windows, including 366 positive windows. Under five-fold validation grouped by run, the random forest scored 0.9484 F1 with telemetry and 0.9738 with the additional compatibility and capacity features. Its 0/90 false alerts were measured in selected rollout windows across 15 benign runs. Some feature calculations use the benchmark's scenario or deployment configuration. The same information would need to be available to an operational monitor. The mean warning lead of 2.04 s uses an estimated breach clock defined from arrival cohorts and the two-second objective.

To write three diagnostic SVG plots to `study2_live_kafka/generated_figures/`, run:

```sh
python study2_live_kafka/generate_study2_figures.py
```

## Collect new traces

The live runner requires Docker with Compose v2. It moves an existing data directory to a timestamped backup before starting. A full run schedules five repetitions of each scenario. Each run offers events for 30 seconds and then waits for the broker drain check, subject to a 45-second timeout. Wall-clock results depend on the host.

```sh
python3 study2_live_kafka/run_live_study2.py --reps 5 --duration 30
```

The Compose file configures the broker and services. The producer schedules the workload, the monitor records broker and health observations, and the model evaluator performs run-grouped and held-out-scenario comparisons. The verifier is specific to the archived run and should be used before collecting new traces.

## License and citation

GitHub release `v1.0.1` includes the recorded Study 2 analysis and the Study 1 run summaries. Cite the [Zenodo archive for this release](https://doi.org/10.5281/zenodo.24147806). The software uses the MIT license. Author and citation details are in `CITATION.cff`.

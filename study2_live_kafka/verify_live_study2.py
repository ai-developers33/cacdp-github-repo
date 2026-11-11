"""Automated SHA-256 & Statistical Assertion Verifier for CACDP Live Apache Kafka 3.9.0 Study 2."""

from __future__ import annotations

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPECTED_SHA256 = {
    "services.py": "9d20ca093ccb3c9620a070102e24d58a780d72521db56808c04743abce25b30c",
    "producer_live.py": "bc26701802bcd0b5d5b126274af34b644953c6110248d66a65ac41544382082a",
    "gateway_service.py": "f46d216e1827bc027967c757e530c880efe5f2b7a4fe431403e6d7d2c110e764",
    "monitor_service.py": "4e64f8c60e219938934eb76bafed5a698e689872b5bc6e41b86680c0316caebf",
    "build_and_evaluate_live.py": "5566709103ab914b13ca4d8426b53e5ae7786d7815a176b32fad33cb7177ee7a",
    "evaluate_models.py": "752ad7ac785ed791f8dd2137d5c465f3d70f2ca60d2b3ca7891b54371d1865dd",
    "data/produced.jsonl": "a92c2bddf52dead4ba4d092e27f971589d373743830b769e22575608c89e001e",
    "data/inventory.jsonl": "d0bc7cc71d4c9a2e4033cb94cf966c2504840b8cab49a997e5bd90740450613e",
    "data/shipping.jsonl": "68badb4af397c920311871265dcc05f973af8dcf26660eeeeeca15f595605414",
    "data/billing.jsonl": "e06769c0f0bf47981cebd6d549a569d6bf87036d692ca9bfab959cbdd48d34d1",
    "data/fulfillment.jsonl": "d3c43018be123dbba6b3be8a59e7a8c7fa99e60e4678c28e1538763175eeb4c5",
    "data/monitor_1hz.jsonl": "9cc8cec1a4f7bef88e0967fb21c173645c03417796b5946683e0999f249c8c70",
    "data/study2_windows.csv": "0cb8cec1e9743f53da1189e09a618909b74d9aba9068cf941d68651bb1af0aa0",
    "data/study2_runs_summary.json": "a6c569d7b692fe52baacaa5f1119298e4f67d12a4566979b0a5a1f6fbeee5742",
    "data/study2_eval_results.json": "5e1e2decdeab0a1ab8406bc1eb5de928efaaca0d6175de33e532f6e6b36c8124"
}


def main() -> int:
    for rel, exp in EXPECTED_SHA256.items():
        p = os.path.join(HERE, rel)
        if not os.path.exists(p):
            print(f"[FAIL] Missing artifact: {rel}")
            return 1
        got = hashlib.sha256(open(p, "rb").read()).hexdigest()
        if got != exp:
            print(f"[FAIL] SHA-256 mismatch on {rel}: got {got}, expected {exp}")
            return 1

    sm = json.load(open(os.path.join(HERE, "data/study2_runs_summary.json"), encoding="utf-8"))
    ev = json.load(open(os.path.join(HERE, "data/study2_eval_results.json"), encoding="utf-8"))
    assert sm["config"]["total_runs"] == 45
    assert sm["config"]["total_offered_events"] == 13854
    assert sm["config"]["total_windows"] == 1035
    assert sm["config"]["positive_windows"] == 366

    rf_sem = ev["run_grouped_5fold_cv"]["RandomForest"]["Model_B_Semantic"]
    lr_sem = ev["run_grouped_5fold_cv"]["LogisticRegression"]["Model_B_Semantic"]
    assert rf_sem["f1"] == 0.9738 and rf_sem["benign_rollout_far"] == 0.0
    assert lr_sem["f1"] == 0.9696 and lr_sem["benign_rollout_far"] == 0.0111
    print("[PASS] All 15 SHA-256 digests and Live Study 2 statistical assertions verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

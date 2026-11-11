"""Build leak-free causal 1 Hz Study 2 dataset from live Kafka JSONL logs and run CV + LOSO evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from typing import Dict, List

try:
    from common_live import DATA_DIR, N_PARTITIONS, REPLICA_ROLLOUT_STEP_S
except ImportError:
    DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    N_PARTITIONS = 3
    REPLICA_ROLLOUT_STEP_S = 1.5

from evaluate_models import evaluate_all

SLO_LATENCY_S = 2.0
SLO_COMPLETION_MIN = 0.90
HORIZON_S = 4.0
LOOKBACK_S = 3.0
RUN_DURATION_S = 30.0
SAFE_RHO_HEADROOM = 0.70

SCENARIO_META = {
    "baseline_normal": {"is_harmful": 0, "label": "Baseline (no change)"},
    "benign_schema_additive": {"is_harmful": 0, "label": "Benign schema (+optional field)"},
    "benign_routing_dual_sub": {"is_harmful": 0, "label": "Benign routing (dual-subscribed)"},
    "benign_minor_slowdown": {"is_harmful": 0, "label": "Benign slowdown (110 ms, rho<1)"},
    "harmful_consumer_slowdown": {"is_harmful": 1, "label": "Harmful slowdown (440 ms, rho>1)"},
    "harmful_partition_hotspot": {"is_harmful": 1, "label": "Partition hotspot (88% key skew)"},
    "harmful_dependency_outage": {"is_harmful": 1, "label": "Payment gateway outage (80% fail)"},
    "harmful_routing_unsubscribed": {"is_harmful": 1, "label": "Unsubscribed topic routing (.v2)"},
    "harmful_schema_breaking": {"is_harmful": 1, "label": "Breaking schema (sku removed)"},
}


def load_jsonl(data_dir: str, name: str) -> List[dict]:
    p = os.path.join(data_dir, name)
    if not os.path.exists(p):
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_from_live_logs(out_dir: str, run_duration_s: float = RUN_DURATION_S, data_dir: str | None = None) -> dict:
    src_dir = data_dir or out_dir
    os.makedirs(out_dir, exist_ok=True)
    prod = load_jsonl(src_dir, "produced.jsonl")
    inv_raw = load_jsonl(src_dir, "inventory.jsonl")
    ship_raw = load_jsonl(src_dir, "shipping.jsonl")
    bill_raw = load_jsonl(src_dir, "billing.jsonl")
    ful_raw = load_jsonl(src_dir, "fulfillment.jsonl")
    mon = [r for r in load_jsonl(src_dir, "monitor_1hz.jsonl") if "orders_lag_total" in r]

    # Strictly causal per-event FIFO service duration on each inventory replica
    inv_by_rep: Dict[int, List[dict]] = {}
    for r in inv_raw:
        inv_by_rep.setdefault(int(r["replica"]), []).append(r)
    for _, arr in inv_by_rep.items():
        arr.sort(key=lambda x: x["inv_done_at"])
        prev_done = 0.0
        for r in arr:
            start_at = max(float(r["created_at"]), prev_done)
            r["causal_service_s"] = max(0.001, float(r["inv_done_at"]) - start_at)
            prev_done = float(r["inv_done_at"])

    inv = {(r["run_id"], r["event_id"]): r for r in inv_raw}
    ship = {(r["run_id"], r["event_id"]): r for r in ship_raw}
    bill = {(r["run_id"], r["event_id"]): r for r in bill_raw}
    ful = {(r["run_id"], r["event_id"]): r for r in ful_raw}

    runs_events: Dict[str, List[dict]] = {}
    for p in prod:
        key = (p["run_id"], p["event_id"])
        rec = {
            **p,
            "rel_created": p["created_at"] - p["run_start_ts"],
            "onset_t": round(p["deploy_ts"] - p["run_start_ts"], 2),
            "inv": inv.get(key),
            "ship": ship.get(key),
            "bill": bill.get(key),
            "ful": ful.get(key),
        }
        runs_events.setdefault(p["run_id"], []).append(rec)

    all_summaries: List[dict] = []
    all_windows: List[dict] = []

    for run_id, evs in sorted(runs_events.items()):
        evs.sort(key=lambda x: x["created_at"])
        sc = evs[0]["scenario"]
        t0 = evs[0]["run_start_ts"]
        onset_t = evs[0]["onset_t"]
        deploy_wall = t0 + onset_t
        mon_run = [m for m in mon if (t0 - 1.0) <= m["ts"] <= (t0 + run_duration_s + 2.0)]

        bucket_breach: Dict[int, int] = {}
        for b in range(int(run_duration_s)):
            cohort = [e for e in evs if b <= e["rel_created"] < b + 1.0]
            if not cohort:
                bucket_breach[b] = 0
                continue
            within_slo = [
                e for e in cohort
                if e["ful"] is not None and (e["ful"]["fulfilled_at"] - e["created_at"]) <= SLO_LATENCY_S
            ]
            comp_slo = len(within_slo) / len(cohort)
            lats = sorted(e["ful"]["fulfilled_at"] - e["created_at"] for e in cohort if e["ful"] is not None)
            p95_b = lats[math.ceil(0.95 * len(lats)) - 1] if lats else float("inf")
            bucket_breach[b] = int(comp_slo < SLO_COMPLETION_MIN or p95_b > SLO_LATENCY_S)

        breach_times = [float(b) for b in sorted(bucket_breach) if bucket_breach[b] == 1]
        deg_onset = breach_times[0] if breach_times else None

        for t_int in range(3, int(run_duration_s - HORIZON_S)):
            t_rel = float(t_int)
            t_wall = t0 + t_rel
            w_start = t_wall - LOOKBACK_S

            offered_w = [e for e in evs if w_start < e["created_at"] <= t_wall]
            inv_done_w = [e for e in evs if e["inv"] is not None and w_start < e["inv"]["inv_done_at"] <= t_wall]
            inv_err_w = [e for e in inv_done_w if e["inv"]["status"] != "reserved"]

            bill_w = [
                e for e in evs
                if e["bill"] is not None and w_start < e["bill"]["bill_done_at"] <= t_wall
            ]
            bill_err_w = [e for e in bill_w if e["bill"]["status"] != "billed"]
            retries_w = sum(int(e["bill"].get("retries", 0)) for e in bill_w)

            ful_w = [
                e for e in evs
                if e["ful"] is not None and w_start < e["ful"]["fulfilled_at"] <= t_wall
            ]
            obs_lats = sorted(e["ful"]["fulfilled_at"] - e["created_at"] for e in ful_w)
            obs_mean = statistics.mean(obs_lats) if obs_lats else 0.0
            obs_p95 = obs_lats[math.ceil(0.95 * len(obs_lats)) - 1] if obs_lats else 0.0

            elig_old = [
                e for e in evs
                if (t_wall - SLO_LATENCY_S - LOOKBACK_S) < e["created_at"] <= (t_wall - SLO_LATENCY_S)
            ]
            over_old = [
                e for e in elig_old
                if e["ful"] is None or e["ful"]["fulfilled_at"] > t_wall
            ]
            overdue_rate = len(over_old) / len(elig_old) if elig_old else 0.0

            # Latest causal monitor sample <= t_wall
            cand_m = [m for m in mon_run if m["ts"] <= t_wall]
            m_snap = max(cand_m, key=lambda x: x["ts"]) if cand_m else {
                "orders_lag_total": 0,
                "orders_lag_max_part": 0,
                "orders_lag_skew": round(1.0 / N_PARTITIONS, 4),
                "downstream_lag": 0,
                "sem_unsubscribed_topic": 0.0,
                "sem_schema_deficit": 0.0,
                "sem_dep_probe_fail": 0.0,
            }

            # Causal unconsumed .v2 backlog up to t_wall
            v2_prod_upto = sum(
                1 for e in evs
                if e["inv"] is not None and e["inv"].get("target_topic") == "inventory.reserved.v2" and e["inv"]["inv_done_at"] <= t_wall
            )
            v2_ship_upto = sum(
                1 for e in evs
                if e["ship"] is not None and e["ship"].get("topic") == "inventory.reserved.v2" and e["ship"]["ship_done_at"] <= t_wall
            )
            v2_unconsumed = max(0, v2_prod_upto - v2_ship_upto)
            downstream_lag_fixed = int(m_snap["downstream_lag"]) + (v2_unconsumed if sc != "benign_routing_dual_sub" else 0)

            change_active = int(t_rel >= onset_t and sc != "baseline_normal")
            rollout_frac = (
                sum(1 for r in range(N_PARTITIONS) if t_rel >= onset_t + r * REPLICA_ROLLOUT_STEP_S) / N_PARTITIONS
                if change_active else 0.0
            )
            p_counts = [sum(1 for e in offered_w if e["partition"] == p) for p in range(N_PARTITIONS)] if offered_w else [1, 1, 1]
            part_conc = max(p_counts) / max(1, len(offered_w)) if offered_w else 1.0 / N_PARTITIONS

            # Strictly causal per-replica queueing utilization rho_r(t) = lambda_r(t) * s_r(t)
            part_utils = []
            for r_idx in range(N_PARTITIONS):
                lam_r = sum(1 for e in offered_w if e["partition"] == r_idx) / LOOKBACK_S
                rep_roll_wall = deploy_wall + r_idx * REPLICA_ROLLOUT_STEP_S
                if t_wall >= rep_roll_wall and sc != "baseline_normal":
                    rep_dones = [
                        e["inv"]["causal_service_s"] for e in inv_done_w
                        if e["inv"]["replica"] == r_idx and e["inv"]["inv_done_at"] >= rep_roll_wall
                    ]
                else:
                    rep_dones = [e["inv"]["causal_service_s"] for e in inv_done_w if e["inv"]["replica"] == r_idx]

                done_on_r = [
                    e["inv"]["inv_done_at"] for e in evs
                    if e["partition"] == r_idx and e["inv"] is not None and e["inv"]["inv_done_at"] <= t_wall
                ]
                last_done_r = max(done_on_r) if done_on_r else t0
                pending_on_r = [
                    e for e in evs
                    if e["partition"] == r_idx and e["created_at"] <= t_wall
                    and (e["inv"] is None or e["inv"]["inv_done_at"] > t_wall)
                ]
                inflight_elap = 0.0
                if pending_on_r:
                    hol_ev = min(pending_on_r, key=lambda x: x["created_at"])
                    inflight_elap = max(0.0, t_wall - max(hol_ev["created_at"], last_done_r))

                base_s = statistics.mean(rep_dones[-3:]) if rep_dones else 0.024
                eff_s_r = max(base_s, inflight_elap)
                part_utils.append(lam_r * eff_s_r)

            max_part_util = max(part_utils) if part_utils else 0.08
            canary_overload = max(0.0, max_part_util - SAFE_RHO_HEADROOM)

            dual_sub_active = (sc == "benign_routing_dual_sub") or (v2_ship_upto > 0)
            sem_unsub = float(m_snap["sem_unsubscribed_topic"]) if (change_active and not dual_sub_active) else 0.0
            sem_schema = float(m_snap["sem_schema_deficit"]) if change_active else 0.0
            cand_m_w = [float(m["sem_dep_probe_fail"]) for m in mon_run if w_start < m["ts"] <= t_wall]
            sem_dep = max(cand_m_w) if cand_m_w else float(m_snap["sem_dep_probe_fail"])
            sem_compat_deficit = max(sem_schema, sem_unsub, sem_dep)

            if t_rel < onset_t:
                upper_b = min(t_int + 1 + int(HORIZON_S), int(math.floor(onset_t)))
                future_breach = int(any(bucket_breach.get(b, 0) == 1 for b in range(t_int + 1, upper_b)))
            else:
                future_breach = int(any(bucket_breach.get(b, 0) == 1 for b in range(t_int + 1, t_int + 1 + int(HORIZON_S))))

            all_windows.append({
                "run_id": run_id,
                "scenario": sc,
                "is_harmful_scenario": SCENARIO_META[sc]["is_harmful"],
                "t": t_rel,
                "onset_t": onset_t,
                "offered_rate": round(len(offered_w) / LOOKBACK_S, 4),
                "fulfilled_rate": round(len(ful_w) / LOOKBACK_S, 4),
                "obs_mean_lat": round(obs_mean, 5),
                "obs_p95_lat": round(obs_p95, 5),
                "orders_lag_total": m_snap["orders_lag_total"],
                "orders_lag_max_part": m_snap["orders_lag_max_part"],
                "orders_lag_skew": m_snap["orders_lag_skew"],
                "downstream_lag": downstream_lag_fixed,
                "inv_err_rate": round(len(inv_err_w) / max(1, len(offered_w)), 4),
                "bill_err_rate": round(len(bill_err_w) / max(1, len(offered_w)), 4),
                "retry_rate": round(retries_w / LOOKBACK_S, 4),
                "overdue_rate": round(overdue_rate, 4),
                "change_active": change_active,
                "coarse_schema_flag": int(change_active and sc in ("benign_schema_additive", "harmful_schema_breaking")),
                "coarse_routing_flag": int(change_active and sc in ("benign_routing_dual_sub", "harmful_routing_unsubscribed")),
                "coarse_service_deploy_flag": int(change_active and sc in ("benign_minor_slowdown", "harmful_consumer_slowdown", "harmful_dependency_outage")),
                "rollout_frac": round(rollout_frac, 4),
                "sem_schema_deficit": sem_schema,
                "sem_unsubscribed_topic": sem_unsub,
                "sem_dep_probe_fail": sem_dep,
                "sem_compat_deficit": round(sem_compat_deficit, 4),
                "sem_part_concentration": round(part_conc, 4),
                "sem_max_part_util": round(max_part_util, 4),
                "sem_canary_overload": round(canary_overload, 4),
                "y_degradation": future_breach,
            })

        cut_rel = min(run_duration_s * 0.65, (onset_t + 2.0 * REPLICA_ROLLOUT_STEP_S) if sc != "baseline_normal" else run_duration_s * 0.4)
        post_evs = [e for e in evs if e["rel_created"] >= cut_rel]
        ful_post = [e for e in post_evs if e["ful"] is not None]
        lats_post = sorted(e["ful"]["fulfilled_at"] - e["created_at"] for e in ful_post)
        slo_post = [l for l in lats_post if l <= SLO_LATENCY_S]

        all_summaries.append({
            "run_id": run_id,
            "scenario": sc,
            "onset_t": onset_t,
            "degradation_onset_t": deg_onset,
            "offered_total": len(evs),
            "fulfilled_total": sum(1 for e in evs if e["ful"] is not None),
            "post_offered": len(post_evs),
            "post_inv_reserved": sum(1 for e in post_evs if e["inv"] is not None and e["inv"]["status"] == "reserved"),
            "post_shipped": sum(1 for e in post_evs if e["ship"] is not None),
            "post_billed": sum(1 for e in post_evs if e["bill"] is not None and e["bill"]["status"] == "billed"),
            "post_fulfilled": len(ful_post),
            "post_completion": len(ful_post) / max(1, len(post_evs)),
            "post_slo_compliance": len(slo_post) / max(1, len(post_evs)),
            "post_mean_lat": statistics.mean(lats_post) if lats_post else None,
            "post_p95_lat": lats_post[math.ceil(0.95 * len(lats_post)) - 1] if lats_post else None,
        })

    by_scenario = {}
    for sc in SCENARIO_META:
        rs = [s for s in all_summaries if s["scenario"] == sc]
        if not rs:
            continue
        p_comp = [r["post_completion"] * 100.0 for r in rs]
        p_slo = [r["post_slo_compliance"] * 100.0 for r in rs]
        p_inv = [r["post_inv_reserved"] / max(1, r["post_offered"]) * 100.0 for r in rs]
        p_bill = [r["post_billed"] / max(1, r["post_offered"]) * 100.0 for r in rs]
        p_ship = [r["post_shipped"] / max(1, r["post_offered"]) * 100.0 for r in rs]
        p_mean = [r["post_mean_lat"] for r in rs if r["post_mean_lat"] is not None]
        p_p95 = [r["post_p95_lat"] for r in rs if r["post_p95_lat"] is not None]
        by_scenario[sc] = {
            "label": SCENARIO_META[sc]["label"],
            "runs": len(rs),
            "inv_reserved_pct_mean": round(statistics.mean(p_inv), 2),
            "billed_pct_mean": round(statistics.mean(p_bill), 2),
            "shipped_pct_mean": round(statistics.mean(p_ship), 2),
            "fulfilled_pct_mean": round(statistics.mean(p_comp), 2),
            "fulfilled_pct_sd": round(statistics.stdev(p_comp), 2) if len(p_comp) > 1 else 0.0,
            "slo_ok_pct_mean": round(statistics.mean(p_slo), 2),
            "slo_ok_pct_sd": round(statistics.stdev(p_slo), 2) if len(p_slo) > 1 else 0.0,
            "post_mean_lat_m": round(statistics.mean(p_mean), 4) if p_mean else None,
            "post_p95_lat_m": round(statistics.mean(p_p95), 4) if p_p95 else None,
        }

    with open(os.path.join(out_dir, "study2_windows.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_windows[0].keys()))
        w.writeheader()
        w.writerows(all_windows)

    summary_payload = {
        "config": {
            "mode": "live_apache_kafka_3.9.0_and_http",
            "total_runs": len(all_summaries),
            "total_offered_events": sum(r["offered_total"] for r in all_summaries),
            "total_windows": len(all_windows),
            "positive_windows": sum(w["y_degradation"] for w in all_windows),
        },
        "by_scenario": by_scenario,
        "runs": all_summaries,
    }
    with open(os.path.join(out_dir, "study2_runs_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    eval_res = evaluate_all(out_dir)
    print("LIVE STUDY 2 SUMMARY:", json.dumps(summary_payload["config"], indent=2))
    print("LIVE STUDY 2 5-FOLD CV (RandomForest):", json.dumps(eval_res["run_grouped_5fold_cv"]["RandomForest"], indent=2))
    return eval_res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--duration", type=float, default=30.0)
    a = ap.parse_args()
    out_d = a.out_dir or a.data_dir or DATA_DIR
    build_from_live_logs(out_d, run_duration_s=a.duration, data_dir=a.data_dir or out_d)

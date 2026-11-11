"""Execute one live 30-s Poisson order workload run on real Apache Kafka and wait until drained."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

from common_live import (
    BOOT,
    DATA_DIR,
    N_PARTITIONS,
    REPLICA_ROLLOUT_STEP_S,
    JsonlAppender,
    ensure_topics,
    partition_for_key,
    retry,
)


def write_control_state(payload: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    tmp = os.path.join(DATA_DIR, "control_state.json.tmp")
    dst = os.path.join(DATA_DIR, "control_state.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    os.replace(tmp, dst)


def wait_until_broker_drained(dual_sub: bool, timeout_s: float = 45.0):
    from kafka import KafkaConsumer, TopicPartition
    from kafka.admin import KafkaAdminClient

    admin = retry(lambda: KafkaAdminClient(bootstrap_servers=BOOT))
    obs = retry(lambda: KafkaConsumer(bootstrap_servers=BOOT, enable_auto_commit=False))
    deadline = time.time() + timeout_s
    try:
        while time.time() < deadline:
            inv_off = admin.list_consumer_group_offsets("cacdp-inventory")
            ship_off = admin.list_consumer_group_offsets("cacdp-shipping")
            bill_off = admin.list_consumer_group_offsets("cacdp-billing")
            ful_off = admin.list_consumer_group_offsets("cacdp-fulfillment")

            checks = [
                ("orders", inv_off),
                ("inventory.reserved", ship_off),
                ("billing.events", bill_off),
                ("fulfillment.events", ful_off),
            ]
            if dual_sub:
                checks.append(("inventory.reserved.v2", ship_off))

            all_tps = [TopicPartition(t, p) for t, _ in checks for p in range(N_PARTITIONS)]
            ends = obs.end_offsets(all_tps)
            rem = 0
            for t_name, grp in checks:
                for p in range(N_PARTITIONS):
                    tp = TopicPartition(t_name, p)
                    com = grp.get(tp)
                    c_val = com.offset if (com is not None and com.offset >= 0) else 0
                    rem += max(0, int(ends.get(tp, 0)) - int(c_val))
            if rem == 0:
                time.sleep(0.6)
                return
            time.sleep(0.4)
    finally:
        obs.close()
        admin.close()


def main():
    from kafka import KafkaProducer

    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--rep", type=int, required=True)
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    ensure_topics()
    rng = random.Random(args.seed)
    run_id = f"{args.scenario}-r{args.rep:02d}"
    t_start = time.time()
    lo, hi = (3.0, 4.2) if args.duration <= 15.0 else (9.0, 13.0)
    deploy_offset = round(rng.uniform(lo, hi), 2) if args.scenario != "baseline_normal" else 999.0
    deploy_ts = t_start + deploy_offset
    manifest_ts = deploy_ts - 1.0
    dual_sub = args.scenario == "benign_routing_dual_sub"

    # Initial pre-manifest control state
    write_control_state({
        "run_id": run_id,
        "scenario": args.scenario,
        "run_start_ts": t_start,
        "deploy_ts": deploy_ts,
        "manifest_ts": manifest_ts,
        "shipping_dual_sub": dual_sub,
        "target_topic": "inventory.reserved",
        "required_fields": ["sku"],
        "producer_fields": ["sku"],
    })
    if dual_sub:
        time.sleep(0.6)  # Allow shipping consumer to pick up dual subscription before run starts
        t_start = time.time()
        deploy_ts = t_start + deploy_offset
        manifest_ts = deploy_ts - 1.0

    producer = retry(lambda: KafkaProducer(bootstrap_servers=BOOT, acks=1, linger_ms=0))
    prod_log = JsonlAppender("produced.jsonl")
    manifest_published = False

    rel_t = 0.0
    seq = 0
    while rel_t < args.duration:
        rate_now = args.rate * (1.0 + 0.15 * math.sin(2.0 * math.pi * rel_t / 9.0))
        rel_t += rng.expovariate(rate_now)
        if rel_t >= args.duration:
            break
        target_wall = t_start + rel_t
        now = time.time()
        if target_wall > now:
            time.sleep(target_wall - now)

        now = time.time()
        if (not manifest_published) and now >= manifest_ts and args.scenario != "baseline_normal":
            prod_fields = ["item_code"] if args.scenario == "harmful_schema_breaking" else (
                ["sku", "promo_code"] if args.scenario == "benign_schema_additive" else ["sku"]
            )
            tgt = "inventory.reserved.v2" if args.scenario in ("harmful_routing_unsubscribed", "benign_routing_dual_sub") else "inventory.reserved"
            write_control_state({
                "run_id": run_id,
                "scenario": args.scenario,
                "run_start_ts": t_start,
                "deploy_ts": deploy_ts,
                "manifest_ts": manifest_ts,
                "shipping_dual_sub": dual_sub,
                "target_topic": tgt,
                "required_fields": ["sku"],
                "producer_fields": prod_fields,
            })
            manifest_published = True

        eid = f"{run_id}-{seq:05d}"
        if now >= deploy_ts and args.scenario == "harmful_partition_hotspot":
            updated = sum(1 for r in range(N_PARTITIONS) if now >= deploy_ts + r * REPLICA_ROLLOUT_STEP_S)
            ramp = updated / N_PARTITIONS
            part = 0 if rng.random() < (0.88 * ramp) else rng.randint(1, N_PARTITIONS - 1)
        else:
            part = partition_for_key(eid)

        rep_updated = now >= (deploy_ts + part * REPLICA_ROLLOUT_STEP_S)
        ev = {
            "run_id": run_id,
            "event_id": eid,
            "seq": seq,
            "scenario": args.scenario,
            "created_at": now,
            "deploy_ts": deploy_ts,
            "partition": part,
        }
        if rep_updated and args.scenario == "harmful_schema_breaking":
            ev["item_code"] = "SKU-001"
        else:
            ev["sku"] = "SKU-001"
            if rep_updated and args.scenario == "benign_schema_additive":
                ev["promo_code"] = "SAVE10"

        producer.send("orders", value=json.dumps(ev).encode("utf-8"), partition=part)
        prod_log.write({
            "run_id": run_id,
            "event_id": eid,
            "scenario": args.scenario,
            "created_at": ev["created_at"],
            "run_start_ts": t_start,
            "deploy_ts": deploy_ts,
            "partition": part,
        })
        seq += 1

    producer.flush()
    producer.close()
    wait_until_broker_drained(dual_sub=dual_sub)
    print(json.dumps({"run_id": run_id, "events_sent": seq, "deploy_offset_s": deploy_offset}), flush=True)


if __name__ == "__main__":
    main()

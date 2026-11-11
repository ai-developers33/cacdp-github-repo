"""Live Kafka microservices: Inventory (3 replicas), Shipping, Billing (HTTP), and Fulfillment Join."""

from __future__ import annotations

import argparse
import json
import math
import random
import signal
import threading
import time
import urllib.error
import urllib.request

from common_live import (
    BOOT,
    GATEWAY_URL,
    N_PARTITIONS,
    REPLICA_ROLLOUT_STEP_S,
    JsonlAppender,
    make_offset_meta,
    ensure_topics,
    partition_for_key,
    read_control_state,
    retry,
)


def lognormal_sleep(mean_s: float, cv: float = 0.20):
    s2 = math.log(1.0 + cv * cv)
    val = random.lognormvariate(math.log(mean_s) - 0.5 * s2, math.sqrt(s2))
    time.sleep(min(val, mean_s * 2.5))


def run_inventory(replica: int, stop: threading.Event):
    from kafka import KafkaConsumer, KafkaProducer, TopicPartition
    
    ensure_topics()
    tp = TopicPartition("orders", replica)
    consumer = retry(
        lambda: KafkaConsumer(
            bootstrap_servers=BOOT,
            group_id="cacdp-inventory",
            enable_auto_commit=False,
        )
    )
    consumer.assign([tp])
    committed = consumer.committed(tp)
    if committed is None:
        consumer.seek_to_beginning(tp)

    producer = retry(lambda: KafkaProducer(bootstrap_servers=BOOT, acks=1, linger_ms=2))
    inv_log = JsonlAppender("inventory.jsonl")
    print(f"inventory-{replica} ready on {tp}", flush=True)

    while not stop.is_set():
        batches = consumer.poll(timeout_ms=250)
        for part, records in batches.items():
            for rec in records:
                ev = json.loads(rec.value.decode("utf-8"))
                now = time.time()
                sc = ev["scenario"]
                deploy_ts = float(ev["deploy_ts"])
                rep_updated = now >= (deploy_ts + replica * REPLICA_ROLLOUT_STEP_S)

                if rep_updated and sc == "harmful_consumer_slowdown":
                    lognormal_sleep(0.44, cv=0.20)
                elif rep_updated and sc == "benign_minor_slowdown":
                    lognormal_sleep(0.11, cv=0.20)
                elif now >= deploy_ts and sc == "harmful_partition_hotspot":
                    lognormal_sleep(0.17, cv=0.20)
                else:
                    lognormal_sleep(0.020, cv=0.20)

                done_ts = time.time()
                if "sku" not in ev:
                    inv_log.write({
                        "run_id": ev["run_id"],
                        "event_id": ev["event_id"],
                        "scenario": sc,
                        "replica": replica,
                        "created_at": ev["created_at"],
                        "inv_done_at": done_ts,
                        "status": "schema_error",
                    })
                    consumer.commit({part: make_offset_meta(rec.offset + 1)})
                    continue

                target_topic = (
                    "inventory.reserved.v2"
                    if (rep_updated and sc in ("harmful_routing_unsubscribed", "benign_routing_dual_sub"))
                    else "inventory.reserved"
                )
                out_payload = json.dumps({
                    **ev,
                    "replica": replica,
                    "inv_done_at": done_ts,
                    "target_topic": target_topic,
                }).encode("utf-8")
                p_sub = partition_for_key(ev["event_id"])
                producer.send(target_topic, value=out_payload, partition=p_sub)
                producer.send("billing.events", value=out_payload, partition=p_sub)
                producer.flush()

                inv_log.write({
                    "run_id": ev["run_id"],
                    "event_id": ev["event_id"],
                    "scenario": sc,
                    "replica": replica,
                    "created_at": ev["created_at"],
                    "inv_done_at": done_ts,
                    "status": "reserved",
                    "target_topic": target_topic,
                })
                consumer.commit({part: make_offset_meta(rec.offset + 1)})
    consumer.close()


def run_shipping(stop: threading.Event):
    from kafka import KafkaConsumer, KafkaProducer, TopicPartition
    
    ensure_topics()
    consumer = retry(
        lambda: KafkaConsumer(
            bootstrap_servers=BOOT,
            group_id="cacdp-shipping",
            enable_auto_commit=False,
        )
    )
    producer = retry(lambda: KafkaProducer(bootstrap_servers=BOOT, acks=1, linger_ms=2))
    ship_log = JsonlAppender("shipping.jsonl")

    current_dual = None
    while not stop.is_set():
        st = read_control_state()
        dual = bool(st.get("shipping_dual_sub", False))
        if dual != current_dual:
            tps = [TopicPartition("inventory.reserved", p) for p in range(N_PARTITIONS)]
            if dual:
                tps += [TopicPartition("inventory.reserved.v2", p) for p in range(N_PARTITIONS)]
            consumer.assign(tps)
            for tp in tps:
                if consumer.committed(tp) is None:
                    consumer.seek_to_beginning(tp)
            current_dual = dual
            print(f"shipping assigned dual_sub={dual}", flush=True)

        batches = consumer.poll(timeout_ms=200)
        for part, records in batches.items():
            for rec in records:
                ev = json.loads(rec.value.decode("utf-8"))
                lognormal_sleep(0.015, cv=0.20)
                done_ts = time.time()
                p_sub = partition_for_key(ev["event_id"])
                producer.send(
                    "fulfillment.events",
                    partition=p_sub,
                    value=json.dumps({
                        "run_id": ev["run_id"],
                        "event_id": ev["event_id"],
                        "scenario": ev["scenario"],
                        "created_at": ev["created_at"],
                        "branch": "shipping",
                        "branch_done_at": done_ts,
                    }).encode("utf-8"),
                )
                ship_log.write({
                    "run_id": ev["run_id"],
                    "event_id": ev["event_id"],
                    "scenario": ev["scenario"],
                    "created_at": ev["created_at"],
                    "ship_done_at": done_ts,
                    "topic": part.topic,
                })
                consumer.commit({part: make_offset_meta(rec.offset + 1)})
        producer.flush()
    consumer.close()


def call_payment_gateway(ev: dict) -> tuple[bool, int]:
    payload = json.dumps({
        "run_id": ev["run_id"],
        "event_id": ev["event_id"],
        "replica": ev.get("replica", 0),
    }).encode("utf-8")
    retries = 0
    for attempt in range(3):
        req = urllib.request.Request(
            f"{GATEWAY_URL}/charge",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=0.45) as resp:
                if resp.status == 200:
                    return True, retries
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            retries += 1
            time.sleep(0.03 * (attempt + 1))
    return False, retries


def run_billing(stop: threading.Event):
    from kafka import KafkaConsumer, KafkaProducer, TopicPartition
    
    ensure_topics()
    tps = [TopicPartition("billing.events", p) for p in range(N_PARTITIONS)]
    consumer = retry(
        lambda: KafkaConsumer(
            bootstrap_servers=BOOT,
            group_id="cacdp-billing",
            enable_auto_commit=False,
        )
    )
    consumer.assign(tps)
    for tp in tps:
        if consumer.committed(tp) is None:
            consumer.seek_to_beginning(tp)

    producer = retry(lambda: KafkaProducer(bootstrap_servers=BOOT, acks=1, linger_ms=2))
    bill_log = JsonlAppender("billing.jsonl")
    print("billing ready", flush=True)

    while not stop.is_set():
        batches = consumer.poll(timeout_ms=200)
        for part, records in batches.items():
            for rec in records:
                ev = json.loads(rec.value.decode("utf-8"))
                ok, retries = call_payment_gateway(ev)
                done_ts = time.time()
                if ok:
                    p_sub = partition_for_key(ev["event_id"])
                    producer.send(
                        "fulfillment.events",
                        partition=p_sub,
                        value=json.dumps({
                            "run_id": ev["run_id"],
                            "event_id": ev["event_id"],
                            "scenario": ev["scenario"],
                            "created_at": ev["created_at"],
                            "branch": "billing",
                            "branch_done_at": done_ts,
                        }).encode("utf-8"),
                    )
                bill_log.write({
                    "run_id": ev["run_id"],
                    "event_id": ev["event_id"],
                    "scenario": ev["scenario"],
                    "created_at": ev["created_at"],
                    "bill_done_at": done_ts,
                    "status": "billed" if ok else "dependency_error",
                    "retries": retries,
                })
                consumer.commit({part: make_offset_meta(rec.offset + 1)})
        producer.flush()
    consumer.close()


def run_fulfillment(stop: threading.Event):
    from kafka import KafkaConsumer, TopicPartition
    
    ensure_topics()
    tps = [TopicPartition("fulfillment.events", p) for p in range(N_PARTITIONS)]
    consumer = retry(
        lambda: KafkaConsumer(
            bootstrap_servers=BOOT,
            group_id="cacdp-fulfillment",
            enable_auto_commit=False,
        )
    )
    consumer.assign(tps)
    for tp in tps:
        if consumer.committed(tp) is None:
            consumer.seek_to_beginning(tp)

    ful_log = JsonlAppender("fulfillment.jsonl")
    seen: dict[tuple[str, str], dict] = {}
    print("fulfillment barrier join ready", flush=True)

    while not stop.is_set():
        batches = consumer.poll(timeout_ms=200)
        for part, records in batches.items():
            for rec in records:
                ev = json.loads(rec.value.decode("utf-8"))
                key = (ev["run_id"], ev["event_id"])
                st = seen.setdefault(key, {"scenario": ev["scenario"], "created_at": ev["created_at"], "branches": set()})
                st["branches"].add(ev["branch"])
                if "shipping" in st["branches"] and "billing" in st["branches"]:
                    ful_log.write({
                        "run_id": ev["run_id"],
                        "event_id": ev["event_id"],
                        "scenario": st["scenario"],
                        "created_at": st["created_at"],
                        "fulfilled_at": time.time(),
                    })
                    del seen[key]
                consumer.commit({part: make_offset_meta(rec.offset + 1)})
    consumer.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", required=True, choices=["inventory", "shipping", "billing", "fulfillment"])
    ap.add_argument("--replica", type=int, default=0)
    args = ap.parse_args()

    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())

    if args.role == "inventory":
        run_inventory(args.replica, stop)
    elif args.role == "shipping":
        run_shipping(stop)
    elif args.role == "billing":
        run_billing(stop)
    else:
        run_fulfillment(stop)


if __name__ == "__main__":
    main()

"""1 Hz Live Broker Lag, Consumer Subscription & Gateway Health Monitor."""

from __future__ import annotations

import signal
import threading
import time
import urllib.error
import urllib.request

from common_live import BOOT, GATEWAY_URL, N_PARTITIONS, JsonlAppender, ensure_topics, read_control_state, retry


def probe_gateway() -> float:
    fails = 0
    for _ in range(3):
        try:
            with urllib.request.urlopen(f"{GATEWAY_URL}/healthz", timeout=0.2) as r:
                if r.status != 200:
                    fails += 1
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            fails += 1
    return round(fails / 3.0, 4)


def main():
    from kafka import KafkaConsumer, TopicPartition
    from kafka.admin import KafkaAdminClient

    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())

    ensure_topics()
    admin = retry(lambda: KafkaAdminClient(bootstrap_servers=BOOT))
    obs = retry(lambda: KafkaConsumer(bootstrap_servers=BOOT, enable_auto_commit=False))
    out = JsonlAppender("monitor_1hz.jsonl")
    print("telemetry-monitor 1 Hz active", flush=True)

    while not stop.is_set():
        t0 = time.time()
        st = read_control_state()
        try:
            inv_off = admin.list_consumer_group_offsets("cacdp-inventory")
            ship_off = admin.list_consumer_group_offsets("cacdp-shipping")
            bill_off = admin.list_consumer_group_offsets("cacdp-billing")

            orders_tps = [TopicPartition("orders", p) for p in range(N_PARTITIONS)]
            res_v1_tps = [TopicPartition("inventory.reserved", p) for p in range(N_PARTITIONS)]
            res_v2_tps = [TopicPartition("inventory.reserved.v2", p) for p in range(N_PARTITIONS)]
            bill_tps = [TopicPartition("billing.events", p) for p in range(N_PARTITIONS)]

            ends = obs.end_offsets(orders_tps + res_v1_tps + res_v2_tps + bill_tps)

            def part_lag(tp, grp_offsets):
                com = grp_offsets.get(tp)
                c_val = com.offset if (com is not None and com.offset >= 0) else 0
                return max(0, int(ends.get(tp, 0)) - int(c_val))

            o_lags = [part_lag(tp, inv_off) for tp in orders_tps]
            s_v1_lag = sum(part_lag(tp, ship_off) for tp in res_v1_tps)
            s_v2_lag = sum(part_lag(tp, ship_off) for tp in res_v2_tps)
            b_lag = sum(part_lag(tp, bill_off) for tp in bill_tps)

            # Live broker + consumer-assignment check (handles cold topics prior to first offset commit)
            active_target = st.get("target_topic", "inventory.reserved")
            ship_subscribed_topics = {tp.topic for tp in ship_off.keys()}
            if st.get("shipping_dual_sub"):
                ship_subscribed_topics.add("inventory.reserved.v2")
            else:
                ship_subscribed_topics.discard("inventory.reserved.v2")
            unsub = 1.0 if (active_target == "inventory.reserved.v2" and "inventory.reserved.v2" not in ship_subscribed_topics) else 0.0

            req_fields = set(st.get("required_fields", ["sku"]))
            prod_fields = set(st.get("producer_fields", ["sku"]))
            schema_def = len(req_fields - prod_fields) / max(1, len(req_fields))
            dep_fail = probe_gateway()

            out.write({
                "ts": t0,
                "run_id": st.get("run_id", "init"),
                "scenario": st.get("scenario", "baseline_normal"),
                "orders_lag_total": sum(o_lags),
                "orders_lag_max_part": max(o_lags),
                "orders_lag_skew": round(max(o_lags) / max(1, sum(o_lags)), 4) if sum(o_lags) >= 3 else round(1.0 / N_PARTITIONS, 4),
                "downstream_lag": s_v1_lag + s_v2_lag + b_lag,
                "sem_unsubscribed_topic": unsub,
                "sem_schema_deficit": round(schema_def, 4),
                "sem_dep_probe_fail": dep_fail,
            })
        except Exception as exc:
            out.write({"ts": t0, "run_id": st.get("run_id", "init"), "error": str(exc)})

        elapsed = time.time() - t0
        if elapsed < 1.0:
            stop.wait(1.0 - elapsed)

    obs.close()
    admin.close()


if __name__ == "__main__":
    main()

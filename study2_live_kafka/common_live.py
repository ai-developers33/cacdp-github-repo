"""Shared helpers, Kafka topic bootstrap, and atomic JSONL writers for Live Study 2."""

from __future__ import annotations

import json
import os
import threading
import time
import zlib

BOOT = os.environ.get("KAFKA_BOOTSTRAP", "kafka:9092")
GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://payment-gateway:8080")
DATA_DIR = os.environ.get("DATA_DIR", "/data")

TOPICS = [
    "orders",
    "inventory.reserved",
    "inventory.reserved.v2",
    "billing.events",
    "fulfillment.events",
]
N_PARTITIONS = 3
REPLICA_ROLLOUT_STEP_S = 1.5


def partition_for_key(key: str) -> int:
    return zlib.crc32(key.encode("utf-8")) % N_PARTITIONS


def make_offset_meta(next_offset: int):
    from kafka.structs import OffsetAndMetadata

    fields = getattr(OffsetAndMetadata, "_fields", ())
    if len(fields) == 3:
        return OffsetAndMetadata(next_offset, "", -1)
    if len(fields) == 4:
        return OffsetAndMetadata(next_offset, "", -1, b"")
    return OffsetAndMetadata(next_offset, "")


def retry(fn, tries: int = 60, delay: float = 1.5):
    last = None
    for _ in range(tries):
        try:
            return fn()
        except Exception as exc:
            last = exc
            time.sleep(delay)
    raise last


def ensure_topics():
    from kafka.admin import KafkaAdminClient, NewTopic
    from kafka.errors import TopicAlreadyExistsError

    admin = retry(lambda: KafkaAdminClient(bootstrap_servers=BOOT))
    try:
        for name in TOPICS:
            try:
                admin.create_topics([NewTopic(name=name, num_partitions=N_PARTITIONS, replication_factor=1)])
            except TopicAlreadyExistsError:
                pass
    finally:
        admin.close()


def read_control_state() -> dict:
    path = os.path.join(DATA_DIR, "control_state.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "run_id": "init",
            "scenario": "baseline_normal",
            "deploy_ts": 9e12,
            "manifest_ts": 9e12,
            "shipping_dual_sub": False,
            "required_fields": ["sku"],
            "producer_fields": ["sku"],
        }


class JsonlAppender:
    def __init__(self, filename: str):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.path = os.path.join(DATA_DIR, filename)
        self._lock = threading.Lock()

    def write(self, record: dict):
        line = (json.dumps(record) + "\n").encode("utf-8")
        with self._lock:
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)

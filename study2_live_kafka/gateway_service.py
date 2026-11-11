"""Live HTTP Payment Gateway microservice (ThreadingHTTPServer on port 8080).

Exposes:
  - GET  /ping    -> 200 OK (container readiness probe)
  - GET  /healthz -> synthetic dependency probe used by Control-Plane Monitor
  - POST /charge  -> live order payment endpoint called by BillingService over TCP/HTTP
"""

from __future__ import annotations

import json
import os
import random
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DATA_DIR = os.environ.get("DATA_DIR", "/data")
PORT = int(os.environ.get("GATEWAY_PORT", "8080"))


def read_control_state() -> dict:
    p = os.path.join(DATA_DIR, "control_state.json")
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"scenario": "baseline_normal", "deploy_ts": 9e12}


class GatewayHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return  # Keep container stdout quiet

    def _send(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/ping":
            self._send(200, {"ok": True})
            return
        if self.path == "/healthz":
            st = read_control_state()
            now = time.time()
            deploy_ts = float(st.get("deploy_ts", 9e12))
            sc = st.get("scenario", "baseline_normal")
            if sc == "harmful_dependency_outage" and now >= deploy_ts:
                ramp = min(1.0, (1.0 + max(0.0, now - deploy_ts) // 1.5) / 3.0)
                if random.random() < 0.78 * ramp:
                    time.sleep(0.05)
                    self._send(503, {"healthy": False, "error": "gateway_timeout"})
                    return
            self._send(200, {"healthy": True})
            return
        self._send(404, {"error": "not_found"})

    def do_POST(self):
        if self.path != "/charge":
            self._send(404, {"error": "not_found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length > 0 else b"{}"
        req = json.loads(raw.decode("utf-8"))
        st = read_control_state()
        now = time.time()
        deploy_ts = float(st.get("deploy_ts", 9e12))
        sc = st.get("scenario", "baseline_normal")
        replica = int(req.get("replica", 0))
        rep_active = now >= (deploy_ts + replica * 1.5)

        if rep_active and sc == "harmful_dependency_outage":
            if random.random() < 0.80:
                time.sleep(0.08)
                self._send(503, {"status": "dependency_error", "reason": "upstream_bank_timeout"})
                return
            time.sleep(0.035)
            self._send(200, {"status": "billed"})
            return

        # Normal operation (with 1.5% realistic transient glitch)
        if random.random() < 0.015:
            time.sleep(0.03)
            self._send(503, {"status": "transient_error"})
            return
        time.sleep(0.015)
        self._send(200, {"status": "billed"})


def main():
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), GatewayHandler)
    print(f"payment-gateway listening on :{PORT}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()

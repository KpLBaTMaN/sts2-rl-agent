"""Local live dashboard: python -m sts2_env.training.dashboard --root <batch>.

No remote dependencies. Training is owned by the independent comparison process;
closing the page or dashboard does not stop the experiments.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import subprocess
import threading
import time
from urllib.parse import parse_qs, urlparse

from sts2_env.training.comparison import experiment_directory, read_json
from sts2_env.training.telemetry import atomic_json

ASSETS = Path(__file__).with_name("dashboard_assets")


def read_metrics(directory):
    records = []
    if directory:
        try:
            with (directory / "metrics.jsonl").open(encoding="utf-8") as stream:
                for line in stream:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # Last append may still be in progress.
        except FileNotFoundError:
            pass
    return records


def tail(path, limit=20000):
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - limit))
            return stream.read().decode("utf-8", errors="replace")
    except FileNotFoundError:
        return "Waiting for this run to start."


def snapshot(root):
    manifest = read_json(root / "batch.json", {})
    runs = []
    for entry in manifest.get("runs", []):
        directory = experiment_directory(root, entry)
        records = read_metrics(directory)
        checkpoints = [r for r in records if r["kind"] == "checkpoint"]
        runs.append(dict(entry, directory=str(directory) if directory else None,
                         config=read_json(Path(entry["config"]), {}),
                         live=read_json(directory / "live.json", {}) if directory else {},
                         rollouts=[r for r in records if r["kind"] == "rollout"],
                         evaluations=[r for r in records if r["kind"] == "evaluation"],
                         checkpoints=checkpoints,
                         final=read_json(directory / "final-results.json") if directory else None,
                         best=read_json(directory / "best-results.json") if directory else None))
    return dict(batch={k: v for k, v in manifest.items() if k != "runs"}, runs=runs,
                control=read_json(root / "queue-control.json", {}),
                server_utc=datetime.now(timezone.utc).isoformat(), root=str(root))


class DashboardServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, root):
        super().__init__(address, Handler)
        self.root = root
        self.hardware = {}
        self.hardware_updated = 0
        self.hardware_lock = threading.Lock()

    def gpu(self):
        with self.hardware_lock:
            if time.monotonic() - self.hardware_updated > 10:
                try:
                    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
                    result = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"],
                                            capture_output=True, text=True, timeout=3, creationflags=flags)
                    if result.returncode == 0:
                        name, utilization, used, total, temperature = result.stdout.splitlines()[0].split(", ")
                        self.hardware = dict(name=name, utilization=float(utilization), memory_used=float(used),
                                             memory_total=float(total), temperature=float(temperature))
                except (OSError, ValueError, subprocess.TimeoutExpired, IndexError):
                    self.hardware = {}
                self.hardware_updated = time.monotonic()
            return self.hardware


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send(self, data, content_type="application/json", status=200, filename=None):
        if not isinstance(data, bytes):
            data = json.dumps(data, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
        if filename:
            self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/", "/app.js", "/style.css"):
            name = {"/": "index.html", "/app.js": "app.js", "/style.css": "style.css"}[parsed.path]
            content_type = {"index.html": "text/html; charset=utf-8", "app.js": "text/javascript; charset=utf-8", "style.css": "text/css; charset=utf-8"}[name]
            return self.send((ASSETS / name).read_bytes(), content_type)
        if parsed.path == "/api/state":
            state = snapshot(self.server.root)
            state["gpu"] = self.server.gpu()
            return self.send(state)
        query = parse_qs(parsed.query)
        profile = query.get("profile", ["sparse"])[0]
        manifest = read_json(self.server.root / "batch.json", {})
        entry = next((e for e in manifest.get("runs", []) if e["profile"] == profile), None)
        if not entry:
            return self.send(dict(error="Unknown run"), status=404)
        if parsed.path == "/api/log":
            return self.send(dict(text=tail(self.server.root / f"{profile}.log")))
        if parsed.path == "/api/export":
            directory = experiment_directory(self.server.root, entry)
            kind = query.get("kind", ["metrics"])[0]
            allowed = {"config": "config.json", "episodes": "best-results-episodes.csv", "final-episodes": "final-results-episodes.csv"}
            if kind in allowed and directory and (directory / allowed[kind]).exists():
                name = allowed[kind]
                return self.send((directory / name).read_bytes(), "application/octet-stream", filename=f"{profile}-{name}")
            if kind == "metrics":
                rows = [r for r in read_metrics(directory) if r["kind"] == "rollout"]
                flattened = [flatten(row) for row in rows]
                output = io.StringIO(newline="")
                if flattened:
                    writer = csv.DictWriter(output, fieldnames=sorted(set().union(*(r.keys() for r in flattened))))
                    writer.writeheader()
                    writer.writerows(flattened)
                return self.send(output.getvalue().encode(), "text/csv", filename=f"{profile}-metrics.csv")
        return self.send(dict(error="Not found or data not available yet"), status=404)

    def do_POST(self):
        # Only the served page may control this queue. No arbitrary files or processes.
        origin = self.headers.get("Origin")
        expected = f"http://{self.headers.get('Host')}"
        if origin != expected or self.headers.get("X-Dashboard-Request") != "1":
            return self.send(dict(error="Same-origin dashboard request required"), status=403)
        if self.path not in ("/api/pause-queue", "/api/resume-queue"):
            return self.send(dict(error="Unknown action"), status=404)
        value = dict(pause_after_current=self.path == "/api/pause-queue")
        atomic_json(self.server.root / "queue-control.json", value)
        self.send(value)


def flatten(value, prefix=""):
    result = {}
    for key, item in value.items():
        name = prefix + key
        if isinstance(item, dict):
            result.update(flatten(item, name + "."))
        else:
            result[name] = item
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    root = args.root.resolve()
    if not (root / "batch.json").exists():
        parser.error("Prepare the comparison first; batch.json is missing")
    server = DashboardServer(("127.0.0.1", args.port), root)
    print(f"Training dashboard: http://127.0.0.1:{args.port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

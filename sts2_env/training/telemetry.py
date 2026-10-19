"""Small, durable progress files shared by the trainer and local dashboard."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    # A Windows reader can briefly hold the destination open.
    for attempt in range(10):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.02)


def publish(directory: Path, kind: str, **values) -> dict:
    record = dict(kind=kind, updated_utc=datetime.now(timezone.utc).isoformat(), **values)
    with (directory / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
    atomic_json(directory / "live.json", record)
    return record

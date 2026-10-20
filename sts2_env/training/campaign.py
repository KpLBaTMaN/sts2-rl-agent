"""Run prepared training queues in order, retaining completed experiments.

Launch this module in a detached process; campaign.json and per-stage logs are
durable, and the training dashboard can watch the campaign root directly.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import traceback

from sts2_env.training.comparison import PROJECT_ROOT, now, read_json
from sts2_env.training.telemetry import atomic_json


def source_digest():
    """Freeze training semantics, while allowing dashboard/catalog improvements."""
    digest = hashlib.sha256()
    for path in sorted((PROJECT_ROOT / "sts2_env").rglob("*.py")):
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        if "/watch/" in relative or path.name in {"campaign.py", "dashboard.py"}:
            continue
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def process_alive(pid):
    if not pid:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            error = ctypes.get_last_error()
            if error == 87:  # No such process.
                return False
            raise RuntimeError(f"Cannot inspect process {pid} (Windows error {error})")
        try:
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                raise RuntimeError(f"Cannot inspect process {pid}")
            return code.value == 259  # STILL_ACTIVE
        finally:
            kernel.CloseHandle(handle)
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False


def recover_queue_lock(root, batch):
    """Only remove an orphan lock after checking both queue and trainer owners."""
    lock = root / "queue.lock"
    owners = {batch.get("pid")}
    if lock.exists():
        owners.add(int(lock.read_text().strip()))
    owners.update(entry.get("pid") for entry in batch["runs"] if entry["status"] == "running")
    for owner in owners:
        if process_alive(owner):
            raise RuntimeError(f"Queue {root} still has an active owner ({owner})")
    if lock.exists():
        lock.unlink()


def execute(root):
    root = Path(root).resolve()
    path, lock = root / "campaign.json", root / "campaign.lock"
    campaign = read_json(path)
    if not campaign:
        raise ValueError(f"Missing campaign.json in {root}")
    if campaign.get("status") == "completed":
        return
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    child = None
    try:
        campaign.update(status="running", pid=os.getpid(), started_utc=now(), error=None)
        atomic_json(path, campaign)
        for stage in campaign["stages"]:
            queue = Path(stage["root"])
            batch = read_json(queue / "batch.json")
            if not batch:
                raise RuntimeError(f"Missing prepared queue: {queue}")
            if batch["status"] == "completed":
                continue
            if source_digest() != campaign["source_digest"]:
                raise RuntimeError("Training code changed during this campaign; stopped before mixing implementations.")
            recover_queue_lock(queue, batch)
            campaign.update(current_stage=stage["id"], updated_utc=now())
            atomic_json(path, campaign)
            command = [sys.executable, "-u", "-m", "sts2_env.training.comparison", "--root", str(queue)]
            if batch["status"] != "queued":
                command.append("--resume")
            with (root / f"{stage['id']}.log").open("a", encoding="utf-8") as log:
                print(f"{now()} Starting {stage['id']}", flush=True)
                child = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL)
                campaign["child_pid"] = child.pid
                atomic_json(path, campaign)
                code = child.wait()
            if code or read_json(queue / "batch.json", {}).get("status") != "completed":
                raise RuntimeError(f"{stage['id']} failed (exit {code}); later stages were not started.")
        campaign.update(status="completed", current_stage=None, child_pid=None, finished_utc=now())
        atomic_json(path, campaign)
    except BaseException as exc:
        if child and child.poll() is None:
            child.terminate()
            child.wait()
        campaign.update(status="failed", error=str(exc), finished_utc=now())
        atomic_json(path, campaign)
        traceback.print_exc()
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    execute(parser.parse_args().root)


if __name__ == "__main__":
    main()

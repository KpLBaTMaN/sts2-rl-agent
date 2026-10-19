"""Prepare and execute a durable, strictly sequential reward comparison."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from sts2_env.training.telemetry import atomic_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILES = ("sparse", "shaped", "survival")


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare(root: Path, *, smoke=False):
    from sts2_env.training.config import load_config
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    entries = []
    for profile in PROFILES:
        config = dict(name=f"long-{profile}", task="act1", character="Ironclad", ascension=0,
                      start_with_neow=False, reward=profile, resume_from=None,
                      seeds=dict(mode="random", values=[42], schedule_seed=123),
                      model=dict(type="entity", d_model=128, layers=3, heads=4),
                      training=dict(total_timesteps=5_000_000, n_envs=16, n_steps=256,
                                    batch_size=512, n_epochs=4, model_seed=42, learning_rate=0.0003,
                                    ent_coef=0.02, gamma=0.995, device="cuda", torch_threads=4,
                                    max_episode_steps=2000, max_combat_turns=200, checkpoint_every=100_000),
                      evaluation=dict(mode="unseen", seeds=list(range(10000, 10050)),
                                      final_seeds=list(range(20000, 20200)), every_steps=250_000,
                                      episodes=50, final_episodes=200, deterministic=True,
                                      random_baseline_episodes=50), output_root=str(root / "runs"))
        if smoke:
            config["name"] = f"smoke-{profile}"
            config["training"].update(total_timesteps=32, n_envs=2, n_steps=16, batch_size=16,
                                      n_epochs=1, max_episode_steps=16, checkpoint_every=16)
            config["evaluation"].update(every_steps=32, episodes=2, final_episodes=2,
                                        random_baseline_episodes=0)
        path = root / "configs" / f"{profile}.json"
        atomic_json(path, config)
        resolved = load_config(path)
        atomic_json(path, resolved)  # Freeze the exact reward weights, not just a mutable profile name.
        entries.append(dict(profile=profile, name=config["name"], config=str(path), status="queued"))
    manifest = dict(title="SpireFormer · Reward comparison", created_utc=now(), status="queued",
                    smoke=smoke, runs=entries, pid=None)
    atomic_json(root / "batch.json", manifest)
    atomic_json(root / "queue-control.json", dict(pause_after_current=False))
    return manifest


def experiment_directory(root, entry):
    directories = sorted((root / "runs").glob(entry["name"] + "-*"))
    return directories[-1] if directories else None


def write_comparison(root, manifest):
    rows = []
    for entry in manifest["runs"]:
        directory = experiment_directory(root, entry)
        summary = read_json(directory / "best-results.json", {}) if directory else {}
        rows.append(dict(profile=entry["profile"], status=entry["status"],
                         **{key: summary.get(key) for key in ("episodes", "wins", "win_rate", "average_floor",
                            "boss_reached_rate", "boss_conversion_rate", "average_winning_hp", "losses", "timeouts", "stalls")}))
    atomic_json(root / "comparison.json", dict(runs=rows))
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def execute(root: Path):
    """Each trainer is a child process; wait for its full evaluation before advancing."""
    root = root.resolve()
    lock = root / "queue.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError("This queue already has an owner (queue.lock). Refusing a duplicate launch.")
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    manifest = read_json(root / "batch.json")
    child = None
    try:
        if not manifest or manifest["status"] != "queued":
            raise RuntimeError("Only a freshly prepared queue can be started; existing work is never restarted implicitly.")
        manifest.update(status="running", pid=os.getpid(), started_utc=now())
        atomic_json(root / "batch.json", manifest)
        for entry in manifest["runs"]:
            while read_json(root / "queue-control.json", {}).get("pause_after_current"):
                if manifest["status"] != "paused":
                    manifest["status"] = "paused"
                    atomic_json(root / "batch.json", manifest)
                time.sleep(1)
            manifest["status"] = "running"
            entry.update(status="running", started_utc=now())
            atomic_json(root / "batch.json", manifest)
            with (root / f"{entry['profile']}.log").open("w", encoding="utf-8") as log:
                child = subprocess.Popen([sys.executable, "-u", "-m", "sts2_env.training.runner", "--config", entry["config"]],
                                         cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT)
                entry["pid"] = child.pid
                atomic_json(root / "batch.json", manifest)
                code = child.wait()
            directory = experiment_directory(root, entry)
            completed = code == 0 and directory and (directory / "final-results.json").exists() and (directory / "best-results.json").exists()
            entry.update(status="completed" if completed else "failed", exit_code=code,
                         finished_utc=now(), experiment=str(directory) if directory else None)
            atomic_json(root / "batch.json", manifest)
            write_comparison(root, manifest)
            if not completed:
                raise RuntimeError(f"{entry['profile']} did not finish successfully. Queue stopped; see {entry['profile']}.log.")
        manifest.update(status="completed", finished_utc=now())
        atomic_json(root / "batch.json", manifest)
    except BaseException as exc:
        if child and child.poll() is None:
            # This exceptional path is recorded as a failure, never as completed training.
            child.terminate()
            child.wait()
        if manifest:
            manifest.update(status="failed", error=str(exc), finished_utc=now())
            atomic_json(root / "batch.json", manifest)
        traceback.print_exc()
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.root, smoke=args.smoke)
        print(args.root.resolve())
    else:
        execute(args.root)


if __name__ == "__main__":
    main()

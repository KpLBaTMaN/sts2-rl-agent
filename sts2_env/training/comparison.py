"""Prepare and execute a durable, strictly sequential queue of training runs.

By default the queue compares the sparse, shaped and survival profiles. With
``--plan`` it runs the preset files a plan lists, in order (for example the
reward ablation stages in ``local/config/presets/reward-ablation/``).
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

from sts2_env.training.telemetry import atomic_json

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILES = ("sparse", "shaped", "survival")
# A smoke queue checks the whole pipeline in seconds; its results mean nothing.
SMOKE_TRAINING = dict(total_timesteps=32, n_envs=2, n_steps=16, batch_size=16,
                      n_epochs=1, max_episode_steps=16, checkpoint_every=16)
SMOKE_EVALUATION = dict(every_steps=32, episodes=2, final_episodes=2, random_baseline_episodes=0)


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def now():
    return datetime.now(timezone.utc).isoformat()


def comparison_configs():
    configs = []
    for profile in PROFILES:
        configs.append(dict(name=f"long-{profile}", task="act1", character="Ironclad", ascension=0,
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
                                      random_baseline_episodes=50)))
    return "SpireFormer · Reward comparison", [(profile, None, config) for profile, config in zip(PROFILES, configs)]


def plan_configs(plan_path):
    """Read a plan: ``{"title": ..., "runs": [preset paths relative to the plan file]}``."""
    from sts2_env.training.config import load_config
    plan_path = Path(plan_path).resolve()
    plan = read_json(plan_path)
    if not isinstance(plan, dict) or not isinstance(plan.get("runs"), list) or not plan["runs"]:
        raise ValueError(f"{plan_path} must be a JSON object with a non-empty runs list")
    configs = []
    for item in plan["runs"]:
        source = (plan_path.parent / item).resolve()
        if not source.exists():
            raise ValueError(f"{plan_path.name} lists a missing preset: {source}")
        config = load_config(source)
        configs.append((config["name"], str(source), config))
    # Run names key the logs and run directories, so each must be unique in a queue.
    names = [name for name, _, _ in configs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"{plan_path.name} repeats run names: {', '.join(duplicates)}")
    return plan.get("title", plan_path.stem), configs


def prepare(root: Path, *, smoke=False, plan=None):
    from sts2_env.training.config import load_config
    title, configs = plan_configs(plan) if plan else comparison_configs()
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    entries = []
    for key, source, config in configs:
        config = dict(config, output_root=str(root / "runs"))
        if smoke:
            config["name"] = f"smoke-{config['name'] if plan else key}"
            config["training"] = dict(config["training"], **SMOKE_TRAINING)
            config["evaluation"] = dict(config["evaluation"], **SMOKE_EVALUATION)
        path = root / "configs" / f"{key}.json"
        atomic_json(path, config)
        resolved = load_config(path)
        atomic_json(path, resolved)  # Freeze the exact reward weights, not just a mutable profile name.
        entry = dict(profile=key, name=config["name"], config=str(path), status="queued")
        if source:
            entry["source"] = source
        entries.append(entry)
    manifest = dict(title=title, created_utc=now(), status="queued",
                    smoke=smoke, runs=entries, pid=None)
    atomic_json(root / "batch.json", manifest)
    atomic_json(root / "queue-control.json", dict(pause_after_current=False))
    return manifest


def experiment_directory(root, entry):
    # The runner names directories <name>-<UTC stamp>-<6 hex>; match exactly so
    # "screen-shaped" never picks up a run called "screen-shaped-2".
    pattern = re.compile(re.escape(entry["name"]) + r"-\d{8}-\d{6}-[0-9a-f]{6}")
    directories = sorted(p for p in (root / "runs").glob(entry["name"] + "-*") if pattern.fullmatch(p.name))
    return directories[-1] if directories else None


def write_comparison(root, manifest):
    rows = []
    for entry in manifest["runs"]:
        directory = experiment_directory(root, entry)
        summary = read_json(directory / "best-results.json", {}) if directory else {}
        averages = summary.get("averages") or {}
        rows.append(dict(profile=entry["profile"], status=entry["status"],
                         **{key: summary.get(key) for key in ("episodes", "wins", "win_rate", "average_floor",
                            "boss_reached_rate", "boss_conversion_rate", "average_winning_hp", "losses", "timeouts", "stalls")},
                         upgraded_cards=averages.get("upgraded_cards"), turns_per_combat=averages.get("turns_per_combat")))
    atomic_json(root / "comparison.json", dict(runs=rows))
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def execute(root: Path, *, resume=False):
    """Each trainer is a child process; wait for its full evaluation before advancing.

    ``resume`` continues a queue that stopped (failure, crash or reboot): completed
    runs are kept, and the run that was interrupted starts again from scratch.
    """
    root = root.resolve()
    lock = root / "queue.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError("This queue already has an owner (queue.lock). Refusing a duplicate launch. "
                           "If that process is no longer running, delete queue.lock and use --resume.")
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    manifest = read_json(root / "batch.json")
    child = None
    try:
        if not manifest:
            raise RuntimeError(f"No queue at {root} (batch.json is missing); prepare it first.")
        if resume:
            if manifest["status"] == "completed":
                print("Every run in this queue has already completed.")
                return
            for entry in manifest["runs"]:
                if entry["status"] in ("running", "failed"):
                    # Keep the abandoned directory on record; the rerun gets a fresh one.
                    directory = entry.get("experiment") or experiment_directory(root, entry)
                    if directory:
                        entry.setdefault("abandoned", []).append(str(directory))
                    entry.update(status="queued", experiment=None)
        elif manifest["status"] != "queued":
            raise RuntimeError("Only a freshly prepared queue can be started; existing work is never restarted implicitly. "
                               "Use --resume to continue a stopped queue.")
        manifest.update(status="running", pid=os.getpid(), started_utc=now(), error=None)
        atomic_json(root / "batch.json", manifest)
        for entry in manifest["runs"]:
            if entry["status"] == "completed":
                continue
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
    parser.add_argument("--prepare", action="store_true", help="Create a new queue directory at --root")
    parser.add_argument("--plan", type=Path, help="With --prepare: queue the presets this plan file lists")
    parser.add_argument("--smoke", action="store_true", help="With --prepare: shrink every run to a seconds-long check")
    parser.add_argument("--resume", action="store_true", help="Continue a stopped queue, keeping completed runs")
    args = parser.parse_args()
    if args.prepare:
        prepare(args.root, smoke=args.smoke, plan=args.plan)
        print(args.root.resolve())
    else:
        execute(args.root, resume=args.resume)


if __name__ == "__main__":
    main()

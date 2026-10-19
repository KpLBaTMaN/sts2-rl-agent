"""Trained runs under output/ and their evaluation results, for comparing in the Model player."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output"
RESULT_FILES = {"best": "best-results.json", "final": "final-results.json", "latest": "latest-evaluation.json"}
# A run whose telemetry is older than this is no longer training.
LIVE_SECONDS = 600


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def model_file(directory: Path, entity: bool, which: str) -> Path:
    extension = "pt" if entity else "zip"
    return directory / (f"final_model.{extension}" if which == "final" else f"best_model/best_model.{extension}")


def seed_label(evaluation: dict, which: str) -> str:
    """Seeds behind a result: the final pool for best/final, the validation pool otherwise."""
    if evaluation.get("mode") == "training_seeds":
        return "training seeds"
    final = which != "latest" and evaluation.get("final_seeds")
    pool = final or evaluation.get("seeds") or []
    count = evaluation.get("final_episodes" if which != "latest" else "episodes", len(pool))
    used = [pool[i % len(pool)] for i in range(count)] if pool else []
    if not used:
        return "—"
    low, high = min(used), max(used)
    return f"{low}–{high}" if len(set(used)) == high - low + 1 else f"{len(set(used))} seeds"


def summarize(summary: dict | None) -> dict | None:
    if not summary:
        return None
    averages = summary.get("averages") or {}
    return dict(episodes=summary.get("episodes"), wins=summary.get("wins"), win_rate=summary.get("win_rate"),
                losses=summary.get("losses"), timeouts=summary.get("timeouts"), stalls=summary.get("stalls"),
                average_floor=summary.get("average_floor"), boss_reached_rate=summary.get("boss_reached_rate"),
                boss_conversion_rate=summary.get("boss_conversion_rate"),
                average_decisions=summary.get("average_decisions"),
                upgraded_cards=averages.get("upgraded_cards"), rest_sites=averages.get("rest_sites"),
                deck_size=averages.get("deck_size"))


def status_of(directory: Path, live: dict | None) -> str:
    if (directory / RESULT_FILES["final"]).exists():
        return "completed"
    if any(directory.glob("interrupted_model.*")):
        return "interrupted"
    if live and live.get("kind") not in {"completed", "interrupted"}:
        try:
            updated = datetime.fromisoformat(live["updated_utc"])
            if (datetime.now(timezone.utc) - updated).total_seconds() < LIVE_SECONDS:
                return "training"
        except (KeyError, ValueError):
            pass
    return "stopped"


def describe(directory: Path, config: dict, output: Path) -> dict:
    entity = config.get("model", {}).get("type", "flat") == "entity"
    training = config.get("training", {})
    evaluation = config.get("evaluation", {})
    reward = config.get("reward") if isinstance(config.get("reward"), dict) else {}
    live = read_json(directory / "live.json")
    return dict(
        id=directory.relative_to(output).as_posix(),
        group=directory.parent.relative_to(output).as_posix(),
        name=config.get("name", directory.name),
        status=status_of(directory, live),
        steps=(live or {}).get("steps"), target_steps=training.get("total_timesteps"),
        policy="SpireFormer" if entity else "Flat MLP", entity=entity,
        task=config.get("task"), character=config.get("character"), ascension=config.get("ascension"),
        start_with_neow=config.get("start_with_neow", False),
        # Runs from before reward profiles trained on the sparse result only.
        reward=(dict(profile=reward.get("profile", "custom"), hp=reward.get("hp", 0),
                     upgrade=reward.get("upgrade", 0), win=reward.get("win", 0)) if reward
                else dict(profile="sparse (legacy)", hp=None, upgrade=None, win=1.0)),
        model_seed=training.get("model_seed"), gamma=training.get("gamma"),
        models={which: model_file(directory, entity, which).exists() for which in ("best", "final")},
        results={which: summarize(read_json(directory / name)) for which, name in RESULT_FILES.items()},
        seeds={which: seed_label(evaluation, which) for which in RESULT_FILES},
        path=str(directory),
    )


def discover_runs(output: Path | None = None) -> list[dict]:
    """Act 1 and full-run experiments by folder; smoke tests and combat-only runs are left out."""
    output = output or OUTPUT
    runs = []
    for config_path in output.rglob("config.json"):
        directory = config_path.parent
        if directory.name.startswith("smoke") or not (directory / "metadata.json").exists():
            continue
        config = read_json(config_path)
        if not isinstance(config, dict) or config.get("task") not in {"act1", "full_run"}:
            continue
        runs.append(describe(directory, config, output))
    return sorted(runs, key=lambda run: (run["group"], run["id"]))


def find_run(run_id: str, output: Path | None = None) -> dict | None:
    return next((run for run in discover_runs(output) if run["id"] == run_id), None)


def playable_reason(run: dict, live: dict, which: str) -> str | None:
    """Why *run* cannot replace the live policy, or None when it can."""
    if not run["models"].get(which):
        return f"No {which} model saved yet"
    for key in ("task", "character", "ascension", "start_with_neow"):
        if run[key] != live[key]:
            return f"Trained for a different {key.replace('_', ' ')} ({run[key]})"
    return None


def format_table(runs: list[dict], which: str = "best") -> str:
    """Plain-text (ASCII) comparison for the terminal."""
    def number(value, pattern="{:.2f}"):
        return "-" if value is None else pattern.format(value)

    def pct(value):
        return number(None if value is None else 100 * value, "{:.1f}%")

    header = (f"{'Run':34} {'Status':10} {'Reward':18} {'Wins':>8} {'Floor':>6} {'Boss':>6} {'Beat':>6} "
              f"{'T/O':>4} {'Stall':>5} {'Upg':>5}  Seeds")
    lines = [f"Results: {which} model (Beat = wins / boss visits)", header, "-" * len(header)]
    for run in runs:
        reward = run["reward"]
        reward_text = (reward["profile"] if reward["hp"] is None
                       else f"{reward['profile']} hp{reward['hp']:g} up{reward['upgrade']:g}")
        prefix = f"{run['name'][:34]:34} {run['status']:10} {reward_text[:18]:18}"
        result = run["results"].get(which)
        if not result:
            lines.append(f"{prefix} {'no results yet':>14}")
            continue
        wins = f"{result['wins']}/{result['episodes']}"
        seeds = run["seeds"][which].replace("–", "-").replace("—", "-")
        lines.append(f"{prefix} {wins:>8} {number(result['average_floor']):>6} {pct(result['boss_reached_rate']):>6} "
                     f"{pct(result['boss_conversion_rate']):>6} {number(result['timeouts'], '{}'):>4} "
                     f"{number(result['stalls'], '{}'):>5} {number(result['upgraded_cards']):>5}  {seeds}")
    return "\n".join(lines)

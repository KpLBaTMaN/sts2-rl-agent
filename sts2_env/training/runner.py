"""Run an experiment: python -m sts2_env.training.runner --config local/config/training.json."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from uuid import uuid4

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import torch

from sts2_env.training.config import load_config
from sts2_env.training.environment import make_env

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def evaluate(config, episodes, model=None):
    """Evaluate with explicit success flags; timeouts are never counted as wins."""
    env = make_env(config, evaluation=True)
    rng = np.random.default_rng(config["training"]["model_seed"] + 777)
    rows = []
    try:
        for episode in range(episodes):
            obs, _ = env.reset()
            score = 0.0
            while True:
                masks = env.env.action_masks()
                if model is None:
                    action = int(rng.choice(np.flatnonzero(masks)))
                else:
                    action, _ = model.predict(obs, action_masks=masks,
                                              deterministic=config["evaluation"]["deterministic"])
                    action = int(action)
                obs, reward, terminated, truncated, info = env.step(action)
                score += reward
                if terminated or truncated:
                    rows.append(dict(episode=episode + 1, seed=info["run_seed"],
                                     win=info["is_success"], floor=info["floor"],
                                     hp=info["remaining_hp"], decisions=info["decisions"],
                                     boss_reached=info["boss_reached"],
                                     death_room=info["death_room"], timed_out=info["timed_out"], reward=score,
                                     **{f"reward_{term}": value for term, value in info["reward_terms"].items()},
                                     **info.get("run_stats", {})))
                    break
    finally:
        env.close()
    wins = sum(row["win"] for row in rows)
    boss_attempts = sum(row["boss_reached"] for row in rows)
    winning_hp = [row["hp"] for row in rows if row["win"]]
    deaths = {}
    for row in rows:
        if row["death_room"]:
            deaths[row["death_room"]] = deaths.get(row["death_room"], 0) + 1
    summary = dict(episodes=episodes, wins=wins, win_rate=wins / episodes,
                   distinct_seeds=len({r["seed"] for r in rows}),
                   boss_reached_rate=boss_attempts / episodes,
                   boss_conversion_rate=wins / boss_attempts if boss_attempts else None,
                   average_winning_hp=float(np.mean(winning_hp)) if winning_hp else None,
                   average_floor=float(np.mean([r["floor"] for r in rows])),
                   max_floor=max(r["floor"] for r in rows),
                   average_decisions=float(np.mean([r["decisions"] for r in rows])),
                   average_reward=float(np.mean([r["reward"] for r in rows])),
                   timeouts=sum(r["timed_out"] for r in rows), deaths_by_room=deaths,
                   average_reward_terms={key[7:]: float(np.mean([r[key] for r in rows]))
                                         for key in rows[0] if key.startswith("reward_")})
    summary["losses"] = episodes - wins - summary["timeouts"]
    if config["task"] == "combat":
        summary["boss_reached_rate"] = None
        summary["boss_conversion_rate"] = None
    else:
        summary.update(run_summary(rows))
    return summary, rows


RUN_AVERAGES = ("act", "gold", "max_hp", "deck_size", "upgraded_cards", "attacks", "skills", "powers",
                "curses_statuses", "relic_count", "potions_held", "card_removals",
                "elites", "monsters", "shops", "rest_sites", "events", "treasures")
OUTCOME_AVERAGES = ("gold", "deck_size", "relic_count", "upgraded_cards", "floor")


def run_summary(rows):
    """End-of-run averages overall and split by outcome, so wins can be compared with losses."""
    def mean(group, key):
        return float(np.mean([r[key] for r in group])) if group else None
    wins = [r for r in rows if r["win"]]
    losses = [r for r in rows if not r["win"] and not r["timed_out"]]
    timeouts = [r for r in rows if r["timed_out"]]
    return dict(averages={key: mean(rows, key) for key in RUN_AVERAGES},
                by_outcome={name: {key: mean(group, key) for key in OUTCOME_AVERAGES}
                            for name, group in (("wins", wins), ("losses", losses), ("timeouts", timeouts))})


def item_stats(rows, column):
    """Per relic or card: how often it ended in the run and the win rate with and without it."""
    items = [[item.rstrip("+") for item in r[column].split(";") if item] for r in rows]
    stats = []
    for item in sorted(set().union(*items)):
        with_item = [r["win"] for r, held in zip(rows, items) if item in held]
        without = [r["win"] for r, held in zip(rows, items) if item not in held]
        stats.append(dict(item=item, runs_with=len(with_item), pick_rate=len(with_item) / len(rows),
                          avg_copies=sum(held.count(item) for held in items) / len(with_item),
                          win_rate_with=float(np.mean(with_item)),
                          win_rate_without=float(np.mean(without)) if without else None))
    return sorted(stats, key=lambda row: (-row["runs_with"], row["item"]))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_summary(summary):
    text = (f"Final: {summary['wins']}/{summary['episodes']} wins ({summary['win_rate']:.1%}), "
            f"{summary['losses']} losses, {summary['timeouts']} timeouts")
    if "averages" in summary:
        avg = summary["averages"]
        text += (f"\n  avg act {avg['act']:.2f} | floor {summary['average_floor']:.2f} | gold {avg['gold']:.0f} "
                 f"| deck {avg['deck_size']:.1f} ({avg['upgraded_cards']:.1f} upgraded) "
                 f"| relics {avg['relic_count']:.1f} | elites {avg['elites']:.1f}")
    return text


def save_evaluation(directory, label, summary, rows):
    (directory / f"{label}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_csv(directory / f"{label}-episodes.csv", rows)
    for column, name in (("relics", "relic-stats"), ("deck", "card-stats")):
        stats = item_stats(rows, column) if column in rows[0] else []
        if stats:
            write_csv(directory / f"{label}-{name}.csv", stats)


def run(config, config_path):
    train, evaluation = config["training"], config["evaluation"]
    torch.set_num_threads(train["torch_threads"])
    if train["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this installation; set training.device to cpu")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    directory = project_path(config["output_root"]) / f"{config['name']}-{stamp}-{uuid4().hex[:6]}"
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    metadata = {"source_config": str(config_path), "started_utc": stamp,
                "objective": config["task"], "evaluation_mode": evaluation["mode"],
                "seed_note": "Simulator seeds are integers, not Steam's displayed seed strings.",
                "model": config["model"]["type"],
                "observation_note": ("Entity lists: hand, piles, enemies, powers, potions, relics and pending choices."
                                     if config["model"]["type"] == "entity" else
                                     "Uses existing simulator observations; noncombat choice/deck information is limited.")}
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Experiment: {directory}", flush=True)
    print(f"Task: {config['task']} | Character: {config['character']} | Seeds: {config['seeds']}", flush=True)
    print(f"Evaluation: {evaluation['mode']} | deterministic={evaluation['deterministic']}", flush=True)
    weights = {k: v for k, v in config["reward"].items() if k not in ("profile", "description") and v}
    print(f"Reward: {config['reward']['profile']} {weights}", flush=True)
    if evaluation["mode"] == "training_seeds" and config["seeds"]["mode"] == "fixed":
        print("Fixed-seed deterministic evaluation repeats one trajectory; it does not measure generalisation.", flush=True)
    if evaluation["random_baseline_episodes"]:
        summary, rows = evaluate(config, evaluation["random_baseline_episodes"])
        save_evaluation(directory, "random-baseline", summary, rows)
        print(f"Random baseline: {summary['wins']}/{summary['episodes']} wins", flush=True)

    if config["model"]["type"] == "entity":
        from sts2_env.entity.ppo import train_entity
        train_entity(config, directory, evaluate, save_evaluation, project_path)
        return directory

    factories = [lambda i=i: make_env(config, worker=i) for i in range(train["n_envs"])]
    env = SubprocVecEnv(factories) if train["n_envs"] > 1 else DummyVecEnv(factories)
    eval_env = DummyVecEnv([lambda: make_env(config, evaluation=True)])
    model = None
    started = time.perf_counter()
    interrupted = False
    try:
        if config["resume_from"]:
            model = MaskablePPO.load(project_path(config["resume_from"]), env=env,
                                    device=train["device"], tensorboard_log=str(directory / "tb_logs"),
                                    learning_rate=train["learning_rate"], ent_coef=train["ent_coef"],
                                    seed=train["model_seed"])
            # Keep resume controls honest instead of silently retaining saved hyperparameters.
            for key in ("n_steps", "batch_size", "n_epochs", "gamma"):
                if getattr(model, key) != train[key]:
                    raise ValueError(f"Resume requires training.{key}={getattr(model, key)} to match the saved model")
            saved_sizes = model.policy_kwargs.get("net_arch")
            if saved_sizes != dict(pi=train["hidden_sizes"], vf=train["hidden_sizes"]):
                raise ValueError("Resume hidden_sizes must match the saved model")
        else:
            model = MaskablePPO("MlpPolicy", env, device=train["device"], seed=train["model_seed"],
                                learning_rate=train["learning_rate"], n_steps=train["n_steps"],
                                batch_size=train["batch_size"], n_epochs=train["n_epochs"],
                                gamma=train["gamma"], ent_coef=train["ent_coef"], verbose=1,
                                tensorboard_log=str(directory / "tb_logs"),
                                policy_kwargs=dict(net_arch=dict(pi=train["hidden_sizes"], vf=train["hidden_sizes"])))
        callbacks = [
            MaskableEvalCallback(eval_env, best_model_save_path=str(directory / "best_model"),
                                 log_path=str(directory / "eval_logs"),
                                 eval_freq=max(evaluation["every_steps"] // train["n_envs"], 1),
                                 n_eval_episodes=evaluation["episodes"], deterministic=evaluation["deterministic"]),
            CheckpointCallback(save_freq=max(train["checkpoint_every"] // train["n_envs"], 1),
                               save_path=str(directory / "checkpoints")),
        ]
        initial_steps = model.num_timesteps
        try:
            model.learn(train["total_timesteps"], callback=callbacks, progress_bar=True,
                        reset_num_timesteps=not bool(config["resume_from"]))
        except KeyboardInterrupt:
            interrupted = True
            print("Interrupted; saving the current model.", flush=True)
        model.save(directory / ("interrupted_model" if interrupted else "final_model"))
        if interrupted:
            return directory
        summary, rows = evaluate(config, evaluation["final_episodes"], model=model)
        summary.update(training_seconds=time.perf_counter() - started,
                       requested_timesteps=train["total_timesteps"],
                       actual_new_timesteps=model.num_timesteps - initial_steps,
                       model="final_model.zip", evaluation_mode=evaluation["mode"])
        save_evaluation(directory, "final-results", summary, rows)
        best = directory / "best_model/best_model.zip"
        if best.exists():
            best_model = MaskablePPO.load(best, device=train["device"])
            best_summary, best_rows = evaluate(config, evaluation["final_episodes"], best_model)
            save_evaluation(directory, "best-results", best_summary, best_rows)
        print(f"\n{format_summary(summary)}", flush=True)
        print(f"Results and models: {directory}", flush=True)
        return directory
    finally:
        env.close()
        eval_env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="local/config/training.json")
    parser.add_argument("--check", action="store_true", help="Validate config and reset an environment without training")
    args = parser.parse_args()
    path = project_path(args.config)
    try:
        config = load_config(path)
        if args.check:
            env = make_env(config)
            try:
                obs, _ = env.reset()
                print(json.dumps(config, indent=2))
                shape = obs.shape if hasattr(obs, "shape") else f"{len(obs)} entity arrays"
                scenario = env.env
                while not hasattr(scenario, "episode_seed"):
                    scenario = scenario.env
                print(f"Validated: observation {shape}; actions {env.action_space}; seed {scenario.episode_seed}")
            finally:
                env.close()
        else:
            run(config, path)
    except (ValueError, FileNotFoundError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Configuration error: {exc}\n")


if __name__ == "__main__":
    main()

"""Strict JSON configuration; game seeds and model seeds are independent."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

DEFAULTS = {
    "name": "act1-fixed",
    "task": "act1",
    "character": "Ironclad",
    "ascension": 0,
    "start_with_neow": False,
    "seeds": {"mode": "fixed", "values": [42], "schedule_seed": 123},
    "training": {
        "total_timesteps": 500000, "n_envs": 4, "model_seed": 42,
        "learning_rate": 0.0003, "n_steps": 2048, "batch_size": 256,
        "n_epochs": 10, "gamma": 0.995, "ent_coef": 0.02,
        "hidden_sizes": [256, 256], "device": "cpu", "torch_threads": 4,
        "max_episode_steps": 2000, "max_combat_turns": 200,
        "checkpoint_every": 100000,
    },
    "evaluation": {
        "mode": "training_seeds", "seeds": list(range(10000, 10100)),
        "every_steps": 20000, "episodes": 20, "final_episodes": 100,
        "deterministic": True, "random_baseline_episodes": 20,
    },
    "model": {"type": "entity", "d_model": 128, "layers": 3, "heads": 4},
    "output_root": "output/experiments",
    "resume_from": None,
}


def _merge(base, supplied, prefix=""):
    if not isinstance(supplied, dict):
        raise ValueError(f"{prefix or 'config'} must be an object")
    for key, value in supplied.items():
        if key not in base:
            raise ValueError(f"Unknown configuration key: {prefix}{key}")
        if isinstance(base[key], dict):
            _merge(base[key], value, f"{prefix}{key}.")
        else:
            base[key] = value


def _integer(value, name, low=0, high=2147483647):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer between {low} and {high}")


def _seed_list(values, name):
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a nonempty list of integer simulator seeds")
    for value in values:
        _integer(value, name)


def load_config(path: str | Path) -> dict:
    config = copy.deepcopy(DEFAULTS)
    _merge(config, json.loads(Path(path).read_text(encoding="utf-8-sig")))
    if config["task"] not in {"combat", "act1", "full_run"}:
        raise ValueError("task must be combat, act1, or full_run")
    name = config["name"]
    if not isinstance(name, str) or not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("name must contain only letters, numbers, underscores, or hyphens")
    if config["character"] not in {"Ironclad", "Silent", "Defect", "Necrobinder", "Regent"}:
        raise ValueError("Unknown character")
    _integer(config["ascension"], "ascension", 0, 9)
    if type(config["start_with_neow"]) is not bool:
        raise ValueError("start_with_neow must be true or false")
    if config["task"] == "combat" and (config["character"] != "Ironclad" or config["ascension"] != 0 or config["start_with_neow"]):
        raise ValueError("combat currently supports Ironclad, ascension 0, without Neow only")
    seed = config["seeds"]
    if seed["mode"] not in {"fixed", "pool", "random"}:
        raise ValueError("seeds.mode must be fixed, pool, or random")
    _seed_list(seed["values"], "seeds.values")
    _integer(seed["schedule_seed"], "seeds.schedule_seed")
    if seed["mode"] == "fixed" and len(seed["values"]) != 1:
        raise ValueError("fixed mode requires exactly one seed")
    train, evaluation = config["training"], config["evaluation"]
    for key in ("total_timesteps", "n_envs", "n_steps", "batch_size", "n_epochs", "torch_threads", "max_episode_steps", "max_combat_turns", "checkpoint_every"):
        _integer(train[key], f"training.{key}", 1)
    _integer(train["model_seed"], "training.model_seed")
    if train["n_steps"] * train["n_envs"] < train["batch_size"] or train["n_steps"] * train["n_envs"] % train["batch_size"]:
        raise ValueError("batch_size must divide n_steps * n_envs")
    for key, low, high in (("learning_rate", 0, 1), ("gamma", 0, 1), ("ent_coef", 0, 10)):
        value = train[key]
        if type(value) not in (float, int) or not math.isfinite(value) or not low <= value <= high or (key != "ent_coef" and value == 0):
            raise ValueError(f"Invalid training.{key}")
    if not isinstance(train["hidden_sizes"], list) or not train["hidden_sizes"]:
        raise ValueError("hidden_sizes must be a nonempty list")
    for size in train["hidden_sizes"]:
        _integer(size, "hidden_sizes", 1, 4096)
    if train["device"] not in {"cpu", "cuda", "auto"}:
        raise ValueError("device must be cpu, cuda, or auto")
    if evaluation["mode"] not in {"training_seeds", "seed_list", "unseen"}:
        raise ValueError("evaluation.mode must be training_seeds, seed_list, or unseen")
    _seed_list(evaluation["seeds"], "evaluation.seeds")
    for key in ("every_steps", "episodes", "final_episodes"):
        _integer(evaluation[key], f"evaluation.{key}", 1)
    _integer(evaluation["random_baseline_episodes"], "evaluation.random_baseline_episodes")
    if type(evaluation["deterministic"]) is not bool:
        raise ValueError("evaluation.deterministic must be true or false")
    if evaluation["mode"] == "unseen" and seed["mode"] != "random" and set(seed["values"]) & set(evaluation["seeds"]):
        raise ValueError("Unseen evaluation seeds overlap training seeds")
    spec = config["model"]
    if spec["type"] not in {"flat", "entity"}:
        raise ValueError("model.type must be flat or entity")
    if spec["type"] == "entity":
        _integer(spec["d_model"], "model.d_model", 8, 2048)
        _integer(spec["layers"], "model.layers", 1, 24)
        _integer(spec["heads"], "model.heads", 1, 64)
        if spec["d_model"] % spec["heads"]:
            raise ValueError("model.d_model must be divisible by model.heads")
    if not isinstance(config["output_root"], str) or not config["output_root"].strip():
        raise ValueError("output_root must be a directory path")
    if config["resume_from"] is not None and not isinstance(config["resume_from"], str):
        raise ValueError("resume_from must be a saved model path or null")
    return config

"""Queue failure handling, live progress, and final holdout isolation."""
import copy
import json
from pathlib import Path
import subprocess

import pytest

from sts2_env.training.comparison import execute, prepare, read_json
from sts2_env.training.config import DEFAULTS, load_config
from sts2_env.training.dashboard import snapshot
from sts2_env.training.environment import ScenarioEnv
from sts2_env.training.runner import final_evaluation_config
from sts2_env.training.telemetry import publish


def test_comparison_freezes_profiles_and_matches_other_settings(tmp_path):
    root = tmp_path / "batch"
    batch = prepare(root)
    configs = [load_config(e["config"]) for e in batch["runs"]]
    for config in configs:
        assert config["training"]["total_timesteps"] == 5_000_000
        assert config["training"]["checkpoint_every"] == 100_000
        assert len(config["evaluation"]["final_seeds"]) == 200
    comparable = [{k: v for k, v in c.items() if k not in ("name", "reward")} for c in configs]
    assert comparable[0] == comparable[1] == comparable[2]
    assert snapshot(root)["runs"][0]["config"]["reward"]["profile"] == "sparse"


def test_failed_training_does_not_advance_queue(tmp_path, monkeypatch):
    root = tmp_path / "batch"
    prepare(root, smoke=True)
    children = []
    class Failure:
        pid = 123
        def __init__(self, *args, **kwargs):
            children.append(args)
        def wait(self):
            return 1
        def poll(self):
            return 1
    monkeypatch.setattr(subprocess, "Popen", Failure)
    with pytest.raises(RuntimeError, match="did not finish"):
        execute(root)
    batch = read_json(root / "batch.json")
    assert batch["status"] == "failed"
    assert [r["status"] for r in batch["runs"]] == ["failed", "queued", "queued"]
    assert len(children) == 1
    assert not (root / "queue.lock").exists()


def test_final_holdout_is_reserved_and_does_not_change_periodic_config(tmp_path):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["seeds"]["mode"] = "random"
    cfg["evaluation"].update(mode="unseen", seeds=[1, 2], final_seeds=[3, 4], final_episodes=2)
    path = tmp_path / "c.json"
    path.write_text(json.dumps(cfg))
    loaded = load_config(path)
    final = final_evaluation_config(loaded)
    assert loaded["evaluation"]["seeds"] == [1, 2]
    assert final["evaluation"]["seeds"] == [3, 4]
    # Force the RNG to draw reserved periodic and final seeds before an allowed one.
    class Sequence:
        values = iter([1, 3, 4, 2, 5])
        def integers(self, *_):
            return next(self.values)
    env = ScenarioEnv(loaded)
    env.sequence = Sequence()
    assert env._next_seed() == 5
    env.close()
    cfg["evaluation"]["final_seeds"] = [2, 3]
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="separate"):
        load_config(path)


def test_partial_telemetry_append_does_not_break_snapshot(tmp_path):
    root = tmp_path / "batch"
    manifest = prepare(root, smoke=True)
    directory = root / "runs" / (manifest["runs"][0]["name"] + "-test")
    directory.mkdir(parents=True)
    publish(directory, "rollout", steps=32, train_win_rate=0.5)
    with (directory / "metrics.jsonl").open("a") as stream:
        stream.write('{"kind":')
    state = snapshot(root)
    assert state["runs"][0]["rollouts"][0]["steps"] == 32
    assert state["runs"][0]["live"]["train_win_rate"] == 0.5

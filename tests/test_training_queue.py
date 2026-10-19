"""Plan queues run preset files in order, and a stopped queue resumes without redoing work."""
import json
from pathlib import Path
import subprocess

import pytest

from sts2_env.training.comparison import PROJECT_ROOT, execute, experiment_directory, prepare, read_json
from sts2_env.training.config import load_config

ABLATION = PROJECT_ROOT / "local" / "config" / "presets" / "reward-ablation"
STAGE1 = ABLATION / "queue-stage1.json"


@pytest.mark.parametrize("plan", ["queue-stage1.json", "queue-stage2-baseline.json"])
def test_ablation_plans_freeze_their_presets_in_order(tmp_path, plan):
    root = tmp_path / "queue"
    batch = prepare(root, plan=ABLATION / plan)
    sources = [ABLATION / item for item in read_json(ABLATION / plan)["runs"]]
    assert [entry["source"] for entry in batch["runs"]] == [str(s.resolve()) for s in sources]
    for entry, source in zip(batch["runs"], sources):
        frozen, original = load_config(entry["config"]), load_config(source)
        assert frozen["output_root"] == str(root / "runs") and entry["name"] == original["name"]
        assert {k: v for k, v in frozen.items() if k != "output_root"} == \
            {k: v for k, v in original.items() if k != "output_root"}


def test_stage1_arms_differ_only_in_their_reward():
    configs = [load_config(ABLATION / item) for item in read_json(STAGE1)["runs"]]
    shared = [{k: v for k, v in c.items() if k not in ("name", "reward", "reward_overrides", "evaluation")}
              for c in configs]
    assert all(s == shared[0] for s in shared)
    turn = {c["name"]: c["reward"]["combat_turn"] for c in configs}
    assert turn.pop("screen-turn-cost") == -0.005 and set(turn.values()) == {0}


def test_plan_rejects_repeated_names_and_missing_presets(tmp_path):
    plan = tmp_path / "plan.json"
    shaped = str(ABLATION / "1-screen-shaped.json")
    plan.write_text(json.dumps({"runs": [shaped, shaped]}))
    with pytest.raises(ValueError, match="repeats run names: screen-shaped"):
        prepare(tmp_path / "a", plan=plan)
    plan.write_text(json.dumps({"runs": ["missing.json"]}))
    with pytest.raises(ValueError, match="missing preset"):
        prepare(tmp_path / "b", plan=plan)


def test_resume_keeps_completed_runs_and_restarts_the_interrupted_one(tmp_path, monkeypatch):
    root = tmp_path / "queue"
    batch = prepare(root, plan=STAGE1, smoke=True)
    names = [entry["name"] for entry in batch["runs"]]
    # A crash during the second run: the first finished, the second left a partial directory.
    partial = root / "runs" / f"{names[1]}-20261019-000000-aaaaaa"
    partial.mkdir(parents=True)
    batch["status"] = "running"
    batch["runs"][0]["status"], batch["runs"][1]["status"] = "completed", "running"
    (root / "batch.json").write_text(json.dumps(batch))
    launched = []

    class FinishedRun:
        pid = 1

        def __init__(self, command, **kwargs):
            config = json.loads(Path(command[command.index("--config") + 1]).read_text(encoding="utf-8"))
            directory = Path(config["output_root"]) / f"{config['name']}-20261019-120000-{len(launched):06x}"
            directory.mkdir(parents=True)
            for name in ("final-results.json", "best-results.json"):
                (directory / name).write_text(json.dumps({"wins": 0, "episodes": 2}))
            launched.append(config["name"])

        def wait(self):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(subprocess, "Popen", FinishedRun)
    with pytest.raises(RuntimeError, match="--resume"):
        execute(root)
    execute(root, resume=True)
    assert launched == names[1:]
    batch = read_json(root / "batch.json")
    assert batch["status"] == "completed" and all(r["status"] == "completed" for r in batch["runs"])
    assert batch["runs"][1]["abandoned"] == [str(partial)]
    assert experiment_directory(root, batch["runs"][1]).name.startswith(f"{names[1]}-20261019-120000-")
    execute(root, resume=True)  # A finished queue is left alone.
    assert launched == names[1:]


def test_run_directories_match_exact_names(tmp_path):
    for name in ("screen-shaped-20261019-120000-abcdef", "screen-shaped-2-20261019-130000-abcdef", "screen-shaped-old"):
        (tmp_path / "runs" / name).mkdir(parents=True)
    assert experiment_directory(tmp_path, {"name": "screen-shaped"}).name == "screen-shaped-20261019-120000-abcdef"

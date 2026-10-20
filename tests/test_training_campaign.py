import json
from pathlib import Path

import pytest

from sts2_env.training import campaign
from sts2_env.training.telemetry import atomic_json


def setup_campaign(tmp_path, states):
    root = tmp_path / "campaign"
    root.mkdir()
    stages = []
    for i, status in enumerate(states):
        queue = tmp_path / f"stage-{i}"
        atomic_json(queue / "batch.json", dict(status=status, pid=99999,
                    runs=[dict(status="running" if status == "running" else status, pid=99998)]))
        stages.append(dict(id=f"stage-{i}", root=str(queue)))
    atomic_json(root / "campaign.json", dict(status="queued", stages=stages, source_digest=campaign.source_digest()))
    return root, stages


def test_campaign_keeps_completed_stages_resumes_then_runs_v2(tmp_path, monkeypatch):
    root, stages = setup_campaign(tmp_path, ["completed", "running", "queued"])
    monkeypatch.setattr(campaign, "process_alive", lambda _: False)
    (Path(stages[1]["root"]) / "queue.lock").write_text("99999")
    commands = []
    class Finished:
        pid = 1
        def __init__(self, command, **_):
            commands.append(command)
            self.root = Path(command[command.index("--root") + 1])
        def wait(self):
            atomic_json(self.root / "batch.json", dict(status="completed"))
            return 0
        def poll(self):
            return 0
    monkeypatch.setattr(campaign.subprocess, "Popen", Finished)
    campaign.execute(root)
    assert len(commands) == 2 and "--resume" in commands[0] and "--resume" not in commands[1]
    assert json.loads((root / "campaign.json").read_text())["status"] == "completed"
    assert not (root / "campaign.lock").exists()


def test_campaign_does_not_remove_an_active_queue_lock(tmp_path, monkeypatch):
    root, stages = setup_campaign(tmp_path, ["running"])
    queue = Path(stages[0]["root"])
    (queue / "queue.lock").write_text("99999")
    monkeypatch.setattr(campaign, "process_alive", lambda pid: pid == 99998)
    with pytest.raises(RuntimeError, match="active owner"):
        campaign.execute(root)
    assert (queue / "queue.lock").exists()


def test_campaign_failure_stops_before_next_stage(tmp_path, monkeypatch):
    root, _ = setup_campaign(tmp_path, ["queued", "queued"])
    monkeypatch.setattr(campaign, "process_alive", lambda _: False)
    commands = []
    class Failed:
        pid = 1
        def __init__(self, command, **_):
            commands.append(command)
        def wait(self):
            return 1
        def poll(self):
            return 1
    monkeypatch.setattr(campaign.subprocess, "Popen", Failed)
    with pytest.raises(RuntimeError, match="later stages were not started"):
        campaign.execute(root)
    assert len(commands) == 1


def test_changed_training_code_prevents_the_next_launch(tmp_path, monkeypatch):
    root, _ = setup_campaign(tmp_path, ["queued"])
    monkeypatch.setattr(campaign, "source_digest", lambda: "changed")
    with pytest.raises(RuntimeError, match="Training code changed"):
        campaign.execute(root)

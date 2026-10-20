from copy import deepcopy
import json
from pathlib import Path
import threading
from urllib.request import Request, urlopen

from sts2_env.training import dashboard
from sts2_env.training.config import DEFAULTS
from sts2_env.training.telemetry import atomic_json, publish
from sts2_env.watch import catalog


def queue(root, name, *, status="queued"):
    config = deepcopy(DEFAULTS)
    config.update(name=name, task="act1", reward=dict(profile="shaped-v2", win=5, hp=1, upgrade=0))
    path = root / "configs" / f"{name}.json"
    atomic_json(path, config)
    entry = dict(name=name, profile=name, status=status, config=str(path))
    atomic_json(root / "batch.json", dict(status=status, runs=[entry]))
    atomic_json(root / "queue-control.json", dict(pause_after_current=False))
    return config, entry


def test_campaign_dashboard_merges_queues_and_routes_logs_exports_and_pause(tmp_path, monkeypatch):
    first, second, campaign = tmp_path / "first", tmp_path / "second", tmp_path / "campaign"
    queue(first, "alpha")
    config, _ = queue(second, "beta")
    directory = second / "runs" / "beta-20261020-000000-abcdef"
    atomic_json(directory / "config.json", config)
    publish(directory, "rollout", steps=32, episodes=2)
    (second / "beta.log").write_text("second queue log")
    atomic_json(campaign / "campaign.json", dict(status="queued", title="test campaign", stages=[
        dict(id="first", label="First", root=str(first)), dict(id="second", label="Second", root=str(second))]))
    state = dashboard.snapshot(campaign)
    assert [r["profile"] for r in state["runs"]] == ["alpha", "beta"]
    assert state["runs"][1]["rollouts"][0]["episodes"] == 2
    monkeypatch.setattr(dashboard, "discover_runs", lambda: [])
    server = dashboard.DashboardServer(("127.0.0.1", 0), campaign)
    monkeypatch.setattr(server, "gpu", lambda: {})
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(origin + "/api/log?profile=beta") as response:
            assert json.load(response)["text"] == "second queue log"
        with urlopen(origin + "/api/export?profile=beta&kind=config") as response:
            assert json.load(response)["name"] == "beta"
        with urlopen(origin + "/api/state") as response:
            assert json.load(response)["models"] == []
        request = Request(origin + "/api/pause-queue", method="POST",
                          headers={"Origin": origin, "X-Dashboard-Request": "1"})
        with urlopen(request) as response:
            assert json.load(response)["pause_after_current"]
        for root in (first, second, campaign):
            assert json.loads((root / "queue-control.json").read_text())["pause_after_current"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_catalog_lists_queued_models_without_inventing_results(tmp_path):
    queue(tmp_path / "v2", "long-shaped-v2")
    runs = catalog.discover_runs(tmp_path)
    assert len(runs) == 1
    run = runs[0]
    assert run["name"] == "long-shaped-v2" and run["status"] == "queued"
    assert run["results"] == dict(best=None, final=None, latest=None)
    assert run["models"] == dict(best=False, final=False)
    assert catalog.playable_reason(run, {}, "best") == "No best model saved yet"


def test_catalog_marks_abandoned_attempts_and_lists_the_new_queue_entry(tmp_path):
    root = tmp_path / "replication"
    config, entry = queue(root, "seed42")
    old = root / "runs" / "seed42-20261020-000000-abcdef"
    atomic_json(old / "config.json", config)
    atomic_json(old / "metadata.json", {})
    entry["abandoned"] = [str(old)]
    atomic_json(root / "batch.json", dict(status="queued", runs=[entry]))
    runs = catalog.discover_runs(tmp_path)
    # A queued retry still has its old directory until the new attempt starts.
    assert {r["status"] for r in runs if r["name"] == "seed42"} == {"abandoned", "queued"}

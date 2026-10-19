"""The Model player lists trained runs with their results and switches between compatible ones."""
from datetime import datetime, timedelta, timezone
import json
import threading
import time

import pytest
import torch

from sts2_env.entity.model import SpireFormer, SpireFormerConfig
from sts2_env.entity.ppo import EntityAgent
from sts2_env.gym_env.run_env import STS2RunEnv
from sts2_env.watch import catalog, player
from sts2_env.watch.control import Playback

LIVE = dict(seed="1", ascension=0, character="Ironclad", task="act1", start_with_neow=False)


def make_run(output, group, name, *, task="act1", character="Ironclad", reward="shaped", results=None,
             models=("best",), live=None):
    directory = output / group / f"{name}-20261019-000000-abcdef"
    (directory / "best_model").mkdir(parents=True)
    config = dict(name=name, task=task, character=character, ascension=0, start_with_neow=False,
                  model=dict(type="entity"), training=dict(total_timesteps=500000, model_seed=42, gamma=0.995),
                  evaluation=dict(mode="unseen", seeds=list(range(10000, 10050)), episodes=50,
                                  final_seeds=list(range(20000, 20200)), final_episodes=200))
    if reward == "shaped":
        config["reward"] = dict(profile="shaped", win=5.0, hp=0.5, upgrade=0.1)
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "metadata.json").write_text("{}", encoding="utf-8")
    for label, summary in (results or {}).items():
        (directory / catalog.RESULT_FILES[label]).write_text(json.dumps(summary), encoding="utf-8")
    for which in models:
        catalog.model_file(directory, True, which).write_bytes(b"")
    if live:
        (directory / "live.json").write_text(json.dumps(live), encoding="utf-8")
    return directory


def summary(wins, timeouts=0, stalls=0):
    return dict(episodes=200, wins=wins, win_rate=wins / 200, losses=200 - wins - timeouts - stalls,
                timeouts=timeouts, stalls=stalls, average_floor=10.5, boss_reached_rate=0.32,
                boss_conversion_rate=wins / 64, average_decisions=300.0,
                averages=dict(upgraded_cards=0.67, rest_sites=2.0, deck_size=14.0))


def test_catalog_lists_act1_runs_with_results_and_skips_smoke_and_combat(tmp_path):
    make_run(tmp_path, "reward-ablation-20261019", "screen-deckbuilder",
             results=dict(best=summary(12, stalls=1), final=summary(9)))
    make_run(tmp_path, "experiments", "old-sparse", reward=None)
    make_run(tmp_path, "experiments", "smoke-act1")
    make_run(tmp_path, "experiments", "entity-combat", task="combat")
    fresh = datetime.now(timezone.utc).isoformat()
    make_run(tmp_path, "reward-ablation-20261019", "screen-sparse", models=(),
             live=dict(kind="rollout", steps=250000, updated_utc=fresh))

    runs = {run["name"]: run for run in catalog.discover_runs(tmp_path)}
    assert set(runs) == {"screen-deckbuilder", "old-sparse", "screen-sparse"}

    deckbuilder = runs["screen-deckbuilder"]
    assert deckbuilder["group"] == "reward-ablation-20261019" and deckbuilder["status"] == "completed"
    assert deckbuilder["reward"] == dict(profile="shaped", hp=0.5, upgrade=0.1, win=5.0)
    assert deckbuilder["results"]["best"]["wins"] == 12 and deckbuilder["results"]["best"]["stalls"] == 1
    assert deckbuilder["results"]["best"]["upgraded_cards"] == 0.67
    assert deckbuilder["results"]["latest"] is None
    assert deckbuilder["seeds"] == {"best": "20000–20199", "final": "20000–20199", "latest": "10000–10049"}

    assert runs["old-sparse"]["reward"]["profile"] == "sparse (legacy)"
    assert runs["screen-sparse"]["status"] == "training" and runs["screen-sparse"]["steps"] == 250000
    table = catalog.format_table(list(runs.values()))
    assert table.isascii() and "12/200" in table and "no results yet" in table


def test_stale_telemetry_is_not_reported_as_training(tmp_path):
    old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    make_run(tmp_path, "experiments", "crashed", live=dict(kind="rollout", steps=10, updated_utc=old))
    assert catalog.discover_runs(tmp_path)[0]["status"] == "stopped"


def test_only_matching_saved_models_can_replace_the_live_policy(tmp_path, monkeypatch):
    make_run(tmp_path, "g", "ironclad", models=("best",))
    make_run(tmp_path, "g", "silent", character="Silent")
    monkeypatch.setattr(catalog, "OUTPUT", tmp_path)
    runs = {run["name"]: run for run in catalog.discover_runs()}
    assert catalog.playable_reason(runs["ironclad"], LIVE, "best") is None
    assert "No final model" in catalog.playable_reason(runs["ironclad"], LIVE, "final")
    assert "character" in catalog.playable_reason(runs["silent"], LIVE, "best")

    request = player.model_request(dict(action="use_model", run=runs["ironclad"]["id"], model="best"), LIVE)
    assert request["path"].endswith("best_model.pt") and request["run"] == runs["ironclad"]["id"]
    for bad in (dict(run="../../outside", model="best"), dict(run=runs["ironclad"]["id"], model="other"),
                dict(run=runs["silent"]["id"], model="best")):
        with pytest.raises(ValueError):
            player.model_request(dict(action="use_model", **bad), LIVE)
    listing = player.run_listing(LIVE, dict(run=runs["ironclad"]["id"], which="best"))
    assert listing["current"] == dict(run=runs["ironclad"]["id"], model="best")
    assert {r["name"]: r["playable"]["best"] for r in listing["runs"]}["ironclad"] is None


def tiny_agent():
    return EntityAgent(SpireFormer(SpireFormerConfig(d_model=32, layers=1, heads=4, n_actions=157)).eval(),
                       torch.device("cpu"))


def live_map_state():
    from tests.test_watch_entity import live_map
    env = STS2RunEnv(act_count=1)
    env.reset(options={"run_seed": 42})
    state = live_map(env)
    env.close()
    state["run_state"].update(seed="1", character="Ironclad")
    return dict(state, request_id="r1")


class _WaitingGame:
    def __init__(self, state):
        self.state, self.sent = state, []

    def connect(self):
        pass

    def receive_state(self):
        return self.state

    def send_action(self, action):
        self.sent.append(action)

    def disconnect(self):
        pass


def wait_for(condition, seconds=10):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return False


def test_switching_models_replans_a_paused_decision_without_sending_it(tmp_path, monkeypatch):
    replacement = tmp_path / "best_model.pt"
    tiny_agent().save(replacement, 1234)
    game = _WaitingGame(live_map_state())
    monkeypatch.setattr(player, "STS2GameClient", lambda *a, **k: game)
    playback = Playback(paused=True, delay=0)
    playback.context = dict(run="experiments/first", which="final")
    thread = threading.Thread(target=player.worker, args=(playback, tiny_agent(), LIVE, tmp_path, 100))
    thread.start()
    try:
        assert wait_for(lambda: playback.proposal is not None), "first proposal never reached the gate"
        playback.request_model(dict(run="ablation/second", name="second", model="best", entity=True,
                                    experiment=str(tmp_path), path=str(replacement)))
        assert wait_for(lambda: playback.context.get("run") == "ablation/second")
        assert wait_for(lambda: playback.status == "Paused before decision")
    finally:
        playback.control({"action": "stop"})
        thread.join(10)
    assert playback.context["which"] == "best" and playback.error is None
    assert playback.count == 0 and not [a for a in game.sent if a["action"] == "choose"]
    records = [json.loads(line) for line in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert records == [dict(model_switch=dict(run="ablation/second", name="second", model="best", entity=True,
                                              experiment=str(tmp_path), path=str(replacement)))]


def test_a_model_that_cannot_load_keeps_the_current_policy(tmp_path):
    playback = Playback()
    current = tiny_agent()
    with (tmp_path / "decisions.jsonl").open("w") as transcript:
        model, _ = player.switch_policy(playback, dict(run="x", name="broken", model="best", entity=True,
                                                       experiment="x", path=str(tmp_path / "missing.pt")),
                                        current, None, transcript)
    assert model is current and "Could not load broken" in playback.error

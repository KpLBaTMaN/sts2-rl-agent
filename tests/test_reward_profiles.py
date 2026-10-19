"""Reward profile resolution and shaped reward terms."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from sts2_env.core.enums import RoomType
from sts2_env.training import rewards
from sts2_env.training.config import DEFAULTS, load_config
from sts2_env.training.environment import ScenarioEnv
from sts2_env.training.rewards import RewardTracker, hp_value, resolve_reward


def config(**changes):
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(changes)
    return cfg


def test_default_profile_depends_on_task():
    assert resolve_reward(config(task="combat"))["profile"] == "sparse"
    shaped = resolve_reward(config(task="act1"))
    assert shaped["profile"] == "shaped" and shaped["win"] == 5 and shaped["hp_curve"] == "dota"


def test_overrides_inline_objects_and_paths(tmp_path):
    cfg = config(reward="shaped", reward_overrides={"floor": 0.25})
    assert resolve_reward(cfg)["floor"] == 0.25
    inline = resolve_reward(config(reward={"win": 2, "death": -2}))
    assert inline["profile"] == "inline" and inline["floor"] == 0 and inline["death"] == -2
    path = tmp_path / "mine.json"
    path.write_text(json.dumps({"win": 3, "hp": 0.5, "hp_curve": "linear"}))
    loaded = resolve_reward(config(reward=str(path)))
    assert loaded["profile"] == "mine" and loaded["hp_curve"] == "linear"


def test_builtin_profiles_work_without_local_files(monkeypatch, tmp_path):
    monkeypatch.setattr(rewards, "REWARDS_DIR", tmp_path)
    assert resolve_reward(config(reward="shaped"))["win"] == 5
    with pytest.raises(ValueError, match="not found"):
        resolve_reward(config(reward="missing"))


@pytest.mark.parametrize("bad", [
    {"reward": {"wins": 1}},
    {"reward": {"win": "big"}},
    {"reward": {"hp_curve": "cubic"}},
    {"reward": "../escape"},
    {"reward_overrides": {"florr": 1}},
])
def test_bad_reward_settings_are_rejected(bad):
    with pytest.raises(ValueError):
        resolve_reward(config(**bad))


def test_resolved_config_round_trips(tmp_path):
    path = tmp_path / "train.json"
    path.write_text(json.dumps({"reward": "shaped", "reward_overrides": {"elite": 1}}))
    loaded = load_config(path)
    assert loaded["reward"]["elite"] == 1 and loaded["reward_overrides"] is None
    path.write_text(json.dumps(loaded))
    assert load_config(path)["reward"] == loaded["reward"]


def test_dota_hp_curve_punishes_low_hp_losses_more():
    assert hp_value(80, 80, "dota") == 1 and hp_value(0, 80, "dota") == 0
    high_loss = hp_value(80, 80, "dota") - hp_value(70, 80, "dota")
    low_loss = hp_value(20, 80, "dota") - hp_value(10, 80, "dota")
    assert low_loss > high_loss
    assert hp_value(40, 80, "linear") == 0.5


def test_sparse_profile_reproduces_original_rewards():
    env = ScenarioEnv(config(reward="sparse"))
    env.reset()
    rewards_seen = []
    while True:
        action = int(np.flatnonzero(env.action_masks())[0])
        _, reward, terminated, truncated, info = env.step(action)
        rewards_seen.append(reward)
        if terminated or truncated:
            break
    assert all(r == 0 for r in rewards_seen[:-1]) and rewards_seen[-1] in (1, -1)
    assert info["reward_terms"]["result"] == rewards_seen[-1]


def test_shaped_episode_reports_terms_that_sum_to_return():
    env = ScenarioEnv(config(reward="shaped"))
    env.reset()
    total = 0.0
    while True:
        action = int(np.flatnonzero(env.action_masks())[0])
        _, reward, terminated, truncated, info = env.step(action)
        total += reward
        if terminated or truncated:
            break
    terms = info["reward_terms"]
    assert total == pytest.approx(sum(terms.values()))
    assert terms["floor"] == pytest.approx(0.1 * info["floor"])


def _fake_run(room, combat, hp=50, floor=3, relics=1, gold=99):
    player = SimpleNamespace(current_hp=hp, max_hp=80, relics=["x"] * relics, gold=gold)
    mgr = SimpleNamespace(run_state=SimpleNamespace(player=player, total_floor=floor),
                          _current_room_type=room, get_combat_state=lambda: combat)
    return SimpleNamespace(_mgr=mgr)


def test_elite_win_enemy_damage_and_relic_terms():
    weights = resolve_reward(config(reward="shaped"))
    enemy = SimpleNamespace(current_hp=40, max_hp=40)
    combat = SimpleNamespace(enemies=[enemy], is_over=False, player_won=False,
                             player=SimpleNamespace(current_hp=50, max_hp=80))
    tracker = RewardTracker(weights, "act1")
    tracker.reset(_fake_run(RoomType.ELITE, combat))

    enemy.current_hp = 20
    reward, terms = tracker.step(_fake_run(RoomType.ELITE, combat), False, False, False)
    assert terms["enemy_damage"] == pytest.approx(0.05 * 0.5) and terms["elite"] == 0

    enemy.current_hp = 0
    combat.is_over, combat.player_won = True, True
    _, terms = tracker.step(_fake_run(RoomType.ELITE, None, relics=2), False, False, False)
    assert terms["elite"] == 0.3 and terms["relic"] == 0.05
    assert terms["enemy_damage"] == pytest.approx(0.05 * 0.5)
    assert tracker.totals["elite"] == 0.3


def test_reviving_enemy_cannot_be_farmed_for_damage_reward():
    weights = resolve_reward(config(reward="shaped"))
    eye = SimpleNamespace(current_hp=6, max_hp=6)
    combat = SimpleNamespace(enemies=[eye], is_over=False, player_won=False,
                             player=SimpleNamespace(current_hp=50, max_hp=80))
    tracker = RewardTracker(weights, "act1")
    tracker.reset(_fake_run(RoomType.MONSTER, combat))
    for _ in range(5):
        eye.current_hp = 0
        tracker.step(_fake_run(RoomType.MONSTER, combat), False, False, False)
        eye.current_hp = 6
        tracker.step(_fake_run(RoomType.MONSTER, combat), False, False, False)
    assert tracker.totals["enemy_damage"] == pytest.approx(0.05)

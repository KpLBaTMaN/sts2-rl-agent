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
    assert resolve_reward(config(reward="deckbuilder"))["upgrade"] == 0.1
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


def test_permanent_upgrade_pays_once_and_not_for_new_upgraded_cards():
    env = _fake_run(RoomType.REST_SITE, None, hp=80)
    card = SimpleNamespace(upgraded=False)
    env._mgr.run_state.player.deck = [card]
    tracker = RewardTracker(resolve_reward(config(reward="deckbuilder")), "act1")
    tracker.reset(env)
    # Selecting Smith without completing its choice earns nothing.
    _, terms = tracker.step(env, False, False, False)
    assert terms["upgrade"] == 0
    card.upgraded = True
    _, terms = tracker.step(env, False, False, False)
    assert terms["upgrade"] == pytest.approx(0.1)
    env._mgr.run_state.player.deck.append(SimpleNamespace(upgraded=True))
    _, terms = tracker.step(env, False, False, False)
    assert terms["upgrade"] == 0
    # Even a downgrade/re-upgrade cannot repeatedly farm the same instance.
    card.upgraded = False
    tracker.step(env, False, False, False)
    card.upgraded = True
    tracker.step(env, False, False, False)
    assert tracker.totals["upgrade"] == pytest.approx(0.1)


def test_deckbuilder_heal_and_upgrade_tradeoff():
    weights = resolve_reward(config(reward="deckbuilder"))
    def heal_reward(hp, healed):
        env = _fake_run(RoomType.REST_SITE, None, hp=hp)
        tracker = RewardTracker(weights, "act1")
        tracker.reset(env)
        env._mgr.run_state.player.current_hp = min(80, hp + healed)
        _, terms = tracker.step(env, False, False, False)
        return terms["hp"]
    assert heal_reward(80, 24) == 0
    assert heal_reward(77, 24) < weights["upgrade"]
    assert heal_reward(20, 24) > weights["upgrade"]


def test_temporary_combat_upgrade_earns_no_permanent_deck_reward():
    permanent = SimpleNamespace(upgraded=False)
    temporary = SimpleNamespace(upgraded=False)
    combat = SimpleNamespace(enemies=[], is_over=False, player_won=False,
                             player=SimpleNamespace(current_hp=50, max_hp=80),
                             hand=[temporary])
    env = _fake_run(RoomType.MONSTER, combat)
    env._mgr.run_state.player.deck = [permanent]
    tracker = RewardTracker(resolve_reward(config(reward="deckbuilder")), "act1")
    tracker.reset(env)
    temporary.upgraded = True
    _, terms = tracker.step(env, False, False, False)
    assert terms["upgrade"] == 0


def test_real_smith_choice_pays_after_card_selection():
    from sts2_env.gym_env.run_env import _REST_START
    env = ScenarioEnv(config(task="act1", reward="deckbuilder"))
    try:
        env.reset()
        mgr = env.env._mgr
        mgr._enter_rest_site()
        env.rewards.reset(env.env)
        _, reward, _, _, _ = env.step(_REST_START + 1)
        assert reward == 0
        _, reward, _, _, _ = env.step(1)  # Complete the pending upgrade choice.
        assert reward == pytest.approx(0.1)
        assert env.rewards.totals["upgrade"] == pytest.approx(0.1)
        assert sum(c.upgraded for c in mgr.run_state.player.deck) == 1
    finally:
        env.close()

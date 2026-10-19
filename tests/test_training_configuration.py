"""Regression checks for seed semantics, Act 1 success, and config mistakes."""
import copy
import json

import numpy as np
import pytest

from sts2_env.core.enums import RoomType
from sts2_env.gym_env.run_env import STS2RunEnv
from sts2_env.run.run_manager import RunManager
from sts2_env.training.config import DEFAULTS, load_config
from sts2_env.training.environment import ScenarioEnv


def config():
    return copy.deepcopy(DEFAULTS)


def trace(env):
    obs, _ = env.reset()
    states = [obs.copy()]
    for _ in range(100):
        action = int(np.flatnonzero(env.action_masks())[0])
        obs, reward, terminated, truncated, info = env.step(action)
        states.append(obs.copy())
        if terminated or truncated:
            break
    return np.stack(states)


def test_fixed_seed_repeats_actual_simulator_seed_and_trajectory():
    env = ScenarioEnv(config())
    first = trace(env)
    assert env.env._mgr._seed == 42
    second = trace(env)
    np.testing.assert_array_equal(first, second)
    assert env.env._mgr._seed == 42


def test_pool_cycles_and_random_reproduces_without_evaluation_leakage():
    cfg = config()
    cfg["seeds"].update(mode="pool", values=[42, 123, 456])
    env = ScenarioEnv(cfg)
    assert [env._next_seed() for _ in range(5)] == [42, 123, 456, 42, 123]
    cfg["seeds"]["mode"] = "random"
    cfg["evaluation"]["mode"] = "unseen"
    first, second = ScenarioEnv(cfg), ScenarioEnv(cfg)
    seeds = [first._next_seed() for _ in range(20)]
    assert seeds == [second._next_seed() for _ in range(20)]
    assert len(set(seeds)) == 20
    assert not set(seeds) & set(cfg["evaluation"]["seeds"])
    test = ScenarioEnv(cfg, evaluation=True)
    assert [test._next_seed() for _ in range(5)] == cfg["evaluation"]["seeds"][:5]


def test_policy_reset_seed_does_not_replace_fixed_game_seed():
    env = ScenarioEnv(config())
    env.reset(seed=999)
    assert env.env._mgr._seed == 42
    env.reset()
    assert env.env._mgr._seed == 42


def setup_boss(env):
    env.reset(seed=0, options={"run_seed": 42})
    manager = env._mgr
    manager._current_room_type = RoomType.BOSS
    manager._enter_combat(RoomType.BOSS)
    return manager, manager.get_combat_state()


def test_act1_objective_ends_on_boss_defeat_without_waiting_for_rewards(monkeypatch):
    env = STS2RunEnv(act_count=1)
    manager, combat = setup_boss(env)
    def win(_action):
        combat.is_over = True
        combat.player_won = True
        manager._phase = RunManager.PHASE_CARD_REWARD
    monkeypatch.setattr(env, "_step_combat", win)
    _, reward, terminated, truncated, _ = env.step(0)
    assert terminated and not truncated and reward == 1
    assert manager.player_won


def test_regular_combat_victory_does_not_finish_act1(monkeypatch):
    env = STS2RunEnv(act_count=1)
    manager, combat = setup_boss(env)
    manager._current_room_type = RoomType.MONSTER
    def win(_action):
        combat.is_over = True
        combat.player_won = True
        manager._phase = RunManager.PHASE_CARD_REWARD
    monkeypatch.setattr(env, "_step_combat", win)
    _, reward, terminated, _, _ = env.step(0)
    assert not terminated and reward == 0


def test_boss_loss_is_not_misreported_as_act1_victory(monkeypatch):
    env = STS2RunEnv(act_count=1)
    manager, combat = setup_boss(env)
    def lose(_action):
        combat.is_over = True
        combat.player_won = False
        manager.run_state.player.current_hp = 0
        manager.run_state.lose_run()
    monkeypatch.setattr(env, "_step_combat", lose)
    _, reward, terminated, _, _ = env.step(0)
    assert terminated and reward == -1 and not manager.player_won


def test_simulator_errors_are_not_training_losses(monkeypatch):
    env = STS2RunEnv(act_count=1, raise_simulation_errors=True)
    manager, _ = setup_boss(env)
    def fail(_action):
        raise RuntimeError("simulator fault")
    monkeypatch.setattr(env, "_step_combat", fail)
    with pytest.raises(RuntimeError, match="simulator fault"):
        env.step(0)
    assert not manager.is_over


def test_timeout_is_failure_and_requires_reset():
    cfg = config()
    cfg["training"]["max_episode_steps"] = 1
    env = ScenarioEnv(cfg)
    env.reset()
    action = int(np.flatnonzero(env.action_masks())[0])
    _, reward, _, truncated, info = env.step(action)
    assert truncated and not info["is_success"]
    # Shaping may add progress terms; the timeout result itself is still -1.
    assert info["reward_terms"]["result"] == -1 and reward == sum(info["reward_terms"].values())
    with pytest.raises(RuntimeError, match="reset"):
        env.step(action)


@pytest.mark.parametrize("override,match", [
    ({"seeds": {"mode": "fixed", "values": [1, 2]}}, "exactly one"),
    ({"seed": 123}, "Unknown"),
    ({"training": {"batch_size": 123}}, "divide"),
    ({"evaluation": {"mode": "unseen", "seeds": [42]}}, "overlap"),
    ({"task": "combat", "character": "Silent"}, "Ironclad"),
])
def test_bad_config_fails_before_training(tmp_path, override, match):
    path = tmp_path / "test.json"
    path.write_text(json.dumps(override))
    with pytest.raises(ValueError, match=match):
        load_config(path)


def test_finished_run_reports_end_of_run_state():
    cfg = config()
    cfg["task"] = "act1"
    env = ScenarioEnv(cfg)
    env.reset()
    while True:
        action = int(np.flatnonzero(env.action_masks())[0])
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break
    stats = info["run_stats"]
    player = env.env._mgr.run_state.player
    assert stats["gold"] == player.gold and stats["deck_size"] == len(player.deck)
    assert stats["relics"].split(";") == player.relics and stats["relic_count"] == len(player.relics)
    assert len(stats["deck"].split(";")) == stats["deck_size"]
    assert stats["attacks"] + stats["skills"] + stats["powers"] + stats["curses_statuses"] <= stats["deck_size"]


def test_combat_task_has_no_run_snapshot():
    cfg = config()
    cfg["task"] = "combat"
    cfg["training"]["max_episode_steps"] = 1
    env = ScenarioEnv(cfg)
    env.reset()
    _, _, _, _, info = env.step(int(np.flatnonzero(env.action_masks())[0]))
    assert "run_stats" not in info


def test_reports_split_outcomes_and_rank_relics_and_cards(tmp_path):
    from sts2_env.training.runner import format_summary, run_summary, save_evaluation

    def row(win, gold, relics, deck, timed_out=False):
        return dict(win=win, timed_out=timed_out, floor=5, gold=gold, deck_size=len(deck.split(";")),
                    relic_count=len(relics.split(";")), upgraded_cards=deck.count("+"), relics=relics, deck=deck)

    rows = [row(True, 100, "BURNING_BLOOD;ANCHOR", "BASH+;STRIKE;STRIKE"),
            row(False, 50, "BURNING_BLOOD", "BASH;STRIKE"),
            row(False, 10, "BURNING_BLOOD", "BASH", timed_out=True)]
    outcome = run_summary([dict(r, **{k: 1 for k in ("act", "max_hp", "attacks", "skills", "powers",
                                                     "curses_statuses", "potions_held", "card_removals",
                                                     "elites", "monsters", "shops", "rest_sites",
                                                     "events", "treasures")}) for r in rows])["by_outcome"]
    assert outcome["wins"]["gold"] == 100 and outcome["losses"]["gold"] == 50 and outcome["timeouts"]["gold"] == 10

    save_evaluation(tmp_path, "final-results", {"wins": 1}, rows)
    relics = (tmp_path / "final-results-relic-stats.csv").read_text(encoding="utf-8").splitlines()
    assert relics[1].startswith("BURNING_BLOOD,3,") and relics[2] == "ANCHOR,1,0.3333333333333333,1.0,1.0,0.0"
    cards = (tmp_path / "final-results-card-stats.csv").read_text(encoding="utf-8").splitlines()
    # Upgrades count as the same card; STRIKE appears twice in the winning deck.
    assert cards[1].startswith("BASH,3,") and cards[2].startswith("STRIKE,2,") and ",1.5," in cards[2]
    text = format_summary({"wins": 1, "episodes": 3, "win_rate": 1 / 3, "losses": 1, "timeouts": 1})
    assert text == "Final: 1/3 wins (33.3%), 1 losses, 1 timeouts"

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
    assert truncated and reward == -1 and not info["is_success"]
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

"""Entity observations, SpireFormer action layout, and the entity training path."""
import copy

import numpy as np
import pytest
import torch

from sts2_env.core.constants import ACTION_SPACE_SIZE, MAX_ENEMIES, MAX_HAND_SIZE
from sts2_env.entity.model import SpireFormer, SpireFormerConfig, masked_logits
from sts2_env.entity.observation import (
    MAX_CHOICES,
    MAX_RELICS,
    OBS_SPEC,
    ROUTE_START,
    EntityObservationWrapper,
    _PATH_TYPES,
    _fill_relics,
    _pending_choice,
    cap_action_mask,
    empty_observation,
    hash_bucket,
    run_options,
)
from sts2_env.entity.ppo import EntityAgent, stack_observations, to_tensors
from sts2_env.gym_env.combat_env import STS2CombatEnv
from sts2_env.gym_env.run_env import _LAYOUT, TOTAL_ACTIONS, STS2RunEnv
from sts2_env.relics.base import RelicId
from sts2_env.run.run_manager import RunManager
from sts2_env.training.config import DEFAULTS, load_config
from sts2_env.training.environment import make_env


def entity_config(task="combat"):
    cfg = copy.deepcopy(DEFAULTS)
    cfg["task"] = task
    cfg["seeds"]["mode"] = "random"
    cfg["model"].update(type="entity", d_model=32, layers=1, heads=4)
    return cfg


def small_model(n_actions=ACTION_SPACE_SIZE):
    torch.manual_seed(0)
    return SpireFormer(SpireFormerConfig(d_model=32, layers=1, heads=4, n_actions=n_actions)).eval()


def run_env(seed=0):
    env = EntityObservationWrapper(STS2RunEnv(raise_simulation_errors=True))
    obs, _ = env.reset(options={"run_seed": seed})
    return env, obs


def test_encoder_matches_spec_and_describes_starting_fight():
    env = EntityObservationWrapper(STS2CombatEnv())
    obs, _ = env.reset(seed=3)
    assert set(obs) == set(OBS_SPEC)
    for key, (shape, dtype) in OBS_SPEC.items():
        assert obs[key].shape == shape and obs[key].dtype == dtype, key
    assert env.observation_space.contains(obs)
    combat = env.unwrapped.combat
    assert obs["hand_mask"].sum() == len(combat.hand)
    assert obs["pile_mask"].sum() == len(combat.draw_pile) + len(combat.discard_pile) + len(combat.exhaust_pile)
    assert obs["enemy_mask"].sum() == len(combat.enemies)
    assert (obs["hand_ids"][obs["hand_mask"]] > 0).all()
    assert not obs["pending_choice"][0]


def test_same_card_gets_same_id_in_any_hand_slot():
    env = EntityObservationWrapper(STS2CombatEnv())
    obs, _ = env.reset(seed=5)
    hand = env.unwrapped.combat.hand
    by_card = {}
    for i, card in enumerate(hand):
        by_card.setdefault(card.card_id, set()).add(int(obs["hand_ids"][i]))
    assert all(len(ids) == 1 for ids in by_card.values())


def test_policy_scores_every_action_with_finite_logits_and_respects_mask():
    env = EntityObservationWrapper(STS2CombatEnv())
    obs, _ = env.reset(seed=7)
    model = small_model()
    logits, value = model(to_tensors(stack_observations([obs, obs]), torch.device("cpu")))
    assert logits.shape == (2, ACTION_SPACE_SIZE) and value.shape == (2,)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    mask = torch.as_tensor(env.action_masks().astype(bool))[None].expand(2, -1)
    probs = torch.softmax(masked_logits(logits, mask), -1)
    assert torch.allclose(probs[~mask], torch.zeros(()))


def test_card_target_logit_comes_from_that_card_and_enemy():
    """Changing hand card 2's details changes the scores of card 2's own actions."""
    env = EntityObservationWrapper(STS2CombatEnv())
    obs, _ = env.reset(seed=9)
    model = small_model()
    base, _ = model(to_tensors(stack_observations([obs]), torch.device("cpu")))
    changed = {k: v.copy() for k, v in obs.items()}
    changed["hand_features"][2, 2] += 1.0
    after, _ = model(to_tensors(stack_observations([changed]), torch.device("cpu")))
    card2 = [1 + 2] + [1 + MAX_HAND_SIZE + 2 * MAX_ENEMIES + j for j in range(MAX_ENEMIES)]
    assert not torch.allclose(base[0, card2], after[0, card2])


def test_choice_mask_is_capped_to_visible_options():
    pending = object()
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
    mask[1 + MAX_CHOICES + 3] = 1
    mask[2] = 1
    capped = cap_action_mask(pending, mask)
    assert capped[2] == 1 and capped[1 + MAX_CHOICES + 3] == 0
    assert (cap_action_mask(None, mask) == mask).all()
    only_hidden = np.zeros_like(mask)
    only_hidden[1 + MAX_CHOICES + 3] = 1
    assert cap_action_mask(pending, only_hidden)[1 + MAX_CHOICES + 3] == 1


def test_random_entity_episodes_stay_legal_and_encodable():
    env = make_env(entity_config())
    rng = np.random.default_rng(0)
    for _ in range(5):
        obs, _ = env.reset()
        for _ in range(500):
            mask = env.env.action_masks()
            obs, _, terminated, truncated, info = env.step(int(rng.choice(np.flatnonzero(mask))))
            assert obs["global"].shape == OBS_SPEC["global"][0]
            if terminated or truncated:
                break
        assert terminated or truncated


def test_agent_predict_and_save_load_round_trip(tmp_path):
    env = make_env(entity_config())
    obs, _ = env.reset()
    agent = EntityAgent(small_model(), torch.device("cpu"))
    mask = env.env.action_masks()
    action, _ = agent.predict(obs, action_masks=mask, deterministic=True)
    assert mask[action]
    agent.save(tmp_path / "m.pt", num_timesteps=123)
    loaded, saved = EntityAgent.load(tmp_path / "m.pt")
    assert saved["num_timesteps"] == 123
    assert loaded.predict(obs, action_masks=mask, deterministic=True)[0] == action


def test_entity_config_validation(tmp_path):
    path = tmp_path / "c.json"
    path.write_text('{"task": "combat", "seeds": {"mode": "random"}, "model": {"type": "entity"}}')
    assert load_config(path)["model"]["type"] == "entity"
    path.write_text('{"task": "act1", "model": {"type": "entity"}}')
    assert load_config(path)["model"]["type"] == "entity"
    path.write_text('{"task": "act1", "model": {"type": "mlp"}}')
    with pytest.raises(ValueError, match="flat or entity"):
        load_config(path)
    path.write_text('{"task": "combat", "seeds": {"mode": "random"}, "model": {"type": "entity", "d_model": 30, "heads": 4}}')
    with pytest.raises(ValueError, match="divisible"):
        load_config(path)
    assert DEFAULTS["model"]["type"] == "entity"


def test_run_start_shows_deck_relics_and_map_options():
    env, obs = run_env(seed=1)
    mgr = env.unwrapped._mgr
    assert mgr.phase == RunManager.PHASE_MAP_CHOICE
    assert obs["pile_mask"].sum() == len(mgr.run_state.player.deck)
    assert (obs["pile_kinds"][obs["pile_mask"]] == 5).all()  # deck cards
    assert obs["relic_mask"].sum() == len(mgr.run_state.relics) and (obs["relic_ids"][obs["relic_mask"]] > 0).all()
    moves = mgr.get_available_actions()
    assert obs["option_mask"].sum() == len(moves)
    mask = env.action_masks()
    assert set(obs["option_slots"][obs["option_mask"]]) == set(np.flatnonzero(mask))
    # Path summaries: every route from the first row reaches at least one fight.
    features = obs["option_features"][obs["option_mask"]]
    most_monsters = ROUTE_START + len(_PATH_TYPES)
    assert (features[:, most_monsters] > 0).all()
    # The same summary, over every first-row node, describes the act from where the player stands.
    assert obs["route_ahead"][len(_PATH_TYPES)] > 0
    assert obs["boss_id"][0] == hash_bucket(f"boss:{mgr.run_state.boss_encounter_id}")


def test_every_owned_relic_is_encoded_up_to_the_cap_and_overflow_warns():
    obs = empty_observation()
    _fill_relics(obs, [relic.name for relic in list(RelicId)[:MAX_RELICS]])
    assert obs["relic_mask"].all() and (obs["relic_ids"] > 0).all()
    with pytest.warns(RuntimeWarning, match="MAX_RELICS"):
        _fill_relics(empty_observation(), list(RelicId)[:MAX_RELICS + 1])


def test_every_legal_run_action_is_scored_by_an_item():
    """Combat actions use card/potion heads, deck picks use choice items, the rest use options."""
    rng = np.random.default_rng(1)
    for seed in range(8):
        env, obs = run_env(seed)
        for _ in range(1500):
            mask = env.action_masks()
            base = env.unwrapped
            pending = _pending_choice(base) is not None
            in_combat = base._mgr.phase == RunManager.PHASE_COMBAT
            options = set(obs["option_slots"][obs["option_mask"]].tolist())
            for slot in np.flatnonzero(mask):
                covered = (pending and slot <= MAX_CHOICES) or (in_combat and not pending and slot < ACTION_SPACE_SIZE)
                assert covered or slot in options, (base._mgr.phase, slot)
            obs, _, terminated, truncated, _ = env.step(int(rng.choice(np.flatnonzero(mask))))
            if terminated or truncated:
                break


def test_option_scores_land_on_their_action_index():
    env, obs = run_env(seed=2)
    model = small_model(TOTAL_ACTIONS)
    logits, _ = model(to_tensors(stack_observations([obs]), torch.device("cpu")))
    assert logits.shape == (1, TOTAL_ACTIONS)
    changed = {k: v.copy() for k, v in obs.items()}
    changed["option_features"][0, 0] += 3.0
    after, _ = model(to_tensors(stack_observations([changed]), torch.device("cpu")))
    slot = int(obs["option_slots"][0])
    assert not torch.isclose(logits[0, slot], after[0, slot])
    mask = torch.as_tensor(env.action_masks().astype(bool))[None]
    probs = torch.softmax(masked_logits(logits, mask), -1)
    assert torch.isclose(probs[mask].sum(), torch.tensor(1.0))


def test_run_options_follow_env_slot_meaning():
    env, _ = run_env(seed=3)
    base = env.unwrapped
    rng = np.random.default_rng(3)
    checked = 0
    for _ in range(400):
        mask = env.action_masks()
        mgr = base._mgr
        if mgr.phase == RunManager.PHASE_MAP_CHOICE:
            for slot, kind, action in run_options(base, mask):
                assert kind == "move" and mgr.get_available_actions()[slot - _LAYOUT.map_start] == action
                checked += 1
        _, _, terminated, truncated, _ = env.step(int(rng.choice(np.flatnonzero(mask))))
        if terminated or truncated:
            break
    assert checked


def test_act1_entity_training_env_runs():
    env = make_env(entity_config("act1"))
    obs, _ = env.reset()
    assert env.action_space.n == TOTAL_ACTIONS
    agent = EntityAgent(small_model(TOTAL_ACTIONS), torch.device("cpu"))
    for _ in range(200):
        mask = env.env.action_masks()
        action, _ = agent.predict(obs, action_masks=mask, deterministic=False)
        assert mask[action]
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

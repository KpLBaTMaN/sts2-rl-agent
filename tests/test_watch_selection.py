"""Live card selection must follow the training protocol: picks are final and Confirm ends it."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sts2_env.core.selection import CardChoiceOption, PendingCardChoice
from sts2_env.entity import observation as E
from sts2_env.entity.model import SpireFormer, SpireFormerConfig
from sts2_env.entity.ppo import EntityAgent
from sts2_env.gym_env.run_env import STS2RunEnv
from sts2_env.watch import player
from sts2_env.watch.entity_adapter import EntityWatchAdapter
from tests.test_watch_entity import card

# Scripted policies replace the network; only the action count is read from the model.
MODEL = SimpleNamespace(action_space=SimpleNamespace(n=157))
CHOICE_KEYS = ("pending_choice", "choice_ids", "choice_kinds", "choice_features", "choice_mask")


def live_select(cards, min_select, max_select):
    return dict(type="card_select", entity_schema=1, cards=cards, min_select=min_select, max_select=max_select,
                deck=cards, relics=[], potions=[],
                run_state=dict(act_index=0, floor=3, act_floor=3, hp=60, max_hp=80, gold=99, deck_size=len(cards),
                               relic_count=0, potion_count=0, max_potion_slots=3, ascension=0,
                               room="RestSite", in_combat=False))


def simple_cards(n):
    return [dict(id=name, source_pile="deck") for name in ["STRIKE_IRONCLAD", "DEFEND_IRONCLAD", "BASH", "ANGER"][:n]]


def scripted(scores, calls):
    """A policy with fixed preferences that records each observation and mask it is shown."""
    def probabilities(model, obs, mask):
        calls.append((obs, mask.copy()))
        logits = np.where(mask, np.resize(np.asarray(scores, dtype=float), mask.shape), -np.inf)
        p = np.exp(logits - logits.max())
        return p / p.sum()
    return probabilities


def test_live_selection_encoding_matches_training_after_each_pick():
    env = STS2RunEnv(act_count=1)
    try:
        env.reset(options={"run_seed": 42})
        deck = env._mgr.run_state.player.deck
        env._mgr.run_state.pending_choice = PendingCardChoice(
            prompt="Choose 2 cards to upgrade", options=[CardChoiceOption(c, "deck") for c in deck],
            resolver=lambda cards: None, min_choices=2, max_choices=2)
        live = live_select([card(c) for c in deck], 2, 2)
        adapter = EntityWatchAdapter()
        for selected in ([], [3]):
            for index in selected:
                env._mgr.take_action({"action": "choose", "index": index})
            expected = E.encode_run_entities(env, env.action_masks())
            actual = adapter.encode(live, selected)
            np.testing.assert_allclose(actual["global"][11:16], expected["global"][11:16])
            for key in CHOICE_KEYS:
                np.testing.assert_allclose(actual[key], expected[key], err_msg=key)
    finally:
        env.close()


def test_any_number_selection_sends_every_pick_then_confirms(monkeypatch):
    calls = []
    # Prefers card 3, then card 0, then confirming; never the other two.
    monkeypatch.setattr(player, "policy_probabilities", scripted([5, 8, 0, 1, 9], calls))
    state = live_select(simple_cards(4), 0, 4)
    wire, proposal = player.model_decision(MODEL, EntityWatchAdapter(), state)
    assert wire == {"action": "choose", "indexes": [3, 0]}
    assert [p["label"] for p in proposal["picks"]] == ["Select ANGER", "Select STRIKE_IRONCLAD", "Confirm selection"]
    # Each pick is shown to the next query and can no longer be chosen.
    (_, first_mask), (second_obs, second_mask), _ = calls
    assert first_mask[4] and not second_mask[4]
    assert second_obs["choice_features"][3][5] == 1 and second_obs["global"][14] == pytest.approx(1 / 5)


def test_policy_that_reselects_its_pick_still_finishes(monkeypatch):
    # The saved checkpoints toggled the same card forever and almost never confirmed.
    calls = []
    def prefers_selected(model, obs, mask):
        calls.append(mask.copy())
        scores = np.full(mask.shape, -1.0)
        scores[0] = -50
        scores[1:1 + E.MAX_CHOICES] = 10 * obs["choice_features"][:, 5]
        p = np.where(mask, np.exp(scores), 0)
        return p / p.sum()
    monkeypatch.setattr(player, "policy_probabilities", prefers_selected)
    wire, _ = player.model_decision(MODEL, EntityWatchAdapter(), live_select(simple_cards(4), 2, 2))
    assert wire["action"] == "choose" and len(wire["indexes"]) == 2 == len(set(wire["indexes"]))
    assert len(calls) == 3 and np.flatnonzero(calls[-1]).tolist() == [0], "at the maximum only Confirm is legal"


def test_single_and_skipped_selections_keep_their_wire_forms(monkeypatch):
    monkeypatch.setattr(player, "policy_probabilities", scripted([0, 1, 9, 2], []))
    wire, _ = player.model_decision(MODEL, EntityWatchAdapter(), live_select(simple_cards(3), 1, 1))
    assert wire == {"action": "choose", "index": 1}
    monkeypatch.setattr(player, "policy_probabilities", scripted([9, 1, 2, 3], []))
    wire, proposal = player.model_decision(MODEL, EntityWatchAdapter(), live_select(simple_cards(3), 0, 3))
    assert wire == {"action": "skip"} and proposal["label"] == "Skip selection"


def test_trained_model_selection_is_legal_and_bounded():
    agent = EntityAgent(SpireFormer(SpireFormerConfig(d_model=32, layers=1, heads=4, n_actions=157)).eval(), torch.device("cpu"))
    state = live_select(simple_cards(4), 0, 3)
    wire, proposal = player.model_decision(agent, EntityWatchAdapter(), state)
    picks = wire.get("indexes", [])
    assert wire["action"] in {"choose", "skip"} and len(picks) == len(set(picks)) <= 3
    assert 1 <= len(proposal["picks"]) <= 4

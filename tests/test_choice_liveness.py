"""Card selections must finish, and decision loops must stop with a diagnosis.

Regression cases for the 19 October 2026 loop audit: optional discard and
exhaust, Whetstone's capped upgrade choice, Gnarled Hammer's enchantment and an
event selection.  A policy that never confirms voluntarily used to deselect and
reselect forever; it must now reach confirmation within ``max + 1`` decisions.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from sts2_env.cards.factory import create_card
from sts2_env.core.enums import CardId, CardType
from sts2_env.core.selection import CardChoiceOption, PendingCardChoice
from sts2_env.entity import observation as E
from sts2_env.entity.model import SpireFormer, SpireFormerConfig
from sts2_env.entity.ppo import EntityAgent
from sts2_env.gym_env.run_env import STS2RunEnv
from sts2_env.potions.effects import _ashwater, _gamblers_brew
from sts2_env.run.events import EventModel, EventOption
from sts2_env.run.run_manager import RunManager
from sts2_env.training import runner
from sts2_env.training.config import load_config
from sts2_env.training.environment import ScenarioEnv
from sts2_env.training.loops import LoopDetector
from sts2_env.watch import player
from sts2_env.watch.control import Playback


def run_env(seed=42):
    env = STS2RunEnv(act_count=1, raise_simulation_errors=True)
    env.reset(options={"run_seed": seed})
    return env


def enter_first_combat(env):
    while env._mgr.phase != RunManager.PHASE_COMBAT:
        env.step(int(np.flatnonzero(env.action_masks())[0]))
    return env._mgr.get_combat_state()


def stubborn_action(mask, favourite):
    """The audited failure: keep clicking one card, confirm only when nothing else is legal."""
    legal = np.flatnonzero(mask[1:]) + 1
    if favourite in legal:
        return favourite
    return int(legal[-1]) if len(legal) else 0


def finish_choice(env, favourite=1):
    """Drive the active choice to completion; return the selections seen."""
    choice = env._mgr.active_card_choice
    history = []
    for _ in range(choice.max_choices + 1):
        history.append(sorted(choice.selected_indices))
        env.step(stubborn_action(env.action_masks(), favourite))
        if env._mgr.active_card_choice is not choice:
            return history
    raise AssertionError(f"{choice.prompt!r} still pending after {choice.max_choices + 1} decisions")


def assert_only_grows(history):
    for before, after in zip(history, history[1:]):
        assert set(before) < set(after), f"selection shrank or stalled: {before} -> {after}"


@pytest.mark.parametrize("potion, prompt", [(_gamblers_brew, "discard"), (_ashwater, "exhaust")])
def test_optional_hand_selection_finishes_without_deselecting(potion, prompt):
    env = run_env()
    combat = enter_first_combat(env)
    potion(combat, combat.player, None)
    choice = combat.pending_choice
    assert prompt in choice.prompt and choice.min_choices == 0

    # Shaped final seed 20007 alternated card 4 between {3, 4} and {4}.
    history = finish_choice(env, favourite=4)
    assert_only_grows(history)
    assert combat.pending_choice is None
    env.close()


def capped_upgrade_choice(env):
    """Whetstone's 'Choose 2 cards to upgrade' with two Angers at indexes 6 and 8."""
    player_state = env._mgr.run_state.player
    deck = player_state.deck
    deck[6:6] = [create_card(CardId.ANGER)]
    deck[8:8] = [create_card(CardId.ANGER)]
    upgraded = []
    assert player_state.request_deck_choice(
        prompt="Choose 2 cards to upgrade", cards=list(deck),
        resolver=lambda cards: upgraded.extend(cards), min_count=2, max_count=2)
    return env._mgr.run_state.pending_choice, upgraded


def test_capped_deck_selection_masks_new_cards_and_keeps_indexes():
    env = run_env()
    choice, upgraded = capped_upgrade_choice(env)
    env.step(1 + 6)
    env.step(1 + 8)
    assert choice.selected_indices == {6, 8}

    mask = env.action_masks()
    assert mask[0] == 1, "confirmation must be legal once two cards are selected"
    assert np.flatnonzero(mask).tolist() == [0], "no third card, and no deselection"

    listed = [a for a in env._mgr.get_available_actions() if a["action"] == "choose"]
    assert [a["index"] for a in listed] == [6, 8], "listing keeps original option indexes"
    assert all(a["card_id"] == "ANGER" and a["selected"] for a in listed)

    env.step(0)
    assert env._mgr.run_state.pending_choice is None
    assert [card.card_id for card in upgraded] == [CardId.ANGER, CardId.ANGER]
    env.close()


def test_rejected_choice_action_fails_loudly_instead_of_repeating():
    env = run_env()
    choice, _ = capped_upgrade_choice(env)
    env.step(1 + 6)
    env.step(1 + 8)
    # Shaped best seed 20026 sent this rejected third selection for 1,881 decisions.
    with pytest.raises(RuntimeError, match="changed nothing"):
        env.step(1 + 1)
    assert choice.selected_indices == {6, 8}
    env.close()


def test_gnarled_hammer_enchantment_finishes():
    env = run_env()
    player_state = env._mgr.run_state.player
    attacks = [card for card in player_state.deck if card.card_type == CardType.ATTACK]
    player_state.enchant_selected_cards("Sharp", 3, 3, cards=attacks, min_count=0)
    choice = env._mgr.run_state.pending_choice
    assert choice is not None and choice.max_choices == 3

    history = finish_choice(env, favourite=1)
    assert_only_grows(history)
    assert sum(card.has_enchantment("Sharp") for card in player_state.deck) == 3
    env.close()


class _PickCardsEvent(EventModel):
    event_id = "TEST_PICK_CARDS"

    def generate_initial_options(self, run_state):
        return [EventOption(option_id="pick", label="Pick")]

    def choose(self, run_state, option_id):
        return self.request_multi_card_choice(
            prompt="Choose any number of cards", cards=list(run_state.player.deck[:4]),
            source_pile="deck", resolver=lambda cards: None, min_count=0, max_count=4, allow_skip=True)


def test_event_card_choice_is_visible_to_the_entity_policy_and_finishes():
    env = run_env()
    mgr = env._mgr
    mgr._phase = RunManager.PHASE_EVENT
    mgr._event_model = _PickCardsEvent()
    mgr._event_options = mgr._event_model.generate_initial_options(mgr.run_state)
    mgr._do_event_choice({"option_id": "pick"})
    choice = mgr.active_card_choice
    assert choice is mgr._event_model.pending_choice

    obs = E.encode_run_entities(env, env.action_masks())
    assert obs["pending_choice"][0] and obs["choice_mask"].sum() == 4

    env.step(1 + 2)
    obs = E.encode_run_entities(env, env.action_masks())
    assert obs["choice_features"][2][5] == 1.0, "selected cards are shown"
    assert env.action_masks()[1 + 2] == 0

    assert_only_grows(finish_choice(env, favourite=3))
    env.close()


def test_choice_with_fewer_options_than_minimum_can_confirm():
    cards = [create_card(CardId.STRIKE_IRONCLAD)]
    choice = PendingCardChoice(prompt="Choose 2 cards", options=[CardChoiceOption(c, "deck") for c in cards],
                               resolver=lambda _: None, min_choices=2, max_choices=2)
    assert not choice.can_confirm()
    assert choice.toggle(0)
    assert choice.can_confirm()
    assert not choice.can_select(0)


# ---------------------------------------------------------------------------
# Loop detection
# ---------------------------------------------------------------------------

def test_loop_detector_reports_shortest_period_after_three_repeats():
    detector = LoopDetector()
    assert [detector.push(key) for key in "aab"] == [None, None, None]
    assert [detector.push(key) for key in "bb"] == [None, 1]

    detector.reset()
    assert [detector.push(key) for key in "xyxyx"] == [None] * 5
    assert detector.push("y") == 2

    detector.reset()
    assert all(detector.push(key) is None for key in "abcdefghabcdefgh")


def experiment_config(tmp_path, deterministic=True):
    path = tmp_path / "loop.json"
    path.write_text(json.dumps({
        "name": "loop-test", "task": "act1", "seeds": {"mode": "fixed", "values": [42], "schedule_seed": 1},
        "training": {"n_envs": 1, "n_steps": 8, "batch_size": 8, "max_episode_steps": 50},
        "evaluation": {"mode": "seed_list", "seeds": [7], "deterministic": deterministic},
        "reward": "sparse"}))
    return load_config(path)


def freeze_simulator(monkeypatch):
    """Make every action leave the run unchanged, like the audited stalls."""
    def step(self, action):
        self._step_count += 1
        return self._encode_obs(), 0.0, False, False, self._build_info()
    monkeypatch.setattr(STS2RunEnv, "step", step)


def test_deterministic_evaluation_stops_a_repeated_state_as_a_stall(tmp_path, monkeypatch):
    config = experiment_config(tmp_path)
    env = ScenarioEnv(config, evaluation=True)
    env.reset()
    freeze_simulator(monkeypatch)
    action = int(np.flatnonzero(env.action_masks())[0])
    results = [env.step(action) for _ in range(3)]
    _, reward, terminated, truncated, info = results[-1]
    assert [r[3] for r in results] == [False, False, True]
    assert not terminated and info["stalled"] and not info["timed_out"]
    assert info["loop"] == dict(period=1, start=1, detected=3)
    assert reward == config["reward"]["timeout"]
    trace = info["decision_trace"]
    assert [r["decision"] for r in trace] == [1, 2, 3]
    assert all(r["action"] == action and not r["changed"] for r in trace)
    assert len({r["state"] for r in trace}) == 1


def test_identical_combat_turns_are_left_to_the_combat_turn_cap(tmp_path, monkeypatch):
    # Saved shaped model, seed 10055: End Turn with nothing changing, ended by the 200-turn cap.
    env = ScenarioEnv(experiment_config(tmp_path), evaluation=True)
    env.reset()
    while env.env._mgr.phase != RunManager.PHASE_COMBAT:
        env.step(int(np.flatnonzero(env.action_masks())[0]))

    def end_turn_without_change(self, action):
        self._step_count += 1
        self._mgr.get_combat_state().turn_count += 1
        return self._encode_obs(), 0.0, False, False, self._build_info()
    monkeypatch.setattr(STS2RunEnv, "step", end_turn_without_change)
    for _ in range(6):
        _, _, _, truncated, info = env.step(0)
        assert not truncated and not info["stalled"]
    assert env.loop is None


def test_sampled_training_episodes_record_loops_without_ending(tmp_path, monkeypatch):
    env = ScenarioEnv(experiment_config(tmp_path), evaluation=False)
    env.reset()
    freeze_simulator(monkeypatch)
    action = int(np.flatnonzero(env.action_masks())[0])
    for _ in range(4):
        _, _, terminated, truncated, info = env.step(action)
        assert not (terminated or truncated)
    assert env.loop["period"] == 1 and env.loop["detected"] == 3


def test_evaluation_reports_stalls_separately_and_saves_their_traces(tmp_path, monkeypatch):
    config = experiment_config(tmp_path)
    freeze_simulator(monkeypatch)
    summary, rows = runner.evaluate(config, 1)
    assert (summary["stalls"], summary["timeouts"], summary["losses"], summary["wins"]) == (1, 0, 0, 0)
    assert summary["loops_detected"] == 1
    assert summary["by_outcome"]["stalls"]["floor"] == rows[0]["floor"]

    runner.save_evaluation(tmp_path, "final-results", summary, rows)
    header = (tmp_path / "final-results-episodes.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "stalled" in header and "loop_start" in header and "decision_trace" not in header
    traces = [json.loads(line) for line in (tmp_path / "final-results-traces.jsonl").read_text().splitlines()]
    assert len(traces) == 1 and traces[0]["outcome"] == "stall"
    # Random actions on one frozen state repeat once their (state, action) pairs cycle.
    assert traces[0]["trace"][-1]["decision"] == rows[0]["decisions"] == traces[0]["decisions"]
    assert traces[0]["loop_start"] == rows[0]["loop_start"] is not None
    assert "stalls" in runner.format_summary(summary)


# ---------------------------------------------------------------------------
# Live watch mode
# ---------------------------------------------------------------------------

def reward_screen(request_id):
    return dict(type="reward_screen", request_id=request_id,
                options=[dict(index=0, action="pick_reward", type="Card", enabled=True)])


def test_live_decision_key_ignores_transport_request_ids():
    wire = {"action": "choose", "index": 0}
    assert player.decision_key(reward_screen("a"), dict(wire, request_id="a")) == \
        player.decision_key(reward_screen("b"), dict(wire, request_id="b"))
    assert player.decision_key(reward_screen("a"), wire) != player.decision_key(reward_screen("a"), {"action": "skip"})


def test_live_reward_skip_alternation_is_a_two_decision_cycle():
    # Watch trace 20261019-195005: CardReward opened, skipped, and reopened 43 times.
    card_reward = dict(type="card_reward", cards=[dict(id="ANGER")], can_skip=True)
    detector = LoopDetector()
    periods = []
    for i in range(6):
        state, wire = ((reward_screen(f"r{i}"), {"action": "choose", "index": 0}) if i % 2 == 0
                       else (dict(card_reward, request_id=f"c{i}"), {"action": "skip"}))
        periods.append(detector.push(player.decision_key(state, wire)))
    assert periods == [None] * 5 + [2]


class _RepeatingGame:
    """A game that keeps asking the same question, like the End Turn and Wound stalls."""

    def __init__(self, *args, **kwargs):
        self.sent = []
        self.requests = 0

    def connect(self):
        pass

    def receive_state(self):
        self.requests += 1
        return reward_screen(f"request-{self.requests}")

    def send_action(self, action):
        self.sent.append(action)

    def disconnect(self):
        pass


def test_watch_worker_stops_repeated_live_decisions(tmp_path, monkeypatch):
    game = _RepeatingGame()
    monkeypatch.setattr(player, "STS2GameClient", lambda *a, **k: game)
    agent = EntityAgent(SpireFormer(SpireFormerConfig(d_model=32, layers=1, heads=4, n_actions=157)).eval(),
                        torch.device("cpu"))
    playback = Playback(paused=False, delay=0)
    player.worker(playback, agent, dict(seed="1", ascension=0, character="Ironclad"), tmp_path, 100)

    assert "repeated the same decision 3 times" in playback.error
    decisions = [a for a in game.sent if a["action"] == "choose"]
    assert len(decisions) == 2, "the third repeat is never sent"
    records = [json.loads(line) for line in (tmp_path / "decisions.jsonl").read_text().splitlines()]
    assert len({r["fingerprint"] for r in records}) == 1
    assert records[-1]["sent"] is None and records[-1]["stopped"] == "loop period 1"

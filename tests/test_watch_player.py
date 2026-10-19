import copy
import threading

import numpy as np

from sts2_env.watch.adapter import WatchAdapter
from sts2_env.watch.control import Playback, Stopped
from sts2_env.gym_env.run_env import STS2RunEnv, _MAP_START, _SHOP_START, _CARD_RWD_START


def state(kind="map_select", floor=0, room="Map"):
    return dict(type=kind, nodes=[dict(index=0, type="Monster")],
                run_state=dict(act_index=0, floor=floor, act_floor=floor, hp=80,
                               max_hp=80, gold=99, deck_size=10, relic_count=1,
                               potion_count=0, max_potion_slots=3, ascension=0, room=room))


def test_live_run_suffix_matches_training_encoder():
    env = STS2RunEnv(act_count=1)
    obs, _ = env.reset(options={"run_seed": 42})
    run = env._mgr.run_state
    payload = state()
    payload["run_state"].update(hp=run.player.current_hp, max_hp=run.player.max_hp,
                                gold=run.player.gold, deck_size=len(run.player.deck),
                                relic_count=len(run.relics), max_potion_slots=run.player.max_potion_slots)
    actual = WatchAdapter().encode(payload)
    np.testing.assert_array_equal(actual, obs)


def test_live_actions_use_training_slots_and_game_wire_indices():
    adapter = WatchAdapter()
    payload = state()
    payload["nodes"] = [dict(index=0, type="Monster"), dict(index=1, type="RestSite")]
    choices, automatic = adapter.choices(payload)
    assert automatic is None
    assert choices[_MAP_START + 1][0] == {"action": "choose", "index": 1}
    payload.update(type="shop", options=[dict(index=0, label="Leave"), dict(index=1, label="Card")])
    assert adapter.choices(payload)[0][_SHOP_START][0]["index"] == 0
    payload.update(type="card_reward", cards=[dict(id="BASH")], can_skip=True)
    assert adapter.choices(payload)[0][_CARD_RWD_START + 3][0] == {"action": "skip"}


def test_single_step_releases_only_one_pending_decision():
    playback = Playback(delay=0)
    first_done = threading.Event()
    second_done = threading.Event()
    def run():
        try:
            playback.gate(state(), dict(label="first", source="test"))
            first_done.set()
            playback.gate(state(floor=1), dict(label="second", source="test"))
            second_done.set()
        except Stopped:
            pass
    thread = threading.Thread(target=run)
    thread.start()
    playback.control({"action": "step"})
    assert first_done.wait(2)
    assert not second_done.wait(0.05)
    assert playback.count == 1
    playback.control({"action": "stop"})
    thread.join(2)
    assert not thread.is_alive()


def test_next_room_pauses_before_new_room_action():
    playback = Playback(paused=False, delay=0)
    playback.state = state()
    playback.control({"action": "next_room"})
    completed = threading.Event()
    def run():
        try:
            playback.gate(state(floor=1), dict(label="enter", source="test"))
            completed.set()
        except Stopped:
            pass
    thread = threading.Thread(target=run)
    thread.start()
    assert not completed.wait(0.05)
    assert playback.paused and playback.count == 0
    playback.control({"action": "step"})
    assert completed.wait(2)
    thread.join(2)

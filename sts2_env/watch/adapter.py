"""Translate live decisions into the Act 1 policy's 151 inputs / 157 actions.

Uses the training encoder's scales and action layout. Real game UI-only screens
are labelled explicitly; those screens were automatic transitions in training.
"""
from __future__ import annotations

import numpy as np

from sts2_env.bridge.state_adapter import StateAdapter
from sts2_env.bridge.agent_runner import _pick_reward_screen_option, _pick_crystal_sphere_option
from sts2_env.gym_env import run_env as layout


class WatchAdapter:
    def __init__(self, observation_size=151):
        if observation_size not in (131, 151):
            raise ValueError(f"Unsupported model observation size: {observation_size}")
        self.size = observation_size
        self.combat = StateAdapter()
        self.last_combat = np.zeros(131, dtype=np.float32)

    def encode(self, state, selected=()):
        # The flat observation has no selection features, in training or here.
        kind = state["type"]
        run = state.get("run_state", {})
        phase = {"map_select": 0, "combat_action": 1, "card_reward": 2,
                 "reward_screen": 2, "card_bundle": 2, "boss_relic": 3,
                 "shop": 4, "rest_site": 5, "event": 6,
                 "crystal_sphere": 6, "treasure": 7}.get(kind, 2)
        if kind == "card_select":
            phase = 1 if run.get("in_combat") else {"RestSite": 5, "Event": 6}.get(run.get("room"), 2)
        snapshot = state if kind == "combat_action" else state.get("combat_snapshot")
        if snapshot and "player" in snapshot:
            self.last_combat = self.combat.encode_observation(snapshot)
        obs = np.zeros(self.size, dtype=np.float32)
        obs[:131] = self.last_combat
        if self.size == 131:
            return obs
        obs[131:140] = [run.get("act_index", 0) / layout.OBS_CURRENT_ACT_SCALE,
                        run.get("floor", 0) / layout.OBS_TOTAL_FLOOR_SCALE,
                        run.get("act_floor", 0) / layout.OBS_ACT_FLOOR_SCALE,
                        run.get("hp", 0) / max(run.get("max_hp", 1), 1),
                        run.get("gold", 0) / layout.OBS_GOLD_SCALE,
                        run.get("deck_size", 0) / layout.OBS_DECK_SIZE_SCALE,
                        run.get("relic_count", 0) / layout.OBS_RELIC_COUNT_SCALE,
                        run.get("potion_count", 0) / max(run.get("max_potion_slots", 1), 1),
                        run.get("max_potion_slots", 0) / layout.OBS_MAX_POTION_SLOTS_SCALE]
        obs[140 + phase] = 1
        obs[148] = run.get("ascension", 0) / layout.OBS_ASCENSION_SCALE
        obs[149] = float(run.get("room") == "Elite")
        obs[150] = float(run.get("room") == "Boss")
        return obs

    def choices(self, state):
        """Return policy action -> wire action and UI label, or a UI-only action."""
        kind = state["type"]
        choices = {}

        def add(action, wire, label):
            choices[action] = (wire, label)

        if kind == "combat_action":
            for action in np.flatnonzero(self.combat.compute_action_mask(state)):
                action = int(action)
                decoded = self.combat.decode_action(action, state)
                if action == 0:
                    add(action, {"action": "end_turn"}, "End turn")
                elif decoded.get("out_of_hand") or decoded.get("type") == "POTION":
                    slot = decoded.get("slot", decoded.get("potion_slot", -1))
                    target = decoded.get("target_index", -1)
                    add(action, {"action": "potion", "slot": slot, "target_index": target}, f"Potion {slot + 1} → target {target}")
                else:
                    index = decoded["card_index"]
                    target = decoded.get("target_index", -1)
                    card = state.get("hand", [])[index]
                    name = str(card.get("id", index)).replace("_", " ").title().removesuffix(" Ironclad")
                    enemies = state.get("enemies", [])
                    target_name = str(enemies[target].get("id", "Enemy")).replace("_", " ").title() if 0 <= target < len(enemies) else "self / untargeted"
                    add(action, {"action": "play", "card_index": index, "target_index": target}, f"{name} (card {index + 1}) → {target_name}")
        elif kind == "card_select":
            for i, card in enumerate(state.get("cards", [])[:layout._COMBAT_SIZE - 1]):
                add(1 + i, {"action": "choose", "index": i}, f"Select {card.get('id', i)}")
            if state.get("min_select", 0) == 0:
                add(0, {"action": "skip"}, "Skip selection")
        elif kind == "card_reward":
            for i, card in enumerate(state.get("cards", [])[:6]):
                action = layout._CARD_RWD_START + i if i < 3 else layout._CARD_RWD_EXTRA_START + i - 3
                add(action, {"action": "choose", "index": i}, f"Take {card.get('id', i)}")
            if state.get("can_skip", True):
                add(layout._CARD_RWD_START + 3, {"action": "skip"}, "Skip card reward")
        elif kind in {"map_select", "shop", "rest_site", "event", "boss_relic", "treasure"}:
            start, size = {"map_select": (layout._MAP_START, layout._MAP_SIZE),
                           "shop": (layout._SHOP_START, layout._SHOP_SIZE),
                           "rest_site": (layout._REST_START, layout._REST_SIZE),
                           "event": (layout._EVENT_START, layout._EVENT_SIZE),
                           "boss_relic": (layout._BOSS_RELIC_START, layout._BOSS_RELIC_SIZE),
                           "treasure": (layout._TREASURE_START, 1)}[kind]
            options = state.get("nodes" if kind == "map_select" else "options", [])
            # Proceed buttons are UI transitions; they were automatic in simulation.
            if options and all(o.get("action") == "proceed" for o in options):
                return {}, ({"action": "choose", "index": options[0].get("index", 0)}, "Proceed", "UI navigation")
            for i, option in enumerate(options[:size]):
                if option.get("enabled", True):
                    index = option.get("index", i)
                    label = option.get("label", option.get("id", option.get("type", str(index))))
                    add(start + i, {"action": "choose", "index": index}, f"{kind}: {label}")
        elif kind in {"reward_screen", "crystal_sphere", "card_bundle"}:
            if kind == "reward_screen":
                index = _pick_reward_screen_option(state)
                wire = {"action": "skip"} if index is None else {"action": "choose", "index": index}
            elif kind == "crystal_sphere":
                wire = {"action": "choose", "index": _pick_crystal_sphere_option(state)}
            else:
                wire = {"action": "choose", "index": 0}
            return {}, (wire, f"{kind}: {wire}", "UI transition / game-specific screen (not trained)")
        else:
            raise ValueError(f"Unsupported live decision: {kind}")
        if not choices:
            raise ValueError(f"No supported legal actions for {kind}")
        return choices, None

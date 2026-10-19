"""Encode watch-mode bridge objects with the existing SpireFormer vocabulary.

This is a live transfer adapter, not a claim of simulator/game parity. Unsupported
game-specific screens retain the viewer's explicitly labelled UI handling.
"""
from __future__ import annotations

import re
import numpy as np

from sts2_env.entity import observation as E
from sts2_env.core.enums import CardId
from sts2_env.cards.factory import create_card
from sts2_env.watch.adapter import WatchAdapter

# Bridge payload version this adapter reads (bridge_mod/Watch/WatchSession.cs).
ENTITY_SCHEMA = 3


def enum_name(value):
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(value or ""))
    return text.replace(" ", "_").upper()


def enum_value(values, name):
    name = enum_name(name)
    return next((v for v in values if v.name == name), None)


def fill_card(card, out, *, hand=False):
    """Live numbers take priority; static known-card metadata fills older offers."""
    cid = E._CARD_BY_NAME.get(card.get("id"))
    reference = create_card(cid, upgraded=bool(card.get("upgraded"))) if cid else None
    if reference:
        E._card_features(None, reference, out, in_hand=False)
    out[0] = max(card.get("cost", reference.cost if reference else 0), 0) / 5
    out[1] = float(card.get("cost_x", reference.has_energy_cost_x if reference else False))
    out[2] = card.get("base_damage", reference.base_damage or 0 if reference else 0) / 50
    out[3] = card.get("base_block", reference.base_block or 0 if reference else 0) / 50
    out[4] = float(bool(card.get("upgraded")))
    out[5] = float(hand and bool(card.get("playable")))
    out[8] = card.get("star_cost", reference.star_cost if reference else 0) / 5
    # Shown numbers are only sent for cards in combat; elsewhere they equal the base ones.
    out[9] = card.get("damage", out[2] * 50) / 50
    out[10] = card.get("block", out[3] * 50) / 50
    if "keywords" in card:
        keywords = {enum_name(k) for k in card["keywords"]}
        out[6] = float("EXHAUST" in keywords)
        out[7] = float("ETHEREAL" in keywords)
        out[11] = float("RETAIN" in keywords or bool(card.get("retain_this_turn")))
        out[12] = float("INNATE" in keywords)
        out[13] = float("SLY" in keywords or bool(card.get("sly_this_turn")))
        out[14] = float("UNPLAYABLE" in keywords or (card.get("cost", 0) < 0 and not card.get("cost_x")))
        out[15] = float("ETERNAL" in keywords)
    types, targets = E.CARD_TYPE_START, E.CARD_TYPE_START + len(E._CARD_TYPES)
    if "type" in card:
        out[types:targets] = 0
        E._one_hot(out, types, E._CARD_TYPES, enum_value(E._CARD_TYPES, card["type"]))
    if "target" in card:
        out[targets:] = 0
        name = "ANY_ALLY" if enum_name(card["target"]) == "ANY_PLAYER" else card["target"]
        E._one_hot(out, targets, E._TARGET_TYPES, enum_value(E._TARGET_TYPES, name))


def fill_tags(card, ids, amounts):
    """The bridge sends at most one enchantment ({id, amount}) and one affliction ID per card."""
    enchantment = card.get("enchantment")
    enchantments = {enchantment["id"]: enchantment.get("amount", 1)} if enchantment else {}
    E._fill_card_tags(ids, amounts, enchantments, card.get("affliction"))


def fill_powers(powers, ids, amounts):
    for i, power in enumerate(powers[:E.MAX_POWERS]):
        value = E.PowerId.__members__.get(enum_name(power.get("id")))
        ids[i] = E._POWER_INDEX.get(value, 0)
        amounts[i] = np.clip(power.get("amount", 0) / 10, -10, 10)


def map_routes(nodes):
    """Route stats for the live map, computed by the training encoder's own code."""
    graph = {(n["col"], n["row"]): (enum_value(E.MapPointType, n["type"]),
                                    [(c["col"], c["row"]) for c in n.get("children", [])]) for n in nodes}
    return E.route_stats(graph)


class EntityWatchAdapter(WatchAdapter):
    def choices(self, state):
        choices, automatic = super().choices(state)
        if state["type"] == "card_select":
            choices = {a: choice for a, choice in choices.items() if a <= E.MAX_CHOICES}
        return choices, automatic

    def encode(self, state, selected=()):
        """``selected`` holds card indexes already picked on a card_select screen."""
        if state.get("entity_schema") != ENTITY_SCHEMA:
            raise ValueError(f"SpireFormer needs the updated entity-enabled bridge mod (entity_schema={ENTITY_SCHEMA})")
        obs = E.empty_observation()
        run = state["run_state"]
        kind = state["type"]
        in_combat = kind == "combat_action" or (kind == "card_select" and run.get("in_combat"))
        combat = state if kind == "combat_action" else state.get("combat_snapshot", {})
        g = obs["global"]
        g[0] = run["hp"] / max(run["max_hp"], 1)
        g[1] = run["max_hp"] / 100
        if in_combat:
            if "player" not in combat:
                raise ValueError("Live combat snapshot is missing; refusing incomplete model input")
            player = combat["player"]
            g[2:7] = [player.get("block", 0) / 50, player.get("energy", 0) / 10,
                       player.get("max_energy", 3) / 10, player.get("stars", 0) / 10, combat.get("round", 0) / 20]
            g[7:11] = [len(combat.get("hand", [])) / E.MAX_HAND_SIZE,
                        combat.get("draw_pile_count", 0) / 20, combat.get("discard_pile_count", 0) / 20,
                        combat.get("exhaust_pile_count", 0) / 20]
            fill_powers(player.get("powers", []), obs["player_power_ids"], obs["player_power_amounts"])
            for i, card in enumerate(combat.get("hand", [])[:E.MAX_HAND_SIZE]):
                obs["hand_ids"][i], obs["hand_mask"][i] = E._card_index(card["id"]), True
                fill_card(card, obs["hand_features"][i], hand=True)
                fill_tags(card, obs["hand_tag_ids"][i], obs["hand_tag_amounts"][i])
                for j, damage in enumerate(card.get("target_damage", [])[:E.MAX_ENEMIES]):
                    obs["hand_target_damage"][i, j] = damage / 50
            for i, enemy in enumerate(combat.get("enemies", [])[:E.MAX_ENEMIES]):
                obs["enemy_ids"][i], obs["enemy_mask"][i] = E.hash_bucket(enemy["id"]), True
                f = obs["enemy_features"][i]
                f[:5] = [float(enemy.get("is_alive", True)), enemy["hp"] / max(enemy["max_hp"], 1),
                         enemy["hp"] / 100, enemy["max_hp"] / 100, enemy.get("block", 0) / 50]
                intents = enemy.get("intents", [dict(type=enemy.get("intent", "UNKNOWN"), damage=enemy.get("intent_damage", 0), hits=enemy.get("intent_hits", 1))])
                attacks = [v for v in intents if enum_name(v["type"]) in {"ATTACK", "MULTI_ATTACK", "DEATH_BLOW"}]
                if attacks:
                    f[5:8] = [attacks[0].get("damage", 0) / 30, attacks[0].get("hits", 1) / 5,
                              sum(v.get("damage", 0) * v.get("hits", 1) for v in attacks) / 60]
                f[8] = len(intents) / 3
                for intent in intents:
                    E._one_hot(f, 9, E._INTENT_TYPES, enum_value(E._INTENT_TYPES, intent["type"]))
                fill_powers(enemy.get("powers", []), obs["enemy_power_ids"][i], obs["enemy_power_amounts"][i])
            piles = [(name, combat.get(name + "_pile", [])) for name in ("draw", "discard", "exhaust")]
        else:
            piles = [("deck", state["deck"])]
        slot = 0
        for name, cards in piles:
            for card in cards:
                if slot == E.MAX_PILE_CARDS:
                    break
                obs["pile_ids"][slot], obs["pile_kinds"][slot], obs["pile_mask"][slot] = E._card_index(card["id"]), E._PILES[name], True
                fill_card(card, obs["pile_features"][slot])
                fill_tags(card, obs["pile_tag_ids"][slot], obs["pile_tag_amounts"][slot])
                slot += 1
        E._fill_relics(obs, state["relics"])
        for i, potion in enumerate(state.get("potions", [])):
            slot = potion.get("slot", i)
            if slot >= E.MAX_POTION_SLOTS:
                continue
            obs["potion_ids"][slot], obs["potion_mask"][slot] = E.hash_bucket(potion["id"]), True
            obs["potion_features"][slot][0] = 1
            E._one_hot(obs["potion_features"][slot], 1, E._POTION_TARGETS, enum_value(E._POTION_TARGETS, potion.get("target_type", potion.get("target"))))
        phase = {"map_select": 0, "combat_action": 1, "card_reward": 2, "boss_relic": 3,
                 "shop": 4, "rest_site": 5, "event": 6, "treasure": 7}.get(kind, 2)
        if kind == "card_select":
            phase = 1 if in_combat else {"RestSite": 5, "Event": 6}.get(run.get("room"), 2)
            obs["pending_choice"][0] = True
            # Same globals as PendingCardChoice: can_confirm, is_multi, selected count, maximum.
            minimum, maximum = state.get("min_select", 0), state.get("max_select", 1)
            can_confirm = len(selected) >= min(minimum, len(state.get("cards", [])))
            g[11:16] = [1, float(can_confirm), float(minimum != 1 or maximum != 1), len(selected) / 5, maximum / 5]
            for i, card in enumerate(state.get("cards", [])[:E.MAX_CHOICES]):
                obs["choice_ids"][i], obs["choice_mask"][i] = E._card_index(card["id"]), True
                obs["choice_kinds"][i] = E._PILES.get(str(card.get("source_pile", "deck")).lower(), 0)
                fill_card(card, obs["choice_features"][i])
                fill_tags(card, obs["choice_tag_ids"][i], obs["choice_tag_amounts"][i])
                obs["choice_features"][i][5] = float(i in selected)
        r = obs["run"]
        r[:12] = [run.get("act_index", 0) / 3, run.get("floor", 0) / 50, run.get("act_floor", 0) / 20,
                  g[0], g[1], run.get("gold", 0) / 500, run.get("deck_size", 0) / 40,
                  run.get("relic_count", 0) / 30, run.get("potion_count", 0) / max(run.get("max_potion_slots", 1), 1),
                  run.get("max_potion_slots", 0) / 5, run.get("ascension", 0) / 10, float(in_combat)]
        E._one_hot(r, 12, E._ROOM_TYPES, enum_value(E._ROOM_TYPES, run.get("room")))
        r[12 + len(E._ROOM_TYPES) + phase] = 1
        r[-1] = float(kind == "card_select" and not in_combat)
        if run.get("boss"):
            obs["boss_id"][0] = E.hash_bucket(f"boss:{E.game_id(run['boss'])}")
        # The bridge sends the map and the nodes reachable next in every phase.
        routes = map_routes(state.get("map_nodes", []))
        obs["route_ahead"][:] = E.route_features(routes, [(c["col"], c["row"]) for c in state.get("next_map_coords", [])])
        if kind not in {"combat_action", "card_select"}:
            self.fill_options(obs, state, routes)
        return obs

    def fill_options(self, obs, state, routes):
        choices, automatic = self.choices(state)
        if automatic:
            return
        kind = state["type"]
        options = state.get("nodes" if kind == "map_select" else "options", [])
        for i, (slot, (wire, _)) in enumerate(list(choices.items())[:E.MAX_OPTIONS]):
            index = wire.get("index", 0)
            option = next((o for j, o in enumerate(options) if o.get("index", j) == index), {})
            opt_kind = {"map_select": "move", "card_reward": "skip_card" if wire["action"] == "skip" else "pick_card",
                        "boss_relic": "pick_boss_relic", "rest_site": "rest_option", "event": "event_choice", "treasure": "collect"}.get(kind, option.get("action"))
            if opt_kind not in E._KIND_INDEX:
                raise ValueError(f"Unsupported entity option: {kind}/{opt_kind}")
            obs["option_slots"][i], obs["option_kinds"][i], obs["option_mask"][i] = slot, E._KIND_INDEX[opt_kind], True
            card = state.get("cards", [])[index] if kind == "card_reward" and wire["action"] != "skip" else {}
            item = option.get("item_id", option.get("id", ""))
            obs["option_cards"][i] = E._card_index(card.get("id") or (item if opt_kind == "buy_card" else None))
            obs["option_relics"][i] = E._relic_index(item) if opt_kind in {"pick_boss_relic", "buy_relic"} else 0
            obs["option_potions"][i] = E.hash_bucket(item) if opt_kind == "buy_potion" else 0
            f = obs["option_features"][i]
            price = option.get("price", 0)
            f[0:4] = [price / 100, (state["run_state"].get("gold", 0) - price) / 300 if price else 0,
                      float(option.get("on_sale", False)), float(card.get("upgraded", option.get("upgraded", False)))]
            rarity = card.get("rarity", option.get("rarity"))
            E._one_hot(f, 7, E._RARITIES, enum_name(rarity))
            name = ""
            if opt_kind == "move":
                name = "point:" + enum_name(option.get("type"))
                f[6] = option.get("row", 0) / 20
                f[E.ROUTE_START:] = E.route_features(routes, [(option.get("col"), option.get("row"))])
            elif opt_kind == "rest_option":
                name = "rest:" + enum_name(option.get("id"))
            elif opt_kind == "event_choice":
                name = f"event:{option.get('event_id', '')}:{option.get('option_id', '')}"
            obs["option_names"][i] = E.hash_bucket(name)

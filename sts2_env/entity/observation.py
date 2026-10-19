"""Entity (object-list) observations for fights and whole runs.

The flat encoders squeeze the game into a fixed vector of numbers.  This
encoder instead describes each card, enemy, power, potion, relic, pending
choice and on-screen option as its own item: integer IDs (looked up in learned
embedding tables by the model) plus a few numeric details.

* In a fight: hand, draw/discard/exhaust piles, enemies (with intents and
  powers), the player's powers, potions, relics and any pending card choice.
* Outside a fight: the deck (as card items), relics, potions, run state, and
  one **option** item per legal choice on screen -- a map node (with a summary
  of what lies down that path), an offered card/relic/potion, a shop item with
  its price, a rest-site or event option.

Every list is padded to a fixed capacity and paired with a boolean mask so
batches can be stacked; the model ignores padded slots.  The environments'
action spaces and legality masks are unchanged: each option item records the
action index it stands for, and the policy scores it there.
"""

from __future__ import annotations

import re
import warnings
import zlib
from typing import NamedTuple

import gymnasium
import numpy as np
from gymnasium import spaces

from sts2_env.cards.base import CardInstance
from sts2_env.core.combat import CombatState
from sts2_env.core.constants import MAX_ENEMIES, MAX_HAND_SIZE, MAX_POTION_SLOTS
from sts2_env.core.creature import Creature
from sts2_env.core.damage import preview_card_block, preview_card_damage, preview_intent_damage
from sts2_env.core.enums import (
    CardId,
    CardRarity,
    CardType,
    IntentType,
    MapPointType,
    PotionRarity,
    PotionTargetType,
    PowerId,
    RelicRarity,
    RoomType,
    TargetType,
)
from sts2_env.gym_env.run_env import _LAYOUT, _PHASE_INDEX, STS2RunEnv
from sts2_env.relics.base import RelicId
from sts2_env.run.run_manager import RunManager

MAX_PILE_CARDS = 60
MAX_POWERS = 12
# Well past what a run collects; _fill_relics warns if a run ever goes over.
MAX_RELICS = 48
MAX_CHOICES = 40
MAX_OPTIONS = 16
# Enchantments and afflictions per card (the game allows one of each).
MAX_CARD_TAGS = 3
HASH_BUCKETS = 1024

_CARD_INDEX = {cid: i + 1 for i, cid in enumerate(CardId)}
_CARD_BY_NAME = {cid.name: cid for cid in CardId}
_POWER_INDEX = {pid: i + 1 for i, pid in enumerate(PowerId)}
_RELIC_INDEX = {rid: i + 1 for i, rid in enumerate(RelicId)}
_RELIC_BY_NAME = {rid.name: rid for rid in RelicId}
_CARD_TYPES = list(CardType)
_TARGET_TYPES = list(TargetType)
_INTENT_TYPES = list(IntentType)
_POTION_TARGETS = list(PotionTargetType)
_ROOM_TYPES = list(RoomType)
_RARITIES = sorted({r.name for r in (*CardRarity, *RelicRarity, *PotionRarity)})
_PILES = {"draw": 1, "discard": 2, "exhaust": 3, "hand": 4, "deck": 5}
# What a map node's onward paths contain; counts summarise every path to the boss.
_PATH_TYPES = [MapPointType.MONSTER, MapPointType.ELITE, MapPointType.REST_SITE,
               MapPointType.SHOP, MapPointType.TREASURE, MapPointType.UNKNOWN]
_MONSTERS, _ELITES, _RESTS, _SHOPS = range(4)
# Real routes kept per node, so counts that go together stay together: fewest
# elites, most elites, most rest sites, most shops.  The whole count vector is
# the last tie-break, so any order of a node's children gives the same route.
_ROUTE_KEYS = [
    lambda v: (-v[_ELITES], v[_RESTS], -v[_MONSTERS], *v),
    lambda v: (v[_ELITES], v[_RESTS], *v),
    lambda v: (v[_RESTS], -v[_ELITES], *v),
    lambda v: (v[_SHOPS], -v[_ELITES], *v),
]
# Floors until the nearest of these on any route.
_DISTANCE_TYPES = [MapPointType.ELITE, MapPointType.REST_SITE, MapPointType.SHOP, MapPointType.TREASURE]
# Fewest and most of each type, the kept routes' counts, then nearness of each distance type.
ROUTE_FEATURES = (2 + len(_ROUTE_KEYS)) * len(_PATH_TYPES) + len(_DISTANCE_TYPES)

OPTION_KINDS = [
    "move", "pick_card", "skip_card", "reroll", "pick_potion", "skip_potion",
    "pick_relic_reward", "skip_relic", "pick_boss_relic", "leave_shop", "buy_card",
    "buy_relic", "buy_potion", "remove_card", "rest_option", "event_choice",
    "collect", "select_player",
]
_KIND_INDEX = {kind: i + 1 for i, kind in enumerate(OPTION_KINDS)}

VOCAB_SIZES = {
    "card": len(_CARD_INDEX) + 1,
    "power": len(_POWER_INDEX) + 1,
    "relic": len(_RELIC_INDEX) + 1,
    "monster": HASH_BUCKETS,
    "potion": HASH_BUCKETS,
    "name": HASH_BUCKETS,
    "card_tag": HASH_BUCKETS,
    "boss": HASH_BUCKETS,
    "pile": len(_PILES) + 1,
    "option_kind": len(OPTION_KINDS) + 1,
}

# Card details: 16 numbers, then card-type and target-type one-hots.
CARD_TYPE_START = 16
CARD_FEATURES = CARD_TYPE_START + len(_CARD_TYPES) + len(_TARGET_TYPES)
GLOBAL_FEATURES = 16
RUN_FEATURES = 12 + len(_ROOM_TYPES) + len(_PHASE_INDEX) + 1
ENEMY_FEATURES = 9 + len(_INTENT_TYPES)
POTION_FEATURES = 1 + len(_POTION_TARGETS)
# Option details: 7 numbers, rarity one-hot, then a map node's route summary.
ROUTE_START = 7 + len(_RARITIES)
OPTION_FEATURES = ROUTE_START + ROUTE_FEATURES

# name -> (shape, dtype); masks are bool, IDs int64, features float32.
OBS_SPEC: dict[str, tuple[tuple[int, ...], type]] = {
    "global": ((GLOBAL_FEATURES,), np.float32),
    "run": ((RUN_FEATURES,), np.float32),
    # The act's boss, and the routes on from where the player is now (every phase).
    "boss_id": ((1,), np.int64),
    "route_ahead": ((ROUTE_FEATURES,), np.float32),
    "player_power_ids": ((MAX_POWERS,), np.int64),
    "player_power_amounts": ((MAX_POWERS,), np.float32),
    "hand_ids": ((MAX_HAND_SIZE,), np.int64),
    "hand_features": ((MAX_HAND_SIZE, CARD_FEATURES), np.float32),
    "hand_mask": ((MAX_HAND_SIZE,), np.bool_),
    # Damage each hand card would deal to each enemy, as the game previews it.
    "hand_target_damage": ((MAX_HAND_SIZE, MAX_ENEMIES), np.float32),
    # Each card's enchantments and affliction: hashed names and their amounts.
    "hand_tag_ids": ((MAX_HAND_SIZE, MAX_CARD_TAGS), np.int64),
    "hand_tag_amounts": ((MAX_HAND_SIZE, MAX_CARD_TAGS), np.float32),
    "pile_ids": ((MAX_PILE_CARDS,), np.int64),
    "pile_kinds": ((MAX_PILE_CARDS,), np.int64),
    "pile_features": ((MAX_PILE_CARDS, CARD_FEATURES), np.float32),
    "pile_tag_ids": ((MAX_PILE_CARDS, MAX_CARD_TAGS), np.int64),
    "pile_tag_amounts": ((MAX_PILE_CARDS, MAX_CARD_TAGS), np.float32),
    "pile_mask": ((MAX_PILE_CARDS,), np.bool_),
    "enemy_ids": ((MAX_ENEMIES,), np.int64),
    "enemy_features": ((MAX_ENEMIES, ENEMY_FEATURES), np.float32),
    "enemy_power_ids": ((MAX_ENEMIES, MAX_POWERS), np.int64),
    "enemy_power_amounts": ((MAX_ENEMIES, MAX_POWERS), np.float32),
    "enemy_mask": ((MAX_ENEMIES,), np.bool_),
    "potion_ids": ((MAX_POTION_SLOTS,), np.int64),
    "potion_features": ((MAX_POTION_SLOTS, POTION_FEATURES), np.float32),
    "potion_mask": ((MAX_POTION_SLOTS,), np.bool_),
    "relic_ids": ((MAX_RELICS,), np.int64),
    "relic_mask": ((MAX_RELICS,), np.bool_),
    "choice_ids": ((MAX_CHOICES,), np.int64),
    "choice_kinds": ((MAX_CHOICES,), np.int64),
    "choice_features": ((MAX_CHOICES, CARD_FEATURES), np.float32),
    "choice_tag_ids": ((MAX_CHOICES, MAX_CARD_TAGS), np.int64),
    "choice_tag_amounts": ((MAX_CHOICES, MAX_CARD_TAGS), np.float32),
    "choice_mask": ((MAX_CHOICES,), np.bool_),
    "pending_choice": ((1,), np.bool_),
    "option_slots": ((MAX_OPTIONS,), np.int64),
    "option_kinds": ((MAX_OPTIONS,), np.int64),
    "option_cards": ((MAX_OPTIONS,), np.int64),
    "option_relics": ((MAX_OPTIONS,), np.int64),
    "option_potions": ((MAX_OPTIONS,), np.int64),
    "option_names": ((MAX_OPTIONS,), np.int64),
    "option_features": ((MAX_OPTIONS, OPTION_FEATURES), np.float32),
    "option_mask": ((MAX_OPTIONS,), np.bool_),
}


def hash_bucket(name: str | None) -> int:
    """Stable 1..HASH_BUCKETS-1 bucket for string IDs (0 is padding)."""
    if not name:
        return 0
    return zlib.crc32(name.encode("utf-8")) % (HASH_BUCKETS - 1) + 1


def observation_space() -> spaces.Dict:
    items = {}
    for key, (shape, dtype) in OBS_SPEC.items():
        if dtype is np.bool_:
            items[key] = spaces.MultiBinary(shape)
        elif dtype is np.int64:
            items[key] = spaces.Box(0, np.iinfo(np.int32).max, shape, np.int64)
        else:
            items[key] = spaces.Box(-np.inf, np.inf, shape, np.float32)
    return spaces.Dict(items)


def empty_observation() -> dict[str, np.ndarray]:
    return {key: np.zeros(shape, dtype) for key, (shape, dtype) in OBS_SPEC.items()}


def _one_hot(out: np.ndarray, start: int, values: list, value) -> None:
    if value in values:
        out[start + values.index(value)] = 1.0


def _card_index(card_id) -> int:
    if isinstance(card_id, str):
        card_id = _CARD_BY_NAME.get(card_id)
    return _CARD_INDEX.get(card_id, 0)


def _relic_index(relic_id) -> int:
    if isinstance(relic_id, str):
        relic_id = _RELIC_BY_NAME.get(relic_id)
    return _RELIC_INDEX.get(relic_id, 0)


def _card_features(combat: CombatState | None, card: CardInstance, out: np.ndarray, in_hand: bool) -> None:
    out[0] = max(card.cost, 0) / 5.0
    out[1] = float(card.has_energy_cost_x)
    out[2] = (card.base_damage or 0) / 50.0
    out[3] = (card.base_block or 0) / 50.0
    out[4] = float(card.upgraded)
    out[5] = float(in_hand and combat is not None and combat.can_play_card(card))
    out[6] = float(card.exhausts)
    out[7] = float(card.is_ethereal)
    out[8] = card.star_cost / 5.0
    # Damage and block as the card shows them now (Strength, Weak, Dexterity, Frail...).
    if combat is not None:
        out[9] = preview_card_damage(combat, card) / 50.0
        out[10] = preview_card_block(combat, card) / 50.0
    else:
        out[9], out[10] = out[2], out[3]
    # Retain and Sly include their this-turn-only versions.
    out[11] = float(card.is_retain or card.single_turn_retain)
    out[12] = float(card.is_innate)
    out[13] = float(card.is_sly)
    out[14] = float(card.is_unplayable)
    out[15] = float(not card.is_removable)  # Eternal
    _one_hot(out, CARD_TYPE_START, _CARD_TYPES, card.card_type)
    _one_hot(out, CARD_TYPE_START + len(_CARD_TYPES), _TARGET_TYPES, card.target_type)


def game_id(name: str) -> str:
    """The game's ID spelling (RoyallyApproved -> ROYALLY_APPROVED) so simulator and bridge names match."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).replace(" ", "_").upper()


def _fill_card_tags(ids: np.ndarray, amounts: np.ndarray, enchantments: dict, affliction: str | None) -> None:
    """A card's enchantments (with amounts) and affliction, as hashed names."""
    tags = [(f"enchantment:{game_id(name)}", amount) for name, amount in enchantments.items()]
    if affliction:
        tags.append((f"affliction:{game_id(affliction)}", 0))  # no affliction uses its amount
    for i, (name, amount) in enumerate(tags[:MAX_CARD_TAGS]):
        ids[i] = hash_bucket(name)
        amounts[i] = np.clip(amount / 10.0, -10.0, 10.0)


def _card_tags(card: CardInstance, ids: np.ndarray, amounts: np.ndarray) -> None:
    _fill_card_tags(ids, amounts, card.enchantments, card.affliction)


def _powers(creature: Creature, ids: np.ndarray, amounts: np.ndarray) -> None:
    for i, power in enumerate(list(creature.powers.values())[:MAX_POWERS]):
        ids[i] = _POWER_INDEX.get(power.power_id, 0)
        amounts[i] = np.clip(power.amount / 10.0, -10.0, 10.0)


def _fill_choice(obs: dict[str, np.ndarray], pending, combat: CombatState | None) -> None:
    """Pending card choice (combat or run): options are scored at actions 1..N."""
    g = obs["global"]
    g[11] = 1.0
    g[12] = float(pending.can_confirm())
    g[13] = float(pending.is_multi)
    g[14] = len(pending.selected_indices) / 5.0
    g[15] = pending.max_choices / 5.0
    obs["pending_choice"][0] = True
    for i, option in enumerate(pending.options[:MAX_CHOICES]):
        obs["choice_ids"][i] = _card_index(option.card.card_id)
        obs["choice_kinds"][i] = _PILES.get(option.source_pile, 0)
        _card_features(combat, option.card, obs["choice_features"][i], in_hand=False)
        _card_tags(option.card, obs["choice_tag_ids"][i], obs["choice_tag_amounts"][i])
        # Reuse the "playable" slot to show which options are already selected.
        obs["choice_features"][i][5] = float(i in pending.selected_indices)
        obs["choice_mask"][i] = True


def encode_entities(combat: CombatState) -> dict[str, np.ndarray]:
    """Encode the current fight as padded entity lists."""
    obs = empty_observation()
    player = combat.player

    g = obs["global"]
    g[0] = player.current_hp / max(player.max_hp, 1)
    g[1] = player.max_hp / 100.0
    g[2] = player.block / 50.0
    g[3] = combat.energy / 10.0
    g[4] = combat.max_energy / 10.0
    g[5] = combat.stars / 10.0
    g[6] = combat.turn_count / 20.0
    g[7] = len(combat.hand) / MAX_HAND_SIZE
    g[8] = len(combat.draw_pile) / 20.0
    g[9] = len(combat.discard_pile) / 20.0
    g[10] = len(combat.exhaust_pile) / 20.0
    _powers(player, obs["player_power_ids"], obs["player_power_amounts"])

    for i, card in enumerate(combat.hand[:MAX_HAND_SIZE]):
        obs["hand_ids"][i] = _card_index(card.card_id)
        _card_features(combat, card, obs["hand_features"][i], in_hand=True)
        _card_tags(card, obs["hand_tag_ids"][i], obs["hand_tag_amounts"][i])
        obs["hand_mask"][i] = True
        if card.base_damage is not None:
            for j, enemy in enumerate(combat.enemies[:MAX_ENEMIES]):
                if enemy.is_alive:
                    obs["hand_target_damage"][i, j] = preview_card_damage(combat, card, enemy) / 50.0

    _fill_piles(obs, combat, (("draw", combat.draw_pile), ("discard", combat.discard_pile),
                              ("exhaust", combat.exhaust_pile)))

    for i, enemy in enumerate(combat.enemies[:MAX_ENEMIES]):
        # Dead enemies stay as tokens so enemy slot j keeps matching target j.
        obs["enemy_ids"][i] = hash_bucket(enemy.monster_id)
        obs["enemy_mask"][i] = True
        f = obs["enemy_features"][i]
        f[0] = float(enemy.is_alive)
        f[1] = enemy.current_hp / max(enemy.max_hp, 1)
        f[2] = enemy.current_hp / 100.0
        f[3] = enemy.max_hp / 100.0
        f[4] = enemy.block / 50.0
        ai = combat.enemy_ais.get(enemy.combat_id)
        if ai is not None and enemy.is_alive and ai.current_move.intents:
            intents = ai.current_move.intents
            attacks = [it for it in intents if it.is_attack]
            if attacks:
                # Per-hit damage as the game shows it (enemy Strength/Weak, our Vulnerable...).
                hits = [(preview_intent_damage(combat, enemy, it.damage), it.hits) for it in attacks]
                f[5] = hits[0][0] / 30.0
                f[6] = hits[0][1] / 5.0
                f[7] = sum(damage * count for damage, count in hits) / 60.0
            f[8] = len(intents) / 3.0
            for intent in intents:
                _one_hot(f, 9, _INTENT_TYPES, intent.intent_type)
        _powers(enemy, obs["enemy_power_ids"][i], obs["enemy_power_amounts"][i])

    _fill_potions(obs, combat.potions)
    _fill_relics(obs, [relic.relic_id for relic in combat.relics])

    if combat.pending_choice is not None:
        _fill_choice(obs, combat.pending_choice, combat)
    return obs


def _fill_piles(obs, combat, piles) -> None:
    slot = 0
    for kind, pile in piles:
        for card in pile:
            if slot >= MAX_PILE_CARDS:
                return
            obs["pile_ids"][slot] = _card_index(card.card_id)
            obs["pile_kinds"][slot] = _PILES[kind]
            _card_features(combat, card, obs["pile_features"][slot], in_hand=False)
            _card_tags(card, obs["pile_tag_ids"][slot], obs["pile_tag_amounts"][slot])
            obs["pile_mask"][slot] = True
            slot += 1


def _fill_potions(obs, potions) -> None:
    for i, potion in enumerate(potions[:MAX_POTION_SLOTS]):
        if potion is None:
            continue
        obs["potion_ids"][i] = hash_bucket(potion.potion_id)
        obs["potion_mask"][i] = True
        obs["potion_features"][i][0] = 1.0
        _one_hot(obs["potion_features"][i], 1, _POTION_TARGETS, potion.target_type)


def _fill_relics(obs, relic_ids) -> None:
    relic_ids = list(relic_ids)
    if len(relic_ids) > MAX_RELICS:
        warnings.warn(f"Only the first {MAX_RELICS} relics are encoded; raise MAX_RELICS",
                      RuntimeWarning, stacklevel=2)
    for i, relic_id in enumerate(relic_ids[:MAX_RELICS]):
        obs["relic_ids"][i] = _relic_index(relic_id)
        obs["relic_mask"][i] = True


# ---------------------------------------------------------------------------
# Whole-run observations
# ---------------------------------------------------------------------------

class RouteStats(NamedTuple):
    low: np.ndarray        # fewest of each path type on any route
    high: np.ndarray       # most of each path type on any route
    profiles: np.ndarray   # one real route per _ROUTE_KEYS entry, all its counts together
    distance: np.ndarray   # floors to the nearest _DISTANCE_TYPES node (0 = this node, inf = none)


def _combine_routes(stats: list[RouteStats]) -> RouteStats:
    """Routes that may start from any of several nodes."""
    return RouteStats(
        np.min([s.low for s in stats], axis=0),
        np.max([s.high for s in stats], axis=0),
        np.array([max((s.profiles[k] for s in stats), key=key) for k, key in enumerate(_ROUTE_KEYS)]),
        np.min([s.distance for s in stats], axis=0),
    )


def route_stats(graph: dict) -> dict:
    """coord -> RouteStats over every route from that node to the boss.

    ``graph`` maps coord -> (point type, child coords); the simulator's map and
    the bridge's map_nodes both go through here so they encode the same.
    """
    memo: dict = {}
    visiting: set = set()

    def visit(coord) -> RouteStats:
        if coord in memo:
            return memo[coord]
        if coord in visiting:
            raise ValueError("Map contains a cycle")
        visiting.add(coord)
        point_type, children = graph[coord]
        own = np.array([float(point_type == t) for t in _PATH_TYPES])
        here = np.array([0.0 if point_type == t else np.inf for t in _DISTANCE_TYPES])
        results = [visit(child) for child in children if child in graph]
        if results:
            # Adding this node's counts keeps each criterion's best child best.
            ahead = _combine_routes(results)
            memo[coord] = RouteStats(own + ahead.low, own + ahead.high, own + ahead.profiles,
                                     np.minimum(here, ahead.distance + 1))
        else:
            memo[coord] = RouteStats(own, own, np.tile(own, (len(_ROUTE_KEYS), 1)), here)
        visiting.remove(coord)
        return memo[coord]

    for coord in graph:
        visit(coord)
    return memo


def route_features(stats: dict, coords) -> np.ndarray:
    """Routes onward from any of ``coords`` (all zeros when none are on the map)."""
    out = np.zeros(ROUTE_FEATURES, np.float32)
    found = [stats[c] for c in coords if c in stats]
    if found:
        s = _combine_routes(found)
        n, profiles = len(_PATH_TYPES), len(_ROUTE_KEYS) * len(_PATH_TYPES)
        out[:n] = s.low / 10.0
        out[n:2 * n] = s.high / 10.0
        out[2 * n:2 * n + profiles] = s.profiles.ravel() / 10.0
        out[2 * n + profiles:] = 1.0 / (1.0 + s.distance)  # 1 = this floor, 0 = none ahead
    return out


def _map_graph(act_map) -> dict:
    return {(p.coord.col, p.coord.row): (p.point_type, [(c.coord.col, c.coord.row) for c in p.children if c is not None])
            for p in act_map.all_points()}


_ROUTE_CACHE: dict = {}


def _cached_routes(act_map) -> dict:
    """Route stats for a map, recomputed only when the map changes (they are read every step)."""
    if act_map is None:
        return {}
    graph = _map_graph(act_map)
    key = tuple((coord, point_type, tuple(children)) for coord, (point_type, children) in graph.items())
    if key not in _ROUTE_CACHE:
        if len(_ROUTE_CACHE) >= 32:  # several envs share this process
            _ROUTE_CACHE.pop(next(iter(_ROUTE_CACHE)))
        _ROUTE_CACHE[key] = route_stats(graph)
    return _ROUTE_CACHE[key]


def run_options(env: STS2RunEnv, mask: np.ndarray) -> list[tuple[int, str, dict]]:
    """(action index, option kind, action dict) for every legal non-combat choice.

    Mirrors how ``STS2RunEnv.step`` interprets each action index.  Pending card
    choices (deck picks) are encoded as choice items instead, so they are not
    listed here.
    """
    mgr = env._mgr
    if mgr is None or mgr.is_over:
        return []
    L = _LAYOUT
    phase = mgr.phase
    actions = mgr.get_available_actions()
    if phase != RunManager.PHASE_COMBAT and any(a.get("action") in {"choose", "confirm_choice"} for a in actions):
        return []
    of = lambda name: [a for a in actions if a.get("action") == name]  # noqa: E731
    options: list[tuple[int, str, dict]] = []
    if phase == RunManager.PHASE_COMBAT:
        options += [(L.player_select_start + i, "select_player", a)
                    for i, a in enumerate(of("select_player")[:L.player_select_size])]
    elif phase == RunManager.PHASE_MAP_CHOICE:
        options += [(L.map_start + i, "move", a) for i, a in enumerate(actions[:L.map_size])]
    elif phase == RunManager.PHASE_CARD_REWARD:
        start = L.card_reward_start
        if of("pick_potion"):
            options += [(start, "pick_potion", of("pick_potion")[0]), (start + 3, "skip_potion", of("skip_potion")[0])]
        elif of("pick_relic_reward"):
            skip = (of("skip_relic") or [{}])[0]
            options += [(start, "pick_relic_reward", of("pick_relic_reward")[0]), (start + 3, "skip_relic", skip)]
        else:
            options.append((start + 3, "skip_card", {}))
            for i, a in enumerate(of("pick_card")):
                if i < 3:
                    options.append((start + i, "pick_card", a))
                elif i - 3 < L.card_reward_extra_size:
                    options.append((L.card_reward_extra_start + i - 3, "pick_card", a))
            if of("reroll_card_reward"):
                options.append((L.card_reward_reroll, "reroll", of("reroll_card_reward")[0]))
    elif phase == RunManager.PHASE_BOSS_RELIC:
        options += [(L.boss_relic_start + i, "pick_boss_relic", a)
                    for i, a in enumerate(of("pick_relic")[:L.boss_relic_size])]
    elif phase == RunManager.PHASE_SHOP:
        options.append((L.shop_start, "leave_shop", {}))
        buyable = [a for a in actions if a.get("action") != "leave_shop"]
        options += [(L.shop_start + 1 + i, a.get("action"), a) for i, a in enumerate(buyable[:L.shop_size - 1])]
    elif phase == RunManager.PHASE_REST_SITE:
        options += [(L.rest_start + i, "rest_option", a) for i, a in enumerate(of("rest_option")[:L.rest_size])]
    elif phase == RunManager.PHASE_EVENT:
        options += [(L.event_start + i, "event_choice", a) for i, a in enumerate(of("event_choice")[:L.event_size])]
    elif phase == RunManager.PHASE_TREASURE:
        options.append((L.treasure_start, "collect", (of("collect") or [{}])[0]))
    return [(slot, kind, a) for slot, kind, a in options if mask[slot] and kind in _KIND_INDEX]


def _fill_options(obs, env: STS2RunEnv, mask: np.ndarray, routes: dict) -> None:
    mgr = env._mgr
    gold = mgr.run_state.player.gold
    event = getattr(mgr, "_event_model", None)
    event_id = getattr(event, "event_id", "") or (type(event).__name__ if event is not None else "")
    for i, (slot, kind, action) in enumerate(run_options(env, mask)[:MAX_OPTIONS]):
        obs["option_slots"][i] = slot
        obs["option_kinds"][i] = _KIND_INDEX[kind]
        obs["option_mask"][i] = True
        if "card_id" in action:
            obs["option_cards"][i] = _card_index(action["card_id"])
        if "relic_id" in action:
            obs["option_relics"][i] = _relic_index(action["relic_id"])
        if "potion_id" in action:
            obs["option_potions"][i] = hash_bucket(action["potion_id"])
        f = obs["option_features"][i]
        price = action.get("price", 0)
        f[0] = price / 100.0
        f[1] = (gold - price) / 300.0 if price else 0.0
        f[2] = float(bool(action.get("on_sale")))
        f[3] = float(bool(action.get("upgraded")))
        f[4] = action.get("rerolls_remaining", 0) / 3.0
        f[5] = float(bool(action.get("enchantments")))
        _one_hot(f, 7, _RARITIES, action.get("rarity"))
        name = ""
        if kind == "move":
            name = f"point:{action.get('point_type')}"
            col, row = action["coord"]
            f[6] = row / 20.0
            f[ROUTE_START:] = route_features(routes, [(col, row)])
        elif kind == "rest_option":
            name = f"rest:{action.get('option_id')}"
        elif kind == "event_choice":
            name = f"event:{event_id}:{action.get('option_id')}"
        elif kind == "select_player":
            name = f"player:{action.get('player_id')}"
        obs["option_names"][i] = hash_bucket(name)


def encode_run_entities(env: STS2RunEnv, mask: np.ndarray) -> dict[str, np.ndarray]:
    """Encode the whole run: the fight (if any) plus deck, relics, run state and options."""
    mgr = env._mgr
    if mgr is None:
        return empty_observation()
    rs = mgr.run_state
    player = rs.player
    in_combat = mgr.phase == RunManager.PHASE_COMBAT and mgr.get_combat_state() is not None
    # Deck selections and event card selections both appear as choice items.
    deck_choice = None if in_combat else mgr.active_card_choice
    if in_combat:
        obs = encode_entities(mgr.get_combat_state())
    else:
        obs = empty_observation()
        obs["global"][0] = player.current_hp / max(player.max_hp, 1)
        obs["global"][1] = player.max_hp / 100.0
        _fill_piles(obs, None, (("deck", player.deck),))
        _fill_potions(obs, player.potions)
        _fill_relics(obs, rs.relics)
        if deck_choice is not None:
            _fill_choice(obs, deck_choice, None)

    r = obs["run"]
    r[0] = rs.current_act_index / 3.0
    r[1] = rs.total_floor / 50.0
    r[2] = rs.act_floor / 20.0
    r[3] = player.current_hp / max(player.max_hp, 1)
    r[4] = player.max_hp / 100.0
    r[5] = player.gold / 500.0
    r[6] = len(player.deck) / 40.0
    r[7] = len(rs.relics) / 30.0
    r[8] = sum(p is not None for p in player.potions) / max(player.max_potion_slots, 1)
    r[9] = player.max_potion_slots / 5.0
    r[10] = rs.ascension_level / 10.0
    r[11] = float(in_combat)
    _one_hot(r, 12, _ROOM_TYPES, mgr._current_room_type)
    phase_base = 12 + len(_ROOM_TYPES)
    if mgr.phase in _PHASE_INDEX:
        r[phase_base + _PHASE_INDEX[mgr.phase]] = 1.0
    r[phase_base + len(_PHASE_INDEX)] = float(deck_choice is not None)

    if rs.boss_encounter_id:
        obs["boss_id"][0] = hash_bucket(f"boss:{game_id(rs.boss_encounter_id)}")
    routes = _cached_routes(rs.map)
    obs["route_ahead"][:] = route_features(routes, [(c.col, c.row) for c in rs.get_available_next_coords()])
    _fill_options(obs, env, mask, routes)
    return obs


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

def _pending_choice(base) -> object | None:
    if isinstance(base, STS2RunEnv):
        mgr = base._mgr
        return None if mgr is None else mgr.active_card_choice
    combat = getattr(base, "combat", None)
    return combat.pending_choice if combat is not None else None


def cap_action_mask(pending, mask: np.ndarray) -> np.ndarray:
    """Hide pending-choice options beyond ``MAX_CHOICES``; the model cannot see them."""
    if pending is None:
        return mask
    capped = mask.copy()
    capped[1 + MAX_CHOICES:] = 0
    return capped if capped.any() else mask


class EntityObservationWrapper(gymnasium.Wrapper):
    """Replace a combat or run env's flat observation with entity lists."""

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        self.observation_space = observation_space()
        self.is_run = isinstance(env.unwrapped, STS2RunEnv)

    def _encode(self) -> dict[str, np.ndarray]:
        base = self.env.unwrapped
        if self.is_run:
            return encode_run_entities(base, self.action_masks())
        combat = getattr(base, "combat", None)
        return encode_entities(combat) if combat is not None else empty_observation()

    def reset(self, **kwargs):
        _, info = self.env.reset(**kwargs)
        return self._encode(), info

    def step(self, action):
        _, reward, terminated, truncated, info = self.env.step(action)
        return self._encode(), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return cap_action_mask(_pending_choice(self.env.unwrapped), self.env.action_masks())

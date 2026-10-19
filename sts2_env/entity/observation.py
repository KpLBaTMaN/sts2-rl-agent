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

import zlib

import gymnasium
import numpy as np
from gymnasium import spaces

from sts2_env.cards.base import CardInstance
from sts2_env.core.combat import CombatState
from sts2_env.core.constants import MAX_ENEMIES, MAX_HAND_SIZE, MAX_POTION_SLOTS
from sts2_env.core.creature import Creature
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
from sts2_env.map.map_point import MapCoord
from sts2_env.relics.base import RelicId
from sts2_env.run.run_manager import RunManager

MAX_PILE_CARDS = 60
MAX_POWERS = 12
MAX_RELICS = 24
MAX_CHOICES = 40
MAX_OPTIONS = 16
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
    "pile": len(_PILES) + 1,
    "option_kind": len(OPTION_KINDS) + 1,
}

CARD_FEATURES = 9 + len(_CARD_TYPES) + len(_TARGET_TYPES)
GLOBAL_FEATURES = 16
RUN_FEATURES = 12 + len(_ROOM_TYPES) + len(_PHASE_INDEX) + 1
ENEMY_FEATURES = 9 + len(_INTENT_TYPES)
POTION_FEATURES = 1 + len(_POTION_TARGETS)
OPTION_FEATURES = 7 + len(_RARITIES) + 2 * len(_PATH_TYPES)

# name -> (shape, dtype); masks are bool, IDs int64, features float32.
OBS_SPEC: dict[str, tuple[tuple[int, ...], type]] = {
    "global": ((GLOBAL_FEATURES,), np.float32),
    "run": ((RUN_FEATURES,), np.float32),
    "player_power_ids": ((MAX_POWERS,), np.int64),
    "player_power_amounts": ((MAX_POWERS,), np.float32),
    "hand_ids": ((MAX_HAND_SIZE,), np.int64),
    "hand_features": ((MAX_HAND_SIZE, CARD_FEATURES), np.float32),
    "hand_mask": ((MAX_HAND_SIZE,), np.bool_),
    "pile_ids": ((MAX_PILE_CARDS,), np.int64),
    "pile_kinds": ((MAX_PILE_CARDS,), np.int64),
    "pile_features": ((MAX_PILE_CARDS, CARD_FEATURES), np.float32),
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
    out[7] = float(card.is_ethereal or card.is_retain)
    out[8] = card.star_cost / 5.0
    _one_hot(out, 9, _CARD_TYPES, card.card_type)
    _one_hot(out, 9 + len(_CARD_TYPES), _TARGET_TYPES, card.target_type)


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
        obs["hand_mask"][i] = True

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
                f[5] = attacks[0].damage / 30.0
                f[6] = attacks[0].hits / 5.0
                f[7] = sum(it.total_damage for it in attacks) / 60.0
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
    for i, relic_id in enumerate(list(relic_ids)[:MAX_RELICS]):
        obs["relic_ids"][i] = _relic_index(relic_id)
        obs["relic_mask"][i] = True


# ---------------------------------------------------------------------------
# Whole-run observations
# ---------------------------------------------------------------------------

def _path_summaries(act_map) -> dict:
    """coord -> (min counts, max counts) of each path type from that node to the boss."""
    memo: dict = {}

    def visit(point):
        key = point.coord
        if key in memo:
            return memo[key]
        own = np.array([float(point.point_type == t) for t in _PATH_TYPES], dtype=np.float32)
        children = [child for child in point.children if child is not None]
        if not children:
            memo[key] = (own, own)
        else:
            results = [visit(child) for child in children]
            memo[key] = (own + np.min([r[0] for r in results], axis=0),
                         own + np.max([r[1] for r in results], axis=0))
        return memo[key]

    for point in act_map.all_points():
        visit(point)
    return memo


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


def _fill_options(obs, env: STS2RunEnv, mask: np.ndarray) -> None:
    mgr = env._mgr
    gold = mgr.run_state.player.gold
    paths = None
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
            act_map = mgr.run_state.map
            if act_map is not None:
                paths = paths if paths is not None else _path_summaries(act_map)
                coord = MapCoord(col, row)
                if coord in paths:
                    low, high = paths[coord]
                    base = 7 + len(_RARITIES)
                    f[base:base + len(_PATH_TYPES)] = low / 10.0
                    f[base + len(_PATH_TYPES):base + 2 * len(_PATH_TYPES)] = high / 10.0
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

    _fill_options(obs, env, mask)
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

"""Configurable reward profiles for training experiments.

A profile is a JSON file in ``local/config/rewards/`` holding term weights.
Training configs select one with ``"reward": "<name>"`` (or a .json path, or an
inline object) and may adjust single weights with ``"reward_overrides"``.

Shaping follows the OpenAI Five pattern: the objective result dominates, and
shaping terms are paid on *changes* in game state so nothing is counted twice.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from sts2_env.core.enums import RoomType

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REWARDS_DIR = PROJECT_ROOT / "local" / "config" / "rewards"

# Every term defaults to zero, so a profile only lists what it pays for.
WEIGHT_KEYS = ("win", "death", "timeout", "floor", "hp", "elite",
               "enemy_damage", "max_hp", "relic", "gold", "upgrade")
TERMS = WEIGHT_KEYS[3:]
TEXT_KEYS = ("profile", "description")
HP_CURVES = ("dota", "linear")

# Used when no profile file exists (for example a fresh checkout without local/).
BUILTIN_PROFILES = {
    "sparse": {"description": "Objective result only: +1 win, -1 death or timeout.",
               "win": 1.0, "death": -1.0, "timeout": -1.0},
    "shaped": {"description": "Default: objective result plus small progress, HP and combat shaping.",
               "win": 5.0, "death": -1.0, "timeout": -1.0, "floor": 0.1, "hp": 1.0,
               "hp_curve": "dota", "elite": 0.3, "enemy_damage": 0.05,
               "max_hp": 0.01, "relic": 0.05, "gold": 0.0},
}
BUILTIN_PROFILES["deckbuilder"] = {
    **BUILTIN_PROFILES["shaped"],
    "description": "Experimental permanent-upgrade shaping with reduced immediate HP weight.",
    "hp": 0.5, "upgrade": 0.1,
}


def default_profile_name(task: str) -> str:
    # Combat already learns from the sparse result; runs need progress signals.
    return "sparse" if task == "combat" else "shaped"


def _read_profile(source: str) -> dict:
    path = Path(source)
    if source.endswith(".json"):
        path = path if path.is_absolute() else PROJECT_ROOT / path
    else:
        if not source or any(c in source for c in "/\\."):
            raise ValueError(f"reward must be a profile name or a .json path, not {source!r}")
        path = REWARDS_DIR / f"{source}.json"
        if not path.exists() and source in BUILTIN_PROFILES:
            return dict(BUILTIN_PROFILES[source], profile=source)
    if not path.exists():
        raise ValueError(f"Reward profile not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"Reward profile {path} must be an object")
    return dict(data, profile=data.get("profile", path.stem))


def _validate(profile: dict, where: str) -> None:
    for key, value in profile.items():
        if key in TEXT_KEYS:
            if not isinstance(value, str):
                raise ValueError(f"{where}.{key} must be text")
        elif key == "hp_curve":
            if value not in HP_CURVES:
                raise ValueError(f"{where}.hp_curve must be one of {', '.join(HP_CURVES)}")
        elif key in WEIGHT_KEYS:
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError(f"{where}.{key} must be a finite number")
        else:
            raise ValueError(f"Unknown reward key: {where}.{key}")


def resolve_reward(config: dict) -> dict:
    """Return the full weight table for a config; already-resolved tables pass through."""
    source = config.get("reward")
    overrides = config.get("reward_overrides") or {}
    if source is None:
        source = default_profile_name(config["task"])
    if isinstance(source, str):
        profile = _read_profile(source)
    elif isinstance(source, dict):
        profile = dict(source)
        profile.setdefault("profile", "inline")
    else:
        raise ValueError("reward must be a profile name, a .json path, or an object")
    if not isinstance(overrides, dict):
        raise ValueError("reward_overrides must be an object")
    _validate(profile, "reward")
    _validate(overrides, "reward_overrides")
    resolved = {key: 0.0 for key in WEIGHT_KEYS}
    resolved["hp_curve"] = "dota"
    resolved.update(profile)
    resolved.update(overrides)
    return resolved


def hp_value(hp: int, max_hp: int, curve: str) -> float:
    """Value of an HP fraction; the Dota curve weights low-HP losses most."""
    x = min(max(hp / max_hp, 0.0), 1.0) if max_hp > 0 else 0.0
    if curve == "linear":
        return x
    return (x + 1.0 - (1.0 - x) ** 4) / 2.0


class RewardTracker:
    """Computes shaped step rewards from simulator state before and after each action."""

    def __init__(self, weights: dict, task: str):
        self.weights = weights
        self.task = task
        self.reset(None)

    def reset(self, env) -> None:
        self.totals = {term: 0.0 for term in ("result",) + TERMS}
        # Hold the objects themselves: id() values are reused once a combat is freed.
        self._tracked_combat = None
        self._enemy_total = 0.0
        self._known_enemies: dict[int, list] = {}  # id -> [creature, credited damage]
        self._credited_upgrades: dict[int, object] = {}
        self._before = self._snapshot(env) if env is not None else None

    # ---- state access -------------------------------------------------

    def _combat(self, env):
        if self.task == "combat":
            return env.combat
        return env._mgr.get_combat_state()

    def _snapshot(self, env) -> dict:
        combat = self._combat(env)
        if self.task == "combat":
            player, floor, relics, gold, room = combat.player, 0, 0, 0, None
            deck = ()
        else:
            mgr = env._mgr
            run_player = mgr.run_state.player
            player = combat.player if combat is not None else run_player
            floor, relics, gold = mgr.run_state.total_floor, len(run_player.relics), run_player.gold
            room = mgr._current_room_type
            # Read the permanent deck, never temporary cards in combat piles.
            deck = getattr(run_player, "deck", ()) if self.weights.get("upgrade", 0) else ()
        enemies = {}
        if combat is not None:
            enemies = {id(e): (e, max(e.current_hp, 0)) for e in combat.enemies}
        return dict(hp=player.current_hp, max_hp=player.max_hp, floor=floor, relics=relics, gold=gold,
                    room=room, combat=combat, combat_over=combat.is_over if combat is not None else True,
                    enemies=enemies, deck={id(c): (c, bool(c.upgraded)) for c in deck})

    # ---- reward -------------------------------------------------------

    def step(self, env, terminated: bool, truncated: bool, success: bool) -> tuple[float, dict]:
        w, before = self.weights, self._before
        after = self._snapshot(env)
        terms = {term: 0.0 for term in TERMS}

        terms["hp"] = w["hp"] * (hp_value(after["hp"], after["max_hp"], w["hp_curve"])
                                 - hp_value(before["hp"], before["max_hp"], w["hp_curve"]))
        terms["max_hp"] = w["max_hp"] * (after["max_hp"] - before["max_hp"])
        terms["floor"] = w["floor"] * max(after["floor"] - before["floor"], 0)
        terms["relic"] = w["relic"] * max(after["relics"] - before["relics"], 0)
        terms["gold"] = w["gold"] * max(after["gold"] - before["gold"], 0)
        if w.get("upgrade", 0):
            for key, (card, upgraded) in after["deck"].items():
                previous = before["deck"].get(key)
                if (previous is not None and previous[0] is card and not previous[1]
                        and upgraded and key not in self._credited_upgrades):
                    self._credited_upgrades[key] = card
                    terms["upgrade"] += w["upgrade"]

        combat = before["combat"]
        if combat is not None:
            # Measure the combat that was active before the action, even if it just ended.
            if combat is not self._tracked_combat:
                self._tracked_combat, self._known_enemies, self._enemy_total = combat, {}, 0.0
            damage = 0
            for enemy in combat.enemies:
                key = id(enemy)
                known = self._known_enemies.get(key)
                if known is None or known[0] is not enemy:
                    known = self._known_enemies[key] = [enemy, 0]
                    self._enemy_total += max(enemy.max_hp, 1)
                previous = before["enemies"].get(key)
                if previous is not None and previous[0] is enemy:
                    # Reviving enemies (Illusion) would otherwise pay out on every kill.
                    dealt = min(max(previous[1] - max(enemy.current_hp, 0), 0), enemy.max_hp - known[1])
                    known[1] += max(dealt, 0)
                    damage += max(dealt, 0)
            if self._enemy_total:
                terms["enemy_damage"] = w["enemy_damage"] * damage / self._enemy_total
            if (before["room"] == RoomType.ELITE and not before["combat_over"]
                    and combat.is_over and combat.player_won):
                terms["elite"] = w["elite"]

        result = 0.0
        if terminated:
            result = w["win"] if success else w["death"]
        elif truncated:
            result = w["timeout"]

        self._before = after
        self.totals["result"] += result
        for term, value in terms.items():
            self.totals[term] += value
        return result + sum(terms.values()), terms

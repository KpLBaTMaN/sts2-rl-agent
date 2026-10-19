"""Damage and block calculation pipelines.

Uses the centralized Hook dispatch system from hooks.py.
Faithfully reproduces Hook.ModifyDamageInternal (Hook.cs:1902) and
Hook.ModifyBlock (Hook.cs:960).
"""

from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from sts2_env.core.enums import ValueProp

if TYPE_CHECKING:
    from sts2_env.core.creature import Creature
    from sts2_env.core.combat import CombatState


@dataclass
class DamageResult:
    """Result of applying damage to a single target."""

    target: Creature
    blocked: int = 0
    hp_lost: int = 0
    was_killed: bool = False
    unblocked_damage: int = 0
    overkill_damage: int = 0
    was_block_broken: bool = False
    was_fully_blocked: bool = False

    @property
    def total_damage(self) -> int:
        return self.blocked + self.unblocked_damage


def calculate_damage(
    base_damage: int,
    dealer: Creature | None,
    target: Creature,
    props: ValueProp,
    combat_or_creatures: CombatState | list[Creature],
) -> int:
    """Calculate final damage after all power modifiers.

    Accepts either a CombatState (uses hooks.modify_damage) or a list of
    Creature (legacy path, iterates powers directly).
    """
    from sts2_env.core.hooks import modify_damage

    # If given a CombatState, use the hook system
    if hasattr(combat_or_creatures, 'all_creatures'):
        combat = combat_or_creatures
        if dealer is not None:
            combat._ensure_pending_attack_context(dealer, target, props)
        return modify_damage(
            base_damage,
            dealer,
            target,
            props,
            combat,
            card_source=getattr(combat, "active_card_source", None),
        )

    # Legacy path: iterate creature list directly
    all_creatures: list[Creature] = combat_or_creatures
    damage = float(base_damage)

    for creature in all_creatures:
        for power in creature.powers.values():
            damage += power.modify_damage_additive(creature, dealer, target, props)

    for creature in all_creatures:
        for power in creature.powers.values():
            damage *= power.modify_damage_multiplicative(creature, dealer, target, props)

    return max(0, math.floor(damage))


def calculate_block(
    base_block: int,
    target: Creature,
    props: ValueProp,
    combat_or_creatures: CombatState | list[Creature],
    card_source: object | None = None,
    card_play: object | None = None,
) -> int:
    """Calculate final block after all power modifiers."""
    from sts2_env.core.hooks import modify_block

    if hasattr(combat_or_creatures, 'all_creatures'):
        if card_source is not None and card_play is None:
            card_play = getattr(combat_or_creatures, "active_card_play_token", None)
        return modify_block(base_block, target, props, combat_or_creatures,
                            card_source=card_source, card_play=card_play)

    all_creatures: list[Creature] = combat_or_creatures
    block = float(base_block)

    for creature in all_creatures:
        for power in creature.powers.values():
            block += power.modify_block_additive(creature, target, props,
                                                  card_source, card_play)

    for creature in all_creatures:
        for power in creature.powers.values():
            block *= power.modify_block_multiplicative(creature, target, props,
                                                        card_source, card_play)

    return max(0, math.floor(block))


# ─── Previews (numbers shown to the player; no side effects) ───────────────

@contextmanager
def _previewing_card(combat: CombatState, card: object | None):
    # Powers read the card being played from the combat rather than a parameter.
    previous = combat._active_card_source
    combat._active_card_source = card
    try:
        yield
    finally:
        combat._active_card_source = previous


def preview_card_damage(combat: CombatState, card: object, target: Creature | None = None) -> int:
    """Damage the card shows, like DamageVar.UpdateCardPreview.

    With no target this is the number printed on the card (Strength, Weak...);
    with a target it also counts that creature's modifiers (Vulnerable...).
    """
    from sts2_env.core.hooks import modify_damage

    if card.base_damage is None:
        return 0
    with _previewing_card(combat, card):
        return modify_damage(card.base_damage, combat.player, target, ValueProp.MOVE, combat, card_source=card)


def preview_card_block(combat: CombatState, card: object) -> int:
    """Block the card shows, like BlockVar.UpdateCardPreview (Dexterity, Frail...)."""
    from sts2_env.core.hooks import modify_block

    if card.base_block is None:
        return 0
    with _previewing_card(combat, card):
        return modify_block(card.base_block, combat.player, ValueProp.MOVE, combat,
                            card_source=card, preview=True)


def preview_intent_damage(combat: CombatState, enemy: Creature, damage: int) -> int:
    """Per-hit damage an attack intent shows, like AttackIntent.GetSingleDamage."""
    from sts2_env.core.hooks import modify_damage

    with _previewing_card(combat, None):
        return modify_damage(damage, enemy, combat.player, ValueProp.MOVE, combat)


def apply_damage(
    target: Creature,
    damage: int,
    props: ValueProp,
    combat: CombatState | None = None,
    dealer: Creature | None = None,
) -> DamageResult:
    """Apply calculated damage to a creature.

    When combat is provided, fires damage hooks and applies HP loss modification.
    Returns DamageResult with details.
    """
    from sts2_env.core.hooks import (
        fire_before_damage_received, fire_after_damage_received,
        fire_after_current_hp_changed, fire_after_damage_given,
        modify_hp_lost_after_osty, modify_hp_lost_before_osty,
        modify_unblocked_damage_target,
    )

    attack = None
    if combat is not None:
        attack = combat.active_attack or combat.pending_auto_attack
        can_hit = getattr(combat, "can_hit_creature", None)
        if callable(can_hit) and not can_hit(target):
            return DamageResult(target=target, blocked=0, hp_lost=0, was_killed=False, unblocked_damage=0)

    # Fire before-damage hooks (Thorns)
    if combat is not None:
        fire_before_damage_received(target, dealer, damage, props, combat)

    # Block absorption
    unblockable = bool(props & ValueProp.UNBLOCKABLE)
    blocked = target.damage_block(damage, unblockable)
    was_block_broken = target.block <= 0 and blocked > 0
    remaining = damage - blocked

    if combat is not None and remaining > 0:
        remaining = modify_hp_lost_before_osty(remaining, target, dealer, props, combat)

    damage_target = target
    if combat is not None and remaining > 0:
        damage_target = modify_unblocked_damage_target(target, remaining, props, dealer, combat)

    # HP loss modification after possible Osty redirection (Intangible, TungstenRod)
    if combat is not None and remaining > 0:
        remaining = modify_hp_lost_after_osty(remaining, damage_target, dealer, props, combat)
    was_fully_blocked = not unblockable and (blocked > 0 or target.block > 0) and remaining == 0

    # Apply HP loss
    was_alive = damage_target.is_alive
    hp_lost = damage_target.lose_hp(remaining, fire_hooks=combat is None)
    was_killed = was_alive and damage_target.is_dead
    overkill_damage = max(0, remaining - hp_lost)

    redirected_result: DamageResult | None = None
    if damage_target is not target:
        redirected_result = DamageResult(
            target=damage_target,
            blocked=0,
            hp_lost=hp_lost,
            was_killed=was_killed,
            unblocked_damage=hp_lost,
            overkill_damage=overkill_damage,
        )
        overkill = overkill_damage
        if combat is not None and overkill > 0:
            overkill = modify_hp_lost_after_osty(overkill, target, dealer, props, combat)
        was_alive = target.is_alive
        hp_lost = target.lose_hp(overkill, fire_hooks=False)
        was_killed = was_alive and target.is_dead
        remaining = overkill
        overkill_damage = max(0, remaining - hp_lost)

    result = DamageResult(
        target=target,
        blocked=blocked,
        hp_lost=hp_lost,
        was_killed=was_killed,
        unblocked_damage=hp_lost,
        overkill_damage=overkill_damage,
        was_block_broken=was_block_broken,
        was_fully_blocked=was_fully_blocked,
    )

    if attack is not None:
        if redirected_result is not None:
            attack.results.append(redirected_result)
        attack.results.append(result)

    # Fire after-damage hooks
    if combat is not None:
        damage_results = [result] if redirected_result is None else [redirected_result, result]
        killed_targets = []
        for damage_result in damage_results:
            combat.record_damage_event(dealer, damage_result.target, props, damage_result.unblocked_damage)
        if was_block_broken:
            for power in list(target.powers.values()):
                on_block_broken = getattr(power, "on_block_broken", None)
                if callable(on_block_broken):
                    on_block_broken(target, combat)
        for damage_result in damage_results:
            if damage_result.hp_lost > 0:
                fire_after_current_hp_changed(
                    damage_result.target,
                    -damage_result.hp_lost,
                    combat,
                )
        if dealer is not None:
            for damage_result in damage_results:
                sentinel = object()
                previous_result = getattr(combat, "_active_damage_result", sentinel)
                combat._active_damage_result = damage_result
                try:
                    fire_after_damage_given(
                        dealer,
                        damage_result.target,
                        damage_result.unblocked_damage,
                        props,
                        combat,
                    )
                finally:
                    if previous_result is sentinel:
                        delattr(combat, "_active_damage_result")
                    else:
                        combat._active_damage_result = previous_result
        for damage_result in damage_results:
            if damage_result.was_killed:
                killed_targets.append(damage_result.target)
            if not damage_result.was_killed or not damage_result.target.is_dead:
                sentinel = object()
                previous_result = getattr(combat, "_active_damage_result", sentinel)
                combat._active_damage_result = damage_result
                try:
                    fire_after_damage_received(
                        damage_result.target,
                        dealer,
                        damage_result.unblocked_damage,
                        props,
                        combat,
                    )
                finally:
                    if previous_result is sentinel:
                        delattr(combat, "_active_damage_result")
                    else:
                        combat._active_damage_result = previous_result
        for killed_target in killed_targets:
            combat.kill_creature(killed_target)

    return result

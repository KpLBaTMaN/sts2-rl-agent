"""Live objects must use the training encoder's scales, IDs and action slots."""
import numpy as np
import pytest
import torch

from sts2_env.core.damage import preview_card_block, preview_card_damage, preview_intent_damage
from sts2_env.core.enums import PowerId
from sts2_env.entity import observation as E
from sts2_env.entity.model import SpireFormer, SpireFormerConfig
from sts2_env.entity.ppo import EntityAgent
from sts2_env.gym_env.run_env import STS2RunEnv
from sts2_env.gym_env.combat_env import STS2CombatEnv
from sts2_env.watch.entity_adapter import ENTITY_SCHEMA, EntityWatchAdapter, enum_name
from sts2_env.watch.player import model_decision


def card(c, playable=False, pile="deck", combat=None):
    data = dict(id=c.card_id.name, cost=c.cost, cost_x=c.has_energy_cost_x,
                base_damage=c.base_damage or 0, base_block=c.base_block or 0,
                upgraded=c.upgraded, playable=playable, star_cost=c.star_cost,
                type=c.card_type.name, target=c.target_type.name, rarity=c.rarity.name,
                source_pile=pile, keywords=[k.capitalize() for k in sorted(c.keywords)],
                retain_this_turn=c.is_retain or c.single_turn_retain, sly_this_turn=c.is_sly)
    # The bridge sends the game's IDs for a card's one enchantment and affliction.
    if c.enchantments:
        name, amount = next(iter(c.enchantments.items()))
        data["enchantment"] = dict(id=enum_name(name), amount=amount)
    if c.affliction:
        data["affliction"] = enum_name(c.affliction)
    # In combat the bridge also sends the numbers the card shows (RlCombatHandler.SerializeCard).
    if combat is not None:
        if c.base_damage is not None:
            data["damage"] = preview_card_damage(combat, c)
            if pile == "hand":
                data["target_damage"] = [preview_card_damage(combat, c, e) if e.is_alive else 0
                                         for e in combat.enemies]
        if c.base_block is not None:
            data["block"] = preview_card_block(combat, c)
    return data


def live_map(env):
    mgr, rs = env._mgr, env._mgr.run_state
    p = rs.player
    # As WatchSession.Enrich: children of the current point, or of the opening point before the first move.
    here = rs.map.get_point(rs.visited_map_coords[-1]) if rs.visited_map_coords else rs.map.start_point
    return dict(type="map_select", entity_schema=ENTITY_SCHEMA,
                next_map_coords=[dict(col=c.coord.col, row=c.coord.row) for c in here.children],
                deck=[card(c) for c in p.deck], relics=list(rs.relics), potions=[],
                nodes=[dict(index=i, type=a["point_type"], col=a["coord"][0], row=a["coord"][1])
                       for i, a in enumerate(mgr.get_available_actions())],
                map_nodes=[dict(col=n.coord.col, row=n.coord.row, type=n.point_type.name,
                           children=[dict(col=c.coord.col, row=c.coord.row) for c in n.children if c])
                           for n in rs.map.all_points()],
                run_state=dict(act_index=rs.current_act_index, floor=rs.total_floor, act_floor=rs.act_floor,
                     hp=p.current_hp, max_hp=p.max_hp, gold=p.gold, deck_size=len(p.deck),
                     relic_count=len(rs.relics), potion_count=0, max_potion_slots=p.max_potion_slots,
                     ascension=rs.ascension_level, room="Map", in_combat=False,
                     boss=rs.boss_encounter_id))


def test_map_entities_match_training_encoder():
    env = STS2RunEnv(act_count=1)
    env.reset(options={"run_seed": 42})
    env._mgr.run_state.player.deck[0].add_enchantment("Swift", 2)
    live = live_map(env)
    expected = E.encode_run_entities(env, env.action_masks())
    actual = EntityWatchAdapter().encode(live)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], err_msg=key)
    assert expected["pile_tag_ids"].any()
    env.close()


def test_live_rest_and_smith_option_names_match_training():
    env = STS2RunEnv(act_count=1)
    try:
        env.reset(options={"run_seed": 42})
        # Stand on a first-row node, so the route ahead is that node's children.
        rs = env._mgr.run_state
        rs.add_visited_coord(rs.get_available_next_coords()[0])
        live = live_map(env)
        env._mgr._enter_rest_site()
        actions = env._mgr.get_available_actions()
        live.update(type="rest_site", options=[dict(index=i, id=a["option_id"],
                    action="rest_option", enabled=True) for i, a in enumerate(actions)])
        live["run_state"].update(room="RestSite", in_combat=False)
        expected = E.encode_run_entities(env, env.action_masks())
        actual = EntityWatchAdapter().encode(live)
        for key in ("option_names", "option_kinds", "option_slots", "option_mask", "route_ahead", "boss_id"):
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
        assert expected["route_ahead"].any() and expected["boss_id"][0]
    finally:
        env.close()


def test_combat_entities_match_training_encoder():
    env = STS2CombatEnv()
    env.reset(seed=42)
    c = env.combat
    # Modifiers make the shown numbers differ from the base ones.
    c.player.apply_power(PowerId.STRENGTH, 3)
    c.player.apply_power(PowerId.WEAK, 1)
    c.player.apply_power(PowerId.DEXTERITY, 2)
    c.enemies[0].apply_power(PowerId.VULNERABLE, 2)
    c.enemies[0].apply_power(PowerId.STRENGTH, 4)
    # Keywords, enchantments and an affliction must arrive the same way too.
    c.hand[0].add_enchantment("Sharp", 3)
    c.hand[1].afflict("hexed")
    c.hand[2].keywords = c.hand[2].keywords | {"sly", "innate"}
    c.draw_pile[0].add_enchantment("RoyallyApproved")
    def powers(creature):
        return [dict(id=p.power_id.name, amount=p.amount) for p in creature.powers.values()]
    enemies = []
    for enemy in c.enemies:
        ai = c.enemy_ais.get(enemy.combat_id)
        intents = ai.current_move.intents if ai else []
        enemies.append(dict(id=enemy.monster_id, hp=enemy.current_hp, max_hp=enemy.max_hp,
                            block=enemy.block, is_alive=enemy.is_alive, powers=powers(enemy),
                            intents=[dict(type=i.intent_type.name, hits=i.hits,
                                          damage=preview_intent_damage(c, enemy, i.damage) if i.is_attack else i.damage)
                                     for i in intents]))
    live = dict(type="combat_action", entity_schema=ENTITY_SCHEMA, deck=[],
                relics=[r.relic_id.name for r in c.relics], potions=[],
                run_state=dict(hp=c.player.current_hp, max_hp=c.player.max_hp, in_combat=True),
                player=dict(block=c.player.block, energy=c.energy, max_energy=c.max_energy,
                            stars=c.stars, powers=powers(c.player)), round=c.turn_count,
                hand=[card(v, c.can_play_card(v), "hand", c) for v in c.hand], enemies=enemies,
                draw_pile_count=len(c.draw_pile), discard_pile_count=len(c.discard_pile), exhaust_pile_count=len(c.exhaust_pile),
                **{name + "_pile": [card(v, pile=name, combat=c) for v in pile] for name, pile in
                   [("draw", c.draw_pile), ("discard", c.discard_pile), ("exhaust", c.exhaust_pile)]})
    expected = E.encode_entities(c)
    actual = EntityWatchAdapter().encode(live)
    for key in expected:
        if key != "run":
            np.testing.assert_allclose(actual[key], expected[key], err_msg=key)
    assert expected["hand_target_damage"].any()
    assert expected["hand_tag_ids"][:2, 0].all() and expected["pile_tag_ids"][0, 0]
    assert expected["hand_features"][2][[12, 13]].all()
    env.close()


def test_entity_decision_is_legal_and_reports_actual_model_probabilities():
    env = STS2RunEnv(act_count=1)
    env.reset(options={"run_seed": 42})
    live = live_map(env)
    agent = EntityAgent(SpireFormer(SpireFormerConfig(d_model=32, layers=1, heads=4, n_actions=157)).eval(), torch.device("cpu"))
    adapter = EntityWatchAdapter()
    wire, proposal = model_decision(agent, adapter, live)
    assert proposal["action"] in adapter.choices(live)[0]
    assert wire["action"] == "choose"
    assert "SpireFormer" in proposal["source"]
    assert sum(p["probability"] for p in proposal["probabilities"]) == pytest.approx(1)
    with pytest.raises(ValueError, match="updated"):
        adapter.encode(dict(live, entity_schema=0))
    env.close()

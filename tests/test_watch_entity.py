"""Live objects must use the training encoder's scales, IDs and action slots."""
import numpy as np
import pytest
import torch

from sts2_env.entity import observation as E
from sts2_env.entity.model import SpireFormer, SpireFormerConfig
from sts2_env.entity.ppo import EntityAgent
from sts2_env.gym_env.run_env import STS2RunEnv
from sts2_env.gym_env.combat_env import STS2CombatEnv
from sts2_env.watch.entity_adapter import EntityWatchAdapter
from sts2_env.watch.player import model_decision


def card(c, playable=False, pile="deck"):
    return dict(id=c.card_id.name, cost=c.cost, cost_x=c.has_energy_cost_x,
                base_damage=c.base_damage or 0, base_block=c.base_block or 0,
                upgraded=c.upgraded, playable=playable, star_cost=c.star_cost,
                type=c.card_type.name, target=c.target_type.name, rarity=c.rarity.name,
                source_pile=pile, keywords=[name for name, enabled in
                [("Exhaust", c.exhausts), ("Ethereal", c.is_ethereal), ("Retain", c.is_retain)] if enabled])


def live_map(env):
    mgr, rs = env._mgr, env._mgr.run_state
    p = rs.player
    return dict(type="map_select", entity_schema=1,
                deck=[card(c) for c in p.deck], relics=list(rs.relics), potions=[],
                nodes=[dict(index=i, type=a["point_type"], col=a["coord"][0], row=a["coord"][1])
                       for i, a in enumerate(mgr.get_available_actions())],
                map_nodes=[dict(col=n.coord.col, row=n.coord.row, type=n.point_type.name,
                           children=[dict(col=c.coord.col, row=c.coord.row) for c in n.children if c])
                           for n in rs.map.all_points()],
                run_state=dict(act_index=rs.current_act_index, floor=rs.total_floor, act_floor=rs.act_floor,
                     hp=p.current_hp, max_hp=p.max_hp, gold=p.gold, deck_size=len(p.deck),
                     relic_count=len(rs.relics), potion_count=0, max_potion_slots=p.max_potion_slots,
                     ascension=rs.ascension_level, room="Map", in_combat=False))


def test_map_entities_match_training_encoder():
    env = STS2RunEnv(act_count=1)
    env.reset(options={"run_seed": 42})
    live = live_map(env)
    expected = E.encode_run_entities(env, env.action_masks())
    actual = EntityWatchAdapter().encode(live)
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], err_msg=key)
    env.close()


def test_live_rest_and_smith_option_names_match_training():
    env = STS2RunEnv(act_count=1)
    try:
        env.reset(options={"run_seed": 42})
        live = live_map(env)
        env._mgr._enter_rest_site()
        actions = env._mgr.get_available_actions()
        live.update(type="rest_site", options=[dict(index=i, id=a["option_id"],
                    action="rest_option", enabled=True) for i, a in enumerate(actions)])
        live["run_state"].update(room="RestSite", in_combat=False)
        expected = E.encode_run_entities(env, env.action_masks())
        actual = EntityWatchAdapter().encode(live)
        for key in ("option_names", "option_kinds", "option_slots", "option_mask"):
            np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)
    finally:
        env.close()


def test_combat_entities_match_training_encoder():
    env = STS2CombatEnv()
    env.reset(seed=42)
    c = env.combat
    def powers(creature):
        return [dict(id=p.power_id.name, amount=p.amount) for p in creature.powers.values()]
    enemies = []
    for enemy in c.enemies:
        ai = c.enemy_ais.get(enemy.combat_id)
        intents = ai.current_move.intents if ai else []
        enemies.append(dict(id=enemy.monster_id, hp=enemy.current_hp, max_hp=enemy.max_hp,
                            block=enemy.block, is_alive=enemy.is_alive, powers=powers(enemy),
                            intents=[dict(type=i.intent_type.name, damage=i.damage, hits=i.hits) for i in intents]))
    live = dict(type="combat_action", entity_schema=1, deck=[],
                relics=[r.relic_id.name for r in c.relics], potions=[],
                run_state=dict(hp=c.player.current_hp, max_hp=c.player.max_hp, in_combat=True),
                player=dict(block=c.player.block, energy=c.energy, max_energy=c.max_energy,
                            stars=c.stars, powers=powers(c.player)), round=c.turn_count,
                hand=[card(v, c.can_play_card(v), "hand") for v in c.hand], enemies=enemies,
                draw_pile_count=len(c.draw_pile), discard_pile_count=len(c.discard_pile), exhaust_pile_count=len(c.exhaust_pile),
                **{name + "_pile": [card(v, pile=name) for v in pile] for name, pile in
                   [("draw", c.draw_pile), ("discard", c.discard_pile), ("exhaust", c.exhaust_pile)]})
    expected = E.encode_entities(c)
    actual = EntityWatchAdapter().encode(live)
    for key in expected:
        if key != "run":
            np.testing.assert_allclose(actual[key], expected[key], err_msg=key)
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

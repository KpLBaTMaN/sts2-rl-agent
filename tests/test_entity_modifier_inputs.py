"""SpireFormer sees each power's own stack size and the numbers the game shows."""
import numpy as np
import torch

import sts2_env.powers  # noqa: F401

from sts2_env.cards.ironclad import create_ironclad_starter_deck
from sts2_env.cards.ironclad_basic import make_defend_ironclad, make_strike_ironclad
from sts2_env.core.constants import MAX_HAND_SIZE
from sts2_env.core.combat import CombatState
from sts2_env.core.enums import PowerId
from sts2_env.core.rng import Rng
from sts2_env.entity import observation as E
from sts2_env.entity.model import SpireFormer, SpireFormerConfig
from sts2_env.gym_env import observation as flat
from sts2_env.monsters.act1_weak import create_shrinker_beetle, create_twig_slime_s
from sts2_env.relics.base import RelicId


def make_combat(relics=(), enemies=1):
    combat = CombatState(player_hp=80, player_max_hp=80, deck=create_ironclad_starter_deck(),
                         rng_seed=7, character_id="Ironclad", relics=list(relics))
    for i in range(enemies):
        creature, ai = (create_shrinker_beetle if i == 0 else create_twig_slime_s)(Rng(7 + i))
        combat.add_enemy(creature, ai)
    combat.start_combat()
    return combat


def set_hand(combat, cards):
    for card in cards:
        card.owner = combat.player
    combat.hand = cards


def pooled(model, powers):
    ids = torch.zeros(1, E.MAX_POWERS, dtype=torch.long)
    amounts = torch.zeros(1, E.MAX_POWERS)
    for i, (power, amount) in enumerate(powers):
        ids[0, i], amounts[0, i] = E._POWER_INDEX[power], amount / 10
    with torch.no_grad():
        return model._pooled_powers(ids, amounts)[0]


def test_pooled_powers_keep_each_stack_size():
    torch.manual_seed(0)
    model = SpireFormer(SpireFormerConfig(d_model=32, layers=1, heads=4)).eval()
    S, V, W = PowerId.STRENGTH, PowerId.VULNERABLE, PowerId.WEAK
    # Same powers and the same total amount, split differently.
    assert not torch.allclose(pooled(model, [(S, 3), (V, 2)]), pooled(model, [(S, 2), (V, 3)]), atol=1e-4)
    assert not torch.allclose(pooled(model, [(S, 50), (W, 1)]), pooled(model, [(S, 1), (W, 50)]), atol=1e-4)
    assert not torch.allclose(pooled(model, [(S, 1)]), pooled(model, [(S, 50)]), atol=1e-4)


def test_cards_show_strength_weak_dexterity_frail_and_target_vulnerable():
    combat = make_combat(enemies=2)
    set_hand(combat, [make_strike_ironclad(), make_defend_ironclad()])
    combat.apply_power_to(combat.player, PowerId.STRENGTH, 2)
    combat.apply_power_to(combat.player, PowerId.DEXTERITY, 2)
    combat.apply_power_to(combat.player, PowerId.FRAIL, 1)
    combat.apply_power_to(combat.enemies[0], PowerId.VULNERABLE, 2)
    combat.enemies[1].current_hp = 0

    obs = E.encode_entities(combat)
    strike, defend = obs["hand_features"][0], obs["hand_features"][1]
    assert strike[2] * 50 == np.float32(6) and defend[3] * 50 == np.float32(5)  # base numbers stay
    assert round(strike[9] * 50) == 8                     # 6 + 2 Strength
    assert round(defend[10] * 50) == 5                    # floor((5 + 2 Dexterity) * 0.75 Frail)
    assert round(obs["hand_target_damage"][0, 0] * 50) == 12  # floor(8 * 1.5 Vulnerable)
    assert obs["hand_target_damage"][0, 1] == 0           # dead enemy
    assert not obs["hand_target_damage"][1].any()         # Defend deals no damage

    combat.apply_power_to(combat.player, PowerId.WEAK, 1)
    obs = E.encode_entities(combat)
    assert round(obs["hand_features"][0][9] * 50) == 6    # floor(8 * 0.75 Weak)
    assert round(obs["hand_target_damage"][0, 0] * 50) == 9  # floor(8 * 0.75 * 1.5)


def test_intent_damage_counts_enemy_strength_and_our_vulnerable():
    combat = make_combat()
    enemy = combat.enemies[0]
    ai = combat.enemy_ais[enemy.combat_id]
    ai._current_state_id = "CHOMP_MOVE"
    base = ai.current_move.intents[0].damage
    combat.apply_power_to(enemy, PowerId.STRENGTH, 3)
    combat.apply_power_to(combat.player, PowerId.VULNERABLE, 2)
    shown = int((base + 3) * 1.5)

    assert round(E.encode_entities(combat)["enemy_features"][0][5] * 30) == shown
    vector = flat.encode_observation(combat)
    enemy_start = 4 + flat.NUM_PLAYER_POWERS + MAX_HAND_SIZE * flat.CARD_FEATURES + flat.PILE_FEATURES
    assert round(vector[enemy_start + 3 + flat.NUM_INTENT_TYPES] * 30) == shown


def test_card_keywords_each_have_their_own_flag():
    card = make_strike_ironclad()
    out = np.zeros(E.CARD_FEATURES, np.float32)
    flags = {"ethereal": 7, "retain": 11, "innate": 12, "sly": 13, "unplayable": 14, "eternal": 15}
    for keyword, index in flags.items():
        card.keywords = frozenset({keyword})
        out[:] = 0
        E._card_features(None, card, out, in_hand=False)
        assert [i for i in flags.values() if out[i]] == [index], keyword
    # Retain and Sly granted for this turn only count too.
    card.keywords = frozenset()
    card.single_turn_retain = True
    card.combat_vars["sly_this_turn"] = True
    out[:] = 0
    E._card_features(None, card, out, in_hand=False)
    assert out[11] == 1 and out[13] == 1


def test_enchantments_and_afflictions_use_game_ids_and_keep_amounts():
    combat = make_combat()
    strike, defend = make_strike_ironclad(), make_defend_ironclad()
    strike.add_enchantment("Sharp", 3)
    defend.afflict("hexed")
    set_hand(combat, [strike, defend])
    combat.draw_pile[0].add_enchantment("RoyallyApproved")

    obs = E.encode_entities(combat)
    assert obs["hand_tag_ids"][0, 0] == E.hash_bucket("enchantment:SHARP")
    assert obs["hand_tag_amounts"][0, 0] == np.float32(0.3)
    assert obs["hand_tag_ids"][1, 0] == E.hash_bucket("affliction:HEXED")
    assert obs["hand_tag_amounts"][1, 0] == 0
    assert not obs["hand_tag_ids"][:, 1:].any()
    assert obs["pile_tag_ids"][0, 0] == E.hash_bucket("enchantment:ROYALLY_APPROVED")
    assert obs["pile_features"][0][11] == 1 and obs["pile_features"][0][12] == 1  # it grants Retain and Innate


def test_card_tags_keep_each_name_and_amount():
    torch.manual_seed(0)
    model = SpireFormer(SpireFormerConfig(d_model=32, layers=1, heads=4)).eval()

    def tags(entries):
        obs = {"hand_tag_ids": torch.zeros(1, 1, E.MAX_CARD_TAGS, dtype=torch.long),
               "hand_tag_amounts": torch.zeros(1, 1, E.MAX_CARD_TAGS)}
        for i, (name, amount) in enumerate(entries):
            obs["hand_tag_ids"][0, 0, i], obs["hand_tag_amounts"][0, 0, i] = E.hash_bucket(name), amount / 10
        with torch.no_grad():
            return model._card_tags(obs, "hand")[0, 0]

    sharp, swift = "enchantment:SHARP", "enchantment:SWIFT"
    assert not tags([]).any()
    assert not torch.allclose(tags([(sharp, 1)]), tags([(sharp, 3)]), atol=1e-4)
    assert not torch.allclose(tags([(sharp, 3), (swift, 1)]), tags([(sharp, 1), (swift, 3)]), atol=1e-4)


def test_previews_change_nothing():
    """Vambrace doubles the first card block; previewing must not use up that trigger."""
    combat = make_combat(relics=["Vambrace"])
    set_hand(combat, [make_defend_ironclad(), make_defend_ironclad(), make_strike_ironclad()])
    combat.energy = 3
    vambrace = next(r for r in combat.relics if r.relic_id == RelicId.VAMBRACE)

    for _ in range(2):
        obs = E.encode_entities(combat)
        assert [round(v * 50) for v in obs["hand_features"][:2, 10]] == [10, 10]
    assert vambrace._triggering_card is None
    assert combat.active_card_source is None

    combat.player.block = 0
    assert combat.play_card(0)
    assert combat.player.block == 10
    assert combat.play_card(0)
    assert combat.player.block == 15

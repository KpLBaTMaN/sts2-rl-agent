"""Combat end once only secondary enemies (minions) remain.

C# refs:
- CombatManager.IsEnding: combat ends when no living enemy IsPrimaryEnemy
  (and no ShouldStopCombatFromEnding hook objects).
- CreatureCmd.KillWithoutCheckingWinCondition: when a primary enemy dies and
  every living teammate IsSecondaryEnemy, those teammates are killed through
  CreatureCmd.Kill, so their death hooks fire.
- MinionPower.OwnerIsSecondaryEnemy => true; IllusionPower applies MinionPower
  and does not override ShouldStopCombatFromEnding.
"""

from __future__ import annotations

from sts2_env.cards.ironclad_basic import create_ironclad_starter_deck
from sts2_env.core.combat import CombatState
from sts2_env.core.enums import PowerId
from sts2_env.core.rng import Rng
from sts2_env.encounters.act1 import setup_the_kin_boss
from sts2_env.monsters.act1 import create_eye_with_teeth, create_fogmog


def _make_combat(rng_seed: int = 42) -> CombatState:
    deck = create_ironclad_starter_deck()
    return CombatState(player_hp=80, player_max_hp=80, deck=deck, rng_seed=rng_seed)


def _fogmog_with_eye(rng_seed: int = 42):
    combat = _make_combat(rng_seed)
    rng = Rng(rng_seed)
    fogmog, fogmog_ai = create_fogmog(rng)
    combat.add_enemy(fogmog, fogmog_ai)
    eye, eye_ai = create_eye_with_teeth(rng)
    combat.add_enemy(eye, eye_ai)
    combat.start_combat()
    assert not combat.is_over
    return combat, fogmog, eye


class TestSecondaryEnemyClassification:
    def test_eye_with_teeth_is_secondary_and_fogmog_is_primary(self):
        _, fogmog, eye = _fogmog_with_eye()

        assert eye.has_power(PowerId.ILLUSION)
        assert eye.has_power(PowerId.MINION)
        assert eye.is_secondary_enemy
        assert not eye.is_primary_enemy
        assert fogmog.is_primary_enemy
        assert not fogmog.is_secondary_enemy

    def test_player_is_neither_primary_nor_secondary_enemy(self):
        combat, _, _ = _fogmog_with_eye()

        assert not combat.player.is_primary_enemy
        assert not combat.player.is_secondary_enemy


class TestFogmogCombatEnd:
    def test_killing_fogmog_with_eye_alive_wins_and_kills_eye(self):
        combat, fogmog, eye = _fogmog_with_eye()

        combat.kill_creature(fogmog)

        assert combat.is_over
        assert combat.player_won
        assert eye.is_dead
        # The eye died through the normal pipeline: IllusionPower.AfterDeath ran.
        assert eye.powers[PowerId.ILLUSION].is_reviving

    def test_damaging_fogmog_to_death_with_eye_alive_wins(self):
        combat, fogmog, eye = _fogmog_with_eye()

        combat.deal_damage(dealer=combat.player, target=fogmog, amount=fogmog.current_hp + 50)

        assert fogmog.is_dead
        assert eye.is_dead
        assert combat.is_over
        assert combat.player_won

    def test_killing_eye_alone_does_not_end_combat(self):
        combat, fogmog, eye = _fogmog_with_eye()

        combat.kill_creature(eye)

        assert eye.is_dead
        assert eye.powers[PowerId.ILLUSION].is_reviving
        assert fogmog.is_alive
        assert not combat.is_over

    def test_killing_fogmog_while_eye_is_reviving_wins(self):
        combat, fogmog, eye = _fogmog_with_eye()
        combat.kill_creature(eye)
        assert eye.powers[PowerId.ILLUSION].is_reviving

        combat.kill_creature(fogmog)

        assert combat.is_over
        assert combat.player_won

    def test_eye_revives_while_fogmog_alive(self):
        combat, fogmog, eye = _fogmog_with_eye()
        combat.kill_creature(eye)

        combat.end_player_turn()

        assert not combat.is_over
        assert fogmog.is_alive
        assert eye.is_alive
        assert eye.current_hp == eye.max_hp
        assert not eye.powers[PowerId.ILLUSION].is_reviving


class TestKinBossCombatEnd:
    def test_killing_kin_priest_kills_followers_and_wins(self):
        combat = _make_combat()
        setup_the_kin_boss(combat, Rng(42))
        combat.start_combat()
        followers = [enemy for enemy in combat.enemies if enemy.monster_id == "KIN_FOLLOWER"]
        priest = next(enemy for enemy in combat.enemies if enemy.monster_id == "KIN_PRIEST")
        assert len(followers) == 2
        assert all(follower.is_secondary_enemy for follower in followers)

        combat.kill_creature(priest)

        assert all(follower.is_dead for follower in followers)
        assert combat.is_over
        assert combat.player_won

    def test_killing_kin_followers_does_not_end_combat(self):
        combat = _make_combat()
        setup_the_kin_boss(combat, Rng(42))
        combat.start_combat()
        priest = next(enemy for enemy in combat.enemies if enemy.monster_id == "KIN_PRIEST")

        for follower in [enemy for enemy in combat.enemies if enemy.monster_id == "KIN_FOLLOWER"]:
            combat.kill_creature(follower)

        assert priest.is_alive
        assert not combat.is_over


class TestDecimillipedeCombatEnd:
    """C# ReattachPower has no ShouldStopCombatFromEnding override, so dead
    segments still marked as reviving do not keep combat going."""

    def test_killing_every_segment_wins(self):
        from sts2_env.encounters.act2 import setup_decimillipede_elite

        combat = _make_combat()
        setup_decimillipede_elite(combat, Rng(42))
        combat.start_combat()
        segments = list(combat.enemies)
        assert len(segments) == 3

        for segment in segments[:-1]:
            combat.kill_creature(segment)
            assert segment.powers[PowerId.REATTACH].is_reviving
            assert not combat.is_over

        combat.kill_creature(segments[-1])

        assert combat.is_over
        assert combat.player_won

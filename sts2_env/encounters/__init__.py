"""Encounter pools per act."""

from __future__ import annotations


def encounter_pools(act_index: int) -> dict[str, list]:
    """Return {weak, normal, elite, boss} encounter setup lists for an act."""
    if act_index == 0:
        from sts2_env.encounters.act1 import (
            WEAK_ENCOUNTERS, NORMAL_ENCOUNTERS, ELITE_ENCOUNTERS, BOSS_ENCOUNTERS,
        )
    elif act_index == 1:
        from sts2_env.encounters.act2 import (
            WEAK_ENCOUNTERS, NORMAL_ENCOUNTERS, ELITE_ENCOUNTERS, BOSS_ENCOUNTERS,
        )
    elif act_index == 2:
        from sts2_env.encounters.act3 import (
            WEAK_ENCOUNTERS, NORMAL_ENCOUNTERS, ELITE_ENCOUNTERS, BOSS_ENCOUNTERS,
        )
    else:
        from sts2_env.encounters.act4 import (
            WEAK_ENCOUNTERS, NORMAL_ENCOUNTERS, ELITE_ENCOUNTERS, BOSS_ENCOUNTERS,
        )
    return {
        "weak": list(WEAK_ENCOUNTERS),
        "normal": list(NORMAL_ENCOUNTERS),
        "elite": list(ELITE_ENCOUNTERS),
        "boss": list(BOSS_ENCOUNTERS),
    }


def encounter_id(setup) -> str:
    """The game's encounter ID for a setup function: setup_the_kin_boss -> THE_KIN_BOSS."""
    return setup.__name__.removeprefix("setup_").upper()

"""SpireFormer knows the act's boss and what lies ahead on the map in every phase."""
import functools

import numpy as np

from sts2_env.core.enums import MapPointType, RoomType
from sts2_env.encounters import encounter_id, encounter_pools
from sts2_env.entity import observation as E
from sts2_env.entity.observation import EntityObservationWrapper
from sts2_env.gym_env.run_env import STS2RunEnv
from sts2_env.run import run_manager
from sts2_env.run.run_manager import RunManager

M, EL, R, S, T, U = (MapPointType.MONSTER, MapPointType.ELITE, MapPointType.REST_SITE,
                     MapPointType.SHOP, MapPointType.TREASURE, MapPointType.UNKNOWN)
N = len(E._PATH_TYPES)


def boss_ids(act_index):
    return {encounter_id(setup) for setup in encounter_pools(act_index)["boss"]}


def test_boss_is_picked_when_the_act_starts_and_is_the_one_fought(monkeypatch):
    env = STS2RunEnv(act_count=1)
    env.reset(options={"run_seed": 7})
    boss = env._mgr.run_state.boss_encounter_id
    assert boss in boss_ids(0)

    other = STS2RunEnv(act_count=1)
    other.reset(options={"run_seed": 7})
    assert other._mgr.run_state.boss_encounter_id == boss
    seen = set()
    for seed in range(20):
        other.reset(options={"run_seed": seed})
        seen.add(other._mgr.run_state.boss_encounter_id)
    assert len(seen) > 1

    fought = []
    real = run_manager.encounter_pools

    def recording(act_index):
        pools = real(act_index)

        def wrap(setup):
            @functools.wraps(setup)
            def run(combat, rng):
                fought.append(encounter_id(setup))
                return setup(combat, rng)
            return run

        pools["boss"] = [wrap(setup) for setup in pools["boss"]]
        return pools

    monkeypatch.setattr(run_manager, "encounter_pools", recording)
    env._mgr._enter_combat(RoomType.BOSS)
    assert fought == [boss]


def test_each_act_has_its_boss_from_the_act_start():
    env = STS2RunEnv()
    env.reset(options={"run_seed": 3})
    rs = env._mgr.run_state
    for act in range(1, len(rs.acts)):
        rs.enter_act(act)
        assert rs.boss_encounter_id in boss_ids(act)


TWO_ROUTES = {
    "start": (MapPointType.ANCIENT, ["a1", "b1"]),
    # Route a: two elites, then two rest sites.  Route b: two fights and a shop.
    "a1": (EL, ["a2"]), "a2": (EL, ["a3"]), "a3": (R, ["a4"]), "a4": (R, ["boss"]),
    "b1": (M, ["b2"]), "b2": (M, ["b3"]), "b3": (S, ["boss"]),
    "boss": (MapPointType.BOSS, []),
}


def test_kept_routes_show_which_counts_go_together():
    f = E.route_features(E.route_stats(TWO_ROUTES), ["start"])
    # Fewest and most of each type alone suggest a route with no elites and two rest sites.
    np.testing.assert_allclose(f[:N] * 10, [0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(f[N:2 * N] * 10, [2, 2, 2, 1, 0, 0])
    profiles = f[2 * N:6 * N].reshape(4, N) * 10
    np.testing.assert_allclose(profiles[0], [2, 0, 0, 1, 0, 0])  # fewest elites: no rest site either
    np.testing.assert_allclose(profiles[1], [0, 2, 2, 0, 0, 0])  # most elites
    np.testing.assert_allclose(profiles[2], [0, 2, 2, 0, 0, 0])  # most rest sites come with both elites
    np.testing.assert_allclose(profiles[3], [2, 0, 0, 1, 0, 0])  # most shops
    # Nearest elite 1 floor on, rest site and shop 3, no treasure: 1 / (1 + floors).
    np.testing.assert_allclose(f[6 * N:], [1 / 2, 1 / 4, 1 / 4, 0])


def test_route_features_do_not_depend_on_child_order():
    tied = dict(TWO_ROUTES, start=(MapPointType.ANCIENT, ["a1", "b1", "d1"]),
                # Ties route b on elites, rest sites and fights; differs in the rest.
                d1=(M, ["d2"]), d2=(M, ["d3"]), d3=(T, ["boss"]))
    flipped = dict(tied, start=(MapPointType.ANCIENT, ["d1", "b1", "a1"]))
    np.testing.assert_array_equal(E.route_features(E.route_stats(tied), ["start"]),
                                  E.route_features(E.route_stats(flipped), ["start"]))


def test_route_ahead_follows_the_player_outside_node_selection():
    env = EntityObservationWrapper(STS2RunEnv(raise_simulation_errors=True))
    obs, _ = env.reset(options={"run_seed": 1})
    assert obs["route_ahead"].any()
    obs, *_ = env.step(int(np.flatnonzero(env.action_masks())[0]))  # first map node
    mgr = env.unwrapped._mgr
    rs = mgr.run_state
    assert mgr.phase != RunManager.PHASE_MAP_CHOICE
    here = rs.map.get_point(rs.visited_map_coords[-1])
    expected = E.route_features(E.route_stats(E._map_graph(rs.map)),
                                [(c.coord.col, c.coord.row) for c in here.children])
    np.testing.assert_allclose(obs["route_ahead"], expected)
    assert obs["route_ahead"].any()
    assert obs["boss_id"][0] == E.hash_bucket(f"boss:{rs.boss_encounter_id}")

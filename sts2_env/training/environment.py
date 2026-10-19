"""Episode seed selection and objective statistics, shared by train and eval."""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_baselines3.common.monitor import Monitor

from sts2_env.core.enums import CardType, RoomType
from sts2_env.core.rng import INT_MAX_EXCLUSIVE
from sts2_env.gym_env.combat_env import STS2CombatEnv
from sts2_env.gym_env.run_env import STS2RunEnv
from sts2_env.training.rewards import RewardTracker, resolve_reward


ROOM_COUNTS = {"elites": RoomType.ELITE, "monsters": RoomType.MONSTER, "shops": RoomType.SHOP,
               "rest_sites": RoomType.REST_SITE, "events": RoomType.EVENT, "treasures": RoomType.TREASURE}


def run_snapshot(run_state):
    """End-of-run state for reports; deck and relic lists are ';'-joined so they fit one CSV cell."""
    player = run_state.player
    deck = player.deck
    rooms = [entry.room_type for entry in run_state.map_point_history]
    snapshot = dict(act=run_state.current_act_index + 1, gold=player.gold, max_hp=player.max_hp,
                    deck_size=len(deck), upgraded_cards=sum(card.upgraded for card in deck),
                    attacks=sum(card.card_type == CardType.ATTACK for card in deck),
                    skills=sum(card.card_type == CardType.SKILL for card in deck),
                    powers=sum(card.card_type == CardType.POWER for card in deck),
                    curses_statuses=sum(card.card_type in (CardType.CURSE, CardType.STATUS) for card in deck),
                    relic_count=len(player.relics), potions_held=len(player.held_potions()),
                    card_removals=player.card_shop_removals_used)
    snapshot.update({name: rooms.count(room) for name, room in ROOM_COUNTS.items()})
    snapshot.update(relics=";".join(player.relics),
                    deck=";".join(card.card_id.name + ("+" if card.upgraded else "") for card in deck))
    return snapshot


class ScenarioEnv(gym.Wrapper):
    def __init__(self, config, worker=0, evaluation=False):
        self.config = config
        self.evaluation = evaluation
        self.sequence = np.random.default_rng(config["seeds"]["schedule_seed"] + worker + (1000000 if evaluation else 0))
        self.episode_index = 0
        self.episode_seed = None
        self.steps = 0
        self.boss_reached = False
        train = config["training"]
        if config["task"] == "combat":
            env = STS2CombatEnv(max_turns=train["max_combat_turns"])
        else:
            env = STS2RunEnv(
                character_id=config["character"], ascension_level=config["ascension"],
                max_steps=train["max_episode_steps"], max_combat_turns=train["max_combat_turns"],
                act_count=1 if config["task"] == "act1" else None,
                start_with_neow=config["start_with_neow"],
                raise_simulation_errors=True,
            )
        super().__init__(env)
        self.rewards = RewardTracker(resolve_reward(config), config["task"])

    def _next_seed(self):
        seed_cfg = self.config["seeds"]
        eval_cfg = self.config["evaluation"]
        if self.evaluation and eval_cfg["mode"] != "training_seeds":
            pool = eval_cfg["seeds"]
            seed = pool[self.episode_index % len(pool)]
        elif seed_cfg["mode"] == "fixed":
            seed = seed_cfg["values"][0]
        elif seed_cfg["mode"] == "pool":
            pool = seed_cfg["values"]
            seed = pool[self.episode_index % len(pool)]
        else:
            excluded = (set(eval_cfg["seeds"]) | set(eval_cfg.get("final_seeds") or [])) if eval_cfg["mode"] == "unseen" else set()
            seed = int(self.sequence.integers(0, INT_MAX_EXCLUSIVE))
            while seed in excluded:
                seed = int(self.sequence.integers(0, INT_MAX_EXCLUSIVE))
        self.episode_index += 1
        return seed

    def reset(self, *, seed=None, options=None):
        # SB3's policy seed must not silently replace the configured game seed.
        self.episode_seed = self._next_seed()
        self.steps = 0
        self.boss_reached = False
        self.finished = False
        if self.config["task"] == "combat":
            result = self.env.reset(seed=self.episode_seed, options=options)
        else:
            options = dict(options or {}, run_seed=self.episode_seed)
            result = self.env.reset(seed=seed, options=options)
        self.rewards.reset(self.env)
        return result

    def action_masks(self):
        return self.env.action_masks()

    def step(self, action):
        if self.finished:
            raise RuntimeError("Episode finished; call reset before stepping again")
        if not self.action_masks()[int(action)]:
            raise ValueError(f"Illegal action: {action}")
        self.steps += 1
        is_combat = self.config["task"] == "combat"
        if not is_combat:
            mgr = self.env._mgr
            self.boss_reached |= mgr._current_room_type == RoomType.BOSS
        obs, _, terminated, truncated, info = self.env.step(action)
        if self.steps >= self.config["training"]["max_episode_steps"] and not terminated:
            truncated = True
        if is_combat:
            combat = self.env.combat
            success = bool(terminated and combat.player_won)
            hp = combat.player.current_hp
            floor = 0
            death_room = "combat" if terminated and not success else None
        else:
            mgr = self.env._mgr
            success = bool(terminated and mgr.player_won)
            hp = mgr.run_state.player.current_hp
            floor = mgr.run_state.total_floor
            death_room = mgr._current_room_type.name if terminated and not success and mgr._current_room_type else None
        # The configured reward profile replaces the simulator's sparse reward.
        reward, _ = self.rewards.step(self.env, terminated, truncated, success)
        self.finished = bool(terminated or truncated)
        if self.finished:
            info["reward_terms"] = dict(self.rewards.totals)
            if not is_combat:
                info["run_stats"] = run_snapshot(self.env._mgr.run_state)
        info.update(run_seed=self.episode_seed, is_success=success,
                    boss_reached=self.boss_reached, remaining_hp=hp,
                    floor=floor, decisions=self.steps, death_room=death_room,
                    timed_out=bool(truncated))
        return obs, reward, terminated, truncated, info


def make_env(config, worker=0, evaluation=False):
    env = ScenarioEnv(config, worker, evaluation)
    if config["model"]["type"] == "entity":
        from sts2_env.entity.observation import EntityObservationWrapper
        env = EntityObservationWrapper(env)
    return Monitor(env, info_keywords=("is_success",))

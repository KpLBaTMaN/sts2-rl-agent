"""Compare combat policies fight-by-fight on the same unseen seeds.

Win rate alone saturates on the combat task: with the starter deck almost
every normal fight is won and almost every boss is lost.  This script also
reports HP kept after wins and how much enemy HP is left after losses, split
by encounter tier.

    python scripts/compare_combat_models.py \
        --model flat=output/experiments/flat-combat-.../final_model.zip \
        --model entity=output/experiments/entity-combat-.../final_model.pt
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from sts2_env.encounters.act1 import BOSS_ENCOUNTERS, ELITE_ENCOUNTERS, NORMAL_ENCOUNTERS, WEAK_ENCOUNTERS
from sts2_env.entity.observation import EntityObservationWrapper
from sts2_env.gym_env.combat_env import STS2CombatEnv

TIERS = {}
for tier, pool in (("weak", WEAK_ENCOUNTERS), ("normal", NORMAL_ENCOUNTERS),
                   ("elite", ELITE_ENCOUNTERS), ("boss", BOSS_ENCOUNTERS)):
    for setup in pool:
        TIERS[setup] = tier


class _Tracked:
    """Encounter setup that remembers which tier was picked (same pool order)."""

    def __init__(self, setup, record):
        self.setup, self.record = setup, record

    def __call__(self, combat, rng):
        self.record["tier"] = TIERS[self.setup]
        self.record["encounter"] = self.setup.__name__.removeprefix("setup_")
        return self.setup(combat, rng)


def load_policy(path: str, device: str):
    if path.endswith(".pt"):
        from sts2_env.entity.ppo import EntityAgent
        agent, _ = EntityAgent.load(path, device)
        return agent, True
    from sb3_contrib import MaskablePPO
    return MaskablePPO.load(path, device="cpu"), False


def play(policy, entity: bool, seed: int, max_turns: int) -> dict:
    record: dict = {}
    from sts2_env.encounters.act1 import ALL_ACT1_ENCOUNTERS
    base = STS2CombatEnv(encounter_pool=[_Tracked(s, record) for s in ALL_ACT1_ENCOUNTERS], max_turns=max_turns)
    env = EntityObservationWrapper(base) if entity else base
    obs, _ = env.reset(seed=seed)
    decisions = 0
    while True:
        mask = env.action_masks()
        action, _ = policy.predict(obs, action_masks=mask, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(int(action))
        decisions += 1
        if terminated or truncated:
            break
    combat = base.combat
    enemy_max = sum(e.max_hp for e in combat.enemies) or 1
    enemy_left = sum(max(e.current_hp, 0) for e in combat.enemies if e.is_alive)
    return dict(seed=seed, encounter=record["encounter"], tier=record["tier"],
                win=bool(combat.player_won), hp_left=combat.player.current_hp if combat.player_won else 0,
                enemy_hp_left_pct=round(100 * enemy_left / enemy_max, 1),
                turns=combat.turn_count, decisions=decisions, timed_out=bool(truncated))


def summarise(rows: list[dict]) -> dict:
    out = {}
    for tier in ("all", "weak", "normal", "elite", "boss"):
        group = [r for r in rows if tier == "all" or r["tier"] == tier]
        if not group:
            continue
        wins = [r for r in group if r["win"]]
        losses = [r for r in group if not r["win"]]
        out[tier] = dict(
            fights=len(group), win_rate=round(len(wins) / len(group), 3),
            hp_left_when_won=round(float(np.mean([r["hp_left"] for r in wins])), 1) if wins else None,
            enemy_hp_left_pct_when_lost=round(float(np.mean([r["enemy_hp_left_pct"] for r in losses])), 1) if losses else None,
            turns=round(float(np.mean([r["turns"] for r in group])), 1),
        )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", action="append", required=True, help="label=path (.zip flat SB3, .pt SpireFormer)")
    parser.add_argument("--seeds", type=int, nargs=2, default=(10000, 10100), metavar=("START", "STOP"))
    parser.add_argument("--max-turns", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None, help="Directory for comparison.json and per-fight CSVs")
    args = parser.parse_args()

    seeds = list(range(*args.seeds))
    results = {}
    for spec in args.model:
        label, _, path = spec.partition("=")
        policy, entity = load_policy(path, args.device)
        rows = [play(policy, entity, seed, args.max_turns) for seed in seeds]
        results[label] = dict(path=path, summary=summarise(rows), rows=rows)

    labels = list(results)
    print(f"{len(seeds)} unseen fights (seeds {args.seeds[0]}-{args.seeds[1] - 1}), deterministic play\n")
    print(f"{'tier':8s} {'fights':>6s}  " + "  ".join(f"{l + ' win%':>12s} {'HP kept':>8s} {'enemy HP%':>9s}" for l in labels))
    for tier in ("all", "weak", "normal", "elite", "boss"):
        cells = []
        for label in labels:
            s = results[label]["summary"].get(tier)
            if s is None:
                break
            hp = "-" if s["hp_left_when_won"] is None else f"{s['hp_left_when_won']:.1f}"
            enemy = "-" if s["enemy_hp_left_pct_when_lost"] is None else f"{s['enemy_hp_left_pct_when_lost']:.1f}"
            cells.append(f"{100 * s['win_rate']:11.1f}% {hp:>8s} {enemy:>9s}")
        else:
            fights = results[labels[0]]["summary"][tier]["fights"]
            print(f"{tier:8s} {fights:6d}  " + "  ".join(cells))
    print("\nHP kept = average HP left after a win. Enemy HP% = average enemy HP left after a loss (lower is closer).")

    if len(labels) == 2:
        a, b = (results[l]["rows"] for l in labels)
        both = [(x, y) for x, y in zip(a, b) if x["win"] and y["win"]]
        if both:
            diff = [y["hp_left"] - x["hp_left"] for x, y in both]
            print(f"On the {len(both)} fights both won, {labels[1]} kept {np.mean(diff):+.2f} HP vs {labels[0]} "
                  f"({sum(d > 0 for d in diff)} better, {sum(d < 0 for d in diff)} worse, {sum(d == 0 for d in diff)} equal).")

    if args.output:
        out = Path(args.output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "comparison.json").write_text(json.dumps(
            {l: dict(path=r["path"], summary=r["summary"]) for l, r in results.items()}, indent=2), encoding="utf-8")
        for label, r in results.items():
            with (out / f"{label}-fights.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(r["rows"][0]))
                writer.writeheader()
                writer.writerows(r["rows"])
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()

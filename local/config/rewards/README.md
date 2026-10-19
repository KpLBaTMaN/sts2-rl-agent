# Reward profiles

Select one in a training config with `"reward": "<file name without .json>"`.
Change single weights for one experiment with `"reward_overrides": {"floor": 0.2}`.
If `reward` is omitted: `combat` uses `sparse`, `act1`/`full_run` use `shaped`.

| Key | Paid when | Unit |
|---|---|---|
| `win` | Objective won (combat won / act boss beaten / run won) | once |
| `death` | Player died | once |
| `timeout` | Episode hit its step limit | once |
| `floor` | A new floor is reached | per floor |
| `hp` | HP changes (in and out of combat) | per change of HP value (0..1) |
| `hp_curve` | `dota`: `(x + 1 - (1-x)^4) / 2`, low-HP losses cost most; `linear`: HP fraction | — |
| `elite` | An elite combat is won | per elite |
| `enemy_damage` | Enemy HP removed, as a fraction of the combat's total enemy max HP | ~1 per combat |
| `max_hp` | Max HP changes (negative when lost) | per point |
| `relic` | A relic is gained | per relic |
| `gold` | Gold is gained | per gold |
| `upgrade` | An existing permanent deck card becomes upgraded | once per card instance per episode |

Keep total shaping over an episode below `win`, so climbing floors never pays more than winning.

## Deckbuilder experiment

`deckbuilder.json` keeps the shaped profile's weights except for `hp: 0.5`
(previously 1.0) and `upgrade: 0.10` (previously 0). The prepared training config
is `local/config/presets/entity-act1-deckbuilder.json`: a fresh 5-million-step
Ironclad Act 1 SpireFormer, random training seeds with schedule seed 123, model
seed 42, checkpoints every 100,000 steps, and the same validation/holdout seed
sets and PPO settings as the existing reward comparison. It has not been launched.

The live trace confirmed Rest at 80/80 HP with over 99.99% policy probability.
Rest/Smith option names and action slots match training. The old reward already
pays zero for a zero-HP heal; it has no upgrade term. The new profile adds a
small immediate signal for completing an upgrade, while keeping healing useful
when needed. For an 80-max-HP character healing 24 HP, the HP-only reward is:

| Choice | Immediate shaping |
|---|---:|
| Rest at 80 HP | 0 |
| Rest at 77 HP (gain 3) | +0.009375 |
| Rest at 40 HP (gain 24) | +0.090225 |
| Rest at 20 HP (gain 24) | +0.143850 |
| Complete a permanent card upgrade | +0.100000 |

These are immediate terms, not a hard HP threshold for the policy: future
survival and boss victory still affect the return. Clicking Smith without
upgrading pays nothing. Newly acquired upgraded cards, duplicate cards and
temporary combat upgrades do not earn the upgrade term. A card cannot repeatedly
earn it through downgrade/re-upgrade. There is no reward for adding any card or
shrinking the deck indiscriminately; those incentives can produce bloated or
nonfunctional decks. Which card to upgrade must still be learned from future
combat outcomes. Rest can also have relic benefits, so it is not forcibly masked
at full HP.

The research reference is OpenAI et al.,
[Dota 2 with Large Scale Deep Reinforcement Learning, Appendix G](https://arxiv.org/html/1912.06680#A7).
That paper uses outcome rewards plus progress shaping and the same quartic HP
curve already implemented here. Its weights are Dota-specific; +0.10 for an
upgrade and 0.5 for HP are proposed Slay the Spire experiment values, not numbers
from the paper. Item purchasing and ability ordering were partly scripted in
OpenAI Five, so the paper does not establish a recipe for learning deck building.

Compare against the current shaped model using win rate, boss reach and upgrades
per run, plus a campfire probe at 25/50/75/100% HP with an upgrade available. Track
Rest probability at full HP and upgrade probabilities/card choices. Use periodic
validation seeds for selection; use the reserved 200-seed holdout once at the
end. Keep gamma at 0.995 to isolate this reward change. A longer-horizon gamma
0.998 test can follow as a separate experiment; changing both at once would
confound the comparison. The old checkpoint's Smith probability is near zero,
so a fresh start avoids relying on it to re-explore that action.

Launch when ready with:

```powershell
.\.venv\Scripts\python.exe -m sts2_env.training.runner --config local/config/presets/entity-act1-deckbuilder.json
```

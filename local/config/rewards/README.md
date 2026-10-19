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

Keep total shaping over an episode below `win`, so climbing floors never pays more than winning.

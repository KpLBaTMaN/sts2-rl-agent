# Entity Policy (SpireFormer)

SpireFormer is the object-based policy recommended by the research plan, and
the default model (`model.type: "entity"`) for every training task: single
fights, Act 1 and full runs. The flat MLP (`model.type: "flat"`) is kept only as
a comparison baseline.

## What the AI sees

Everything is a list of items. Each item has an ID (looked up in a learned
embedding table) plus a few numbers.

| Items | In a fight | Outside a fight |
|---|---|---|
| Cards | Hand, draw, discard and exhaust piles | Whole deck |
| Enemies | HP, block, intent, damage, powers | - |
| Player | HP, block, energy, powers | HP |
| Potions, relics | Yes | Yes |
| Pending card choice | Cards offered by a card effect | Cards to remove, upgrade or transform |
| **Options** | Which player acts (co-op) | One item per legal choice: map node, offered card/relic/potion, shop item with price, rest or event option, skip, reroll |
| Run state | Act, floor, gold, room type, phase... | Same |

Map-node options also carry a **path summary**: the minimum and maximum number
of fights, elites, rest sites, shops, treasures and unknown rooms on the routes
from that node to the boss. This hand-made summary stands in for the map graph
network suggested by the report.

## How it chooses

A 3-layer Transformer lets every item look at every other item. Each action's
score comes from the item it refers to:

- play card *i* → card *i*'s token (and enemy *j*'s token if it is targeted)
- use potion *i* → potion *i*'s token (and the target enemy)
- pick option *k* → option *k*'s token
- end turn / confirm → the summary token

Scores are written into the environments' existing action indices (115 for
fights, 157 for runs), so the legality masks are unchanged.

## Files

- `sts2_env/entity/observation.py`: `encode_entities()` (fights),
  `encode_run_entities()` (runs), `run_options()`, `EntityObservationWrapper`
- `sts2_env/entity/model.py`: `SpireFormer`
- `sts2_env/entity/ppo.py`: masked PPO training loop, `EntityAgent` (predict/save/load)
- `scripts/compare_combat_models.py`: fight-by-fight comparison split by weak, normal, elite and boss
- `tests/test_entity_policy.py`

## Run it

```bash
.venv\Scripts\python.exe -m sts2_env.training.runner --config local/config/training.json
.venv\Scripts\python.exe -m sts2_env.training.runner --config local/config/presets/entity-act1.json
.venv\Scripts\python.exe -m sts2_env.training.runner --config local/config/presets/flat-act1.json
```

Or pass a preset path to `Train-Run.cmd`. Quick checks:
`smoke-entity.json` (fight) and `smoke-entity-act1.json` (run).

## Results so far

| Task | Flat MLP | SpireFormer |
|---|---|---|
| Single fights, starter deck, 100 unseen fights | 82% | 83% |

Both win every weak fight and about 93% of normal fights, and lose almost every
elite and boss, so this task cannot separate them.

## Current limits

- No memory (recurrent layer) yet.
- Up to 60 cards in piles or deck, 40 pending-choice options and 16 options are shown.
- Monster, potion and event-option IDs are strings hashed into 1,024 buckets.
- Card-bundle rewards and the card-reward "sacrifice" option are not reachable in the run environment (for either model).
- Models saved before the run update (`spireformer-v1`) must be retrained.
- The real-game watch tool and bridge still run flat SB3 models only.

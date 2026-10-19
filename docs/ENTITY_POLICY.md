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

**Powers** (Strength, Vulnerable, Weak...) are a list of (power ID, amount ÷ 10)
per creature. Each power has its own learned amount vector, scaled by its
amount, so a creature's summed powers keep every stack size separately.

**Numbers as the game shows them.** Cards carry their base damage/block and
the damage/block they show now (Strength, Weak, Dexterity, Frail... applied,
like the game's card preview). Each hand card also has its damage against each
enemy (adding that enemy's Vulnerable...), which feeds the targeting score.
Enemy attack intents show per-hit damage with the enemy's Strength/Weak and
the player's Vulnerable applied, like the game's intent icon.

**Card keywords, enchantments and afflictions.** Every card has its own flag
for Exhaust, Ethereal, Retain, Innate, Sly, Unplayable and Eternal (Retain and
Sly include this-turn-only grants). Its enchantment and affliction are hashed
by game ID (`enchantment:SHARP`, `affliction:HEXED`). Each name has a learned
vector plus its own amount vector scaled by the enchantment's amount, and these
are added to the card's token. This holds for hand, pile, deck and
pending-choice cards.

Map-node options also carry a **path summary** of the routes from that node to
the boss:
- the fewest and most fights, elites, rest sites, shops, treasures and unknown
  rooms on any route
- four real routes with all their counts together: fewest elites, most elites,
  most rest sites, most shops. Fewest and most alone can't say whether the
  elite-free route is also the one with two rest sites.
- how near the next elite, rest site, shop and treasure are: 1 / (1 + floors),
  0 if none is ahead

The same summary, over the nodes the player can move to next, is in
`route_ahead` in **every phase**, so rest, shop and reward choices can see
what is coming. The act's boss (`boss_id`, hashed encounter ID such as
`boss:VANTOM_BOSS`) is known from the act's start, as on the game's map. Both
are added to the global token. This hand-made summary stands in for the map
graph network suggested by the report.

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
- `tests/test_entity_policy.py`, `tests/test_entity_modifier_inputs.py`

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
- Up to 60 cards in piles or deck, 48 relics, 40 pending-choice options and 16 options are shown.
- An enchantment's disabled state is not shown, and offered shop/reward cards only say whether they are enchanted.
- Monster, potion and event-option IDs are strings hashed into 1,024 buckets.
- Card-bundle rewards and the card-reward "sacrifice" option are not reachable in the run environment (for either model).
- Models saved before the run update (`spireformer-v1`) or before per-power
  amounts, shown damage/block, card keywords/enchantments/afflictions, the
  act's boss and route summaries (`spireformer-v2`) must be retrained.
- The real-game watch tool now has a SpireFormer v3 adapter requiring the updated
  entity-enabled mod (`entity_schema: 3`). Encoder regression checks pass, and the Sparse checkpoint
  has executed live map and combat decisions. Transfer parity and live win rates
  still need broader evaluation. The resource pack contains only the manifest.

# Reward ablation after the choice-loop fix

Ready-to-run presets for the experiment plan in the 19 October 2026 training
audit and loop report (`docs/TRAINING_LOOP_REPORT_2026_10_19.md`). They need the
choice-loop fixes (add-only card selection, stall reporting) in the code that runs
them; on another machine, pull those changes first.

## What the data says

| Finding | Number | Consequence here |
|---|---|---|
| Choice loops wasted evaluations | 13.5–17% timeouts; 53–58% of evaluation decisions | Fixed in code. Old checkpoints were trained on the broken interface, so every run starts fresh (`resume_from: null`). |
| Fix checked on a saved model | Local shaped model, seeds 10000–10099: 9 timeouts → 0, wins 7 → 9 | Timeouts and stalls should now be about 0; anything else is a regression. |
| More steps did not help | Both selected checkpoints at 3.0M; shaped fell from 10/200 (3.0M) to 1/200 (5.0M) | Short budgets: 0.5M screen, 1M replication. Judge by the best checkpoint, never the final one. |
| Reward profile not settled | Shaped vs sparse wins −1.0 pt, 95% interval [−4.0, +2.0]; floor +1.98 | Keep sparse as a control arm in the screen. |
| Shaped reaches the boss, rarely beats it | Boss reach 32% vs 19.5%; conversion 15.6% vs 30.8% | Track `boss_reached_rate` and boss conversion separately. |
| Campfires: always Rest, almost never Smith | Smith 0.0013% at full HP (shaped); 0.67 upgrades per run (shaped), 0.14 (sparse) | Test the upgrade reward and the lower HP weight separately: the 2×2 below. |
| One model seed per arm | No between-run variance measured | Replicate with model seeds 42/43/44. |
| Old holdout inspected | 20000–20199 used for diagnosis | Now development data. Fresh final holdout: 30000–30999. |

PPO and model settings are the same as the 5M-step comparison (16 envs × 256
steps, batch 512, 4 epochs, lr 3e-4, entropy 0.02, gamma 0.995, SpireFormer
128/3/4, 2,000-decision limit, 200-turn combat cap, schedule seed 123), so only
the variable named in each file changes. Validation uses the same 50 seeds as
before (10000–10049), every 250K decisions, with checkpoints every 100K.

## Running the stages as a queue

Each `queue-*.json` file lists presets to train one after another. The queue
runs each preset as its own process, waits for its final and best evaluations,
then starts the next. It refuses a second launch on the same queue, and a crash
or reboot can be resumed without redoing finished runs.

| Plan | Runs | Rough time (RTX 4090) |
|---|---|---|
| `queue-stage1.json` | The six Stage 1 arms below | about 3.5 hours |
| `queue-stage2-baseline.json` | Shaped replicates s42/s43/s44, then the gamma 0.998 run | about 7 hours |

Neither needs a decision first. Only the Stage 2 candidate does: it is chosen
from Stage 1 results by the rules below, then queued separately.

On the server, from the repository root, with the project's Python environment
active (on Windows use `.venv\Scripts\python.exe` for `python`):

```bash
python -m sts2_env.training.comparison --prepare --root output/ablation/stage1 --plan local/config/presets/reward-ablation/queue-stage1.json
python -m sts2_env.training.comparison --prepare --root output/ablation/stage2-baseline --plan local/config/presets/reward-ablation/queue-stage2-baseline.json
nohup sh -c 'python -m sts2_env.training.comparison --root output/ablation/stage1 && python -m sts2_env.training.comparison --root output/ablation/stage2-baseline' > output/ablation/queue.log 2>&1 &
```

`--prepare` freezes each preset, with its exact reward weights, into
`<root>/configs/`; later edits to the preset files do not change a prepared
queue. Run directories go to `<root>/runs/`. Each run logs to `<root>/<name>.log`,
queue state is in `<root>/batch.json`, and `<root>/comparison.csv` adds a row as
each run finishes (best-checkpoint test results, upgrades and turns per combat).

- **Check the pipeline first** (seconds per run): prepare a separate root with
  `--smoke` added, and run it.
- **Pause after the current run:** write `{"pause_after_current": true}` to
  `<root>/queue-control.json`; set it back to `false` to continue.
- **Resume after a failure, crash or reboot:** if the queue process is gone, delete
  `<root>/queue.lock`, then run the same command with `--resume`. Completed runs are
  kept; the interrupted run starts again from scratch and its partial directory is
  recorded under `abandoned` in `batch.json`.

The training dashboard (`docs/TRAINING_DASHBOARD.md`) reads the same queue
files, but its labels and colours only know the sparse, shaped and survival
profiles, so these runs show up unlabelled there.

## Stage 1: screen (6 × 0.5M decisions, model seed 42)

| File | HP weight | Upgrade reward | Question |
|---|---:|---:|---|
| `1-screen-shaped.json` | 1.0 | 0 | Baseline on the fixed environment |
| `1-screen-upgrade.json` | 1.0 | 0.10 | Does paying for upgrades alone help? |
| `1-screen-half-hp.json` | 0.5 | 0 | Does lower healing pressure alone help? |
| `1-screen-deckbuilder.json` | 0.5 | 0.10 | Both (same weights as `deckbuilder.json`) |
| `1-screen-sparse.json` | — | — | Control: best holdout wins so far (12/200) |
| `1-screen-turn-cost.json` | 1.0 | 0 | Does a cost per extra combat turn help? |

The turn-cost arm is the baseline with `combat_turn: -0.005` (see
`local/config/rewards/README.md`): it asks whether faster fights win more. Judge
it on wins and HP lost as well as `averages.turns_per_combat`; shorter fights
that cost more HP are not an improvement. A shaped screen trained before this
term existed has no turns-per-combat figure; rerun it to compare tempo.

The final test is the 200 development seeds (20000–20199), so results line up
seed-for-seed with the old checkpoints: sparse best 12/200, shaped best 10/200,
boss reach 39/200 and 64/200. Only the baseline arm runs the 50-episode random
baseline; it is identical for every arm.

**Decide before looking at results:**

1. Every arm must show 0 stalls and at most 1% timeouts in `best-results.json`,
   and a `rollout/loop_rate` near 0. Otherwise fix the environment first.
2. Drop arms that still never upgrade: compare `averages.upgraded_cards` with
   0.67 (old shaped) and rerun the campfire probe (`scripts/training_audit_probes.py`)
   for Rest at full HP.
3. Rank the remaining arms by `best-results` win rate, then average floor. At about
   5% wins, 200 seeds give a ±3-point interval, so a single screen run cannot
   prove one arm is better; it only picks what to replicate.
4. Take forward the strongest arm that is not the baseline. If none beats the
   shaped screen on wins or floor, replicate shaped alone and stop changing rewards.

## Stage 2: replication (3 model seeds × 1M decisions per arm)

`2-replicate-shaped-s42/s43/s44.json` are the baseline. For the Stage 1
candidate, copy these three files, then change `name` and `reward_overrides`
(or `reward`) only. Model seeds differ while the game-seed schedule stays the
same, so candidate and baseline runs with the same model seed are paired.

The final test is the fresh 1,000-seed holdout (30000–30999). Report the mean
and spread over the three seeds and the seed-paired differences. Do not use
these holdout numbers to pick further settings.

## Stage 3: horizon (optional, after Stage 2)

`3-horizon-shaped-gamma998.json` is `2-replicate-shaped-s42.json` with gamma
0.998 (about a 500-decision horizon instead of 200), to test whether deck
choices need a longer horizon. Compare it with that run only.

## Rough cost on one RTX 4090

Based on the local 1M-step shaped run (43 minutes) and 100-episode CPU
evaluations (about 3 minutes):

- Stage 1: about 35 minutes per arm, about 3.5 hours for the six arms.
- Stage 2: about 1 hour 40 minutes per run, because the 1,000-episode final and
  best evaluations take about 30 minutes each. That is about 5 hours per arm.

## Not covered by a config

These need code changes and come after this experiment:

- A dedicated confirmation head (Confirm currently shares the End Turn output).
- Learning-rate decay.
- The critic coefficient (effectively 0.25 × MSE here, against 0.5 × MSE in SB3).
- KL, explained variance and per-phase entropy logging.
- A campfire-specific entropy bonus.

`../entity-act1-deckbuilder.json` (5M steps) is superseded by the screen.

# Configure a training experiment

Start here for the local configurable runner. The old `Train.cmd` still runs the
original combat-only script. The upstream `scripts/train_full_run.py` is stale
and passes constructor arguments the current simulator does not support; use
the runner described here for Act 1.

## Quick start

1. Open `local/config/training.json`.
2. Change `seeds.values` from `[42]` to your chosen integer simulator seed.
3. Double-click `Train-Run.cmd`.

The supplied `local/config/training.json` starts a **new Ironclad Act 1 model on seed 42**.
It requests 500,000 training decisions, four parallel environments, and CPU
training. The PPO rollout size means the actual total can round up slightly.
Fixed-seed deterministic evaluations use a single attempt by default because
repeating the identical trajectory would not add evidence.
An Act 1 attempt succeeds as soon as the Act 1 boss is defeated. Death or the
episode limit is a failure. Rewards come from the `shaped` reward profile (see
[Rewards](#rewards)). This trains by repeated attempts, not exhaustive branch search.

From PowerShell in the project folder:

```powershell
# Validate and show all resolved settings without training:
.\.venv\Scripts\python.exe -m sts2_env.training.runner --config local/config/training.json --check

# Run a different saved experiment configuration:
.\Train-Run.cmd local\config\presets\act1-random.json
.\Train-Run.cmd local\config\presets\act1-seed-pool.json
.\Train-Run.cmd local\config\presets\default-combat.json

# A short diagnostic run, not a competitive trained model:
.\Train-Run.cmd local\config\presets\smoke-act1.json
```

## Seed modes

Edit the `seeds` object:

```json
{"mode": "fixed", "values": [42], "schedule_seed": 123}
```

`fixed` restarts **the same simulator seed every episode in every worker**.
The policy still explores different actions while training. The same seed plus
the same decisions reproduces the same simulator run. For combat-only training,
the integer is the Gym combat-reset seed, which selects an encounter/deck shuffle;
it is not a full-run map seed. For Act 1/full-run training it is passed directly
to `RunManager`.

`pool` cycles through the integers in `values`, such as `[42, 123, 456]`.
`random` chooses a fresh seed from a repeatable random sequence; `values` is
ignored. `schedule_seed` controls that sequence. Parallel workers have separate
sequences, so changing `n_envs` also changes the training seed schedule.

These are **integer simulator seeds**. Displayed Steam game seed strings are not
accepted or automatically converted. Simulator fidelity and seed matching with
the installed game still require separate verification.

`training.model_seed` seeds neural network initialisation and policy sampling.
Changing it does not change a fixed game seed. It is useful for comparing training
attempts on the same scenario. Bit-for-bit training reproducibility across hardware,
library versions or operating systems is not guaranteed.

## Scenario controls

- `task`: `combat`, `act1`, or `full_run`. `combat` preserves the existing
  starter-deck combat task. `act1` stops at the first act's boss victory.
  `full_run` uses the simulator's normal run-completion rules.
- `character`: `Ironclad`, `Silent`, `Defect`, `Necrobinder`, or `Regent`.
- `ascension`: 0–9. The configurable runner rejects 10+ because its Act 1
  success condition has not been validated for double-boss acts. Some ascension
  effects are only partially implemented in the underlying simulator.
- `start_with_neow`: `false` starts at the map, matching the existing simulator
  training default. `true` includes the opening event.
- `training.max_episode_steps`: hard decision limit per attempt.
- `training.max_combat_turns`: combat-turn cap supplied to the environment.
  Combat-only episodes apply this directly; run mode inherits the current
  run environment's enforcement of this cap.

The existing combat-only environment supports Ironclad at ascension 0 only;
other scenario settings are rejected for that task instead of silently ignored.
Training raises simulator exceptions so bugs are not silently reported as deaths.

## Model

```json
"model": {"type": "entity", "d_model": 128, "layers": 3, "heads": 4}
```

- `type`: `entity` (default) trains SpireFormer, the object-based Transformer
  policy, on any task. `flat` trains the older MLP on the flat 131/151-number
  observation and is kept as a comparison baseline.
- `d_model`, `layers`, `heads`: SpireFormer size (entity only). `d_model` must be
  divisible by `heads`. See [Entity policy](ENTITY_POLICY.md).

SpireFormer runs its network on the GPU and steps all `n_envs` games in one
process, so use more workers and shorter rollouts than the flat model, e.g.
`n_envs: 16`, `n_steps: 256`, `batch_size: 512`, `n_epochs: 4`.

## Rewards

Reward profiles are JSON files in `local/config/rewards/`, kept next to the
training presets. Pick one per experiment:

```json
"reward": "shaped",
"reward_overrides": {"floor": 0.2}
```

- `reward`: a profile name from `local/config/rewards/`, a `.json` path, or an
  inline object of weights. If omitted, `combat` uses `sparse` and
  `act1`/`full_run` use `shaped`.
- `reward_overrides`: change single weights for this experiment only.

Supplied profiles:

- `sparse`: +1 win, -1 death or timeout. Identical to training before profiles
  existed; the original presets are pinned to it so their results stay comparable.
- `shaped` (default for runs): win +5, death/timeout -1, +0.1 per floor, HP change on
  the Dota curve (weight 1), +0.3 per elite won, up to +0.05 per combat for enemy HP
  removed, +0.01 per max HP point, +0.05 per relic.
- `survival`: like `shaped`, with HP loss weighted twice as heavily.

The design follows OpenAI Five's Dota 2 reward: the objective result is the
largest term, shaping is paid on *changes* in state so nothing is counted twice,
and HP uses `(x + 1 - (1-x)^4) / 2` so losing HP when already low costs most.
Keep total shaping per episode below `win`. `local/config/rewards/README.md`
lists every key.

The resolved weight table is saved in each experiment's `config.json`.
Per-term episode totals are logged to TensorBoard under `reward/` (entity model)
and as `reward_<term>` columns in evaluation CSVs, with averages in
`average_reward_terms` of the results JSON.

## Training controls

- `total_timesteps`: new environment decisions to train for, summed across workers.
- `n_envs`: parallel workers; start with 4.
- `learning_rate`: speed of neural network updates.
- `n_steps`: decisions per worker before a PPO update.
- `batch_size`: update minibatch size; must divide `n_steps * n_envs`.
- `n_epochs`: update passes per rollout.
- `gamma`: how strongly future rewards count.
- `ent_coef`: exploration incentive.
- `hidden_sizes`: flat model layer widths, e.g. `[256, 256]` (ignored by `entity`).
- `device`: `cpu`, `auto`, or `cuda`. CUDA-enabled PyTorch 2.11.0+cu128 is installed and verified on the RTX 4090. The simulator remains CPU-based; CUDA handles network calculations.
- `torch_threads`: CPU threads for network operations.
- `checkpoint_every`: training decisions between periodic model saves.
- `resume_from`: `null` starts fresh. Set a saved model path (`.pt` for entity, `.zip` for flat) to continue
  training it for `total_timesteps` **additional** decisions. Relative model
  paths resolve from the project root. The network shape, task observation/action
  spaces and rollout settings must match the saved model. Resume starts new
  episodes and does not restore the exact previous environment/RNG state.

The old 131-input combat model cannot be resumed as the 151-input Act 1 model.
`Play-AI.cmd` continues using the old combat model. The new full-run policy is
not automatically installed into the live-game bridge.

## Evaluation controls

- `mode: "training_seeds"`: evaluate using the configured fixed/pool/random
  training seed rule. For random mode this is a separate reproducible seed sequence.
- `mode: "seed_list"`: evaluate on `evaluation.seeds`, cycling if needed.
- `mode: "unseen"`: use `evaluation.seeds` and reserve them from random training.
  Overlap with a fixed seed or seed pool is rejected.
- `every_steps`: evaluate periodically and save the best reward-scoring policy.
- `episodes`: attempts per periodic evaluation.
- `final_episodes`: attempts per final and best-model evaluation.
- `deterministic`: choose the highest-scoring action at evaluation time when true;
  false samples from the policy. Training always samples exploratory actions.
- `random_baseline_episodes`: random legal-action attempts on the same evaluation
  seed schedule before training; set to 0 to skip.

**Repeated deterministic attempts on a single fixed seed reproduce one trajectory.**
20/20 wins there means the model can repeat that solution, not that it has a
measured 100% win rate on unfamiliar runs. Generalisation needs reserved seeds.
Stochastic evaluation on one seed instead measures variation in policy actions.

## Outputs and statistics

Each invocation creates a new folder under `output/experiments/` containing its
name, a UTC timestamp, and a unique suffix. Existing models are not overwritten.

- `config.json`: complete settings used, with omitted defaults filled in.
- `metadata.json`: source config path and relevant interpretation notes.
- `random-baseline.json` / `random-baseline-episodes.csv`: baseline results.
- `best_model/best_model.pt` (entity) or `.zip` (flat): best periodically
  evaluated policy, ranked by win rate, then average floor (entity).
- `checkpoints/`: periodic backups.
- `final_model.pt` (entity) or `final_model.zip` (flat): policy at the end of training.
- `final-results.json` / `final-results-episodes.csv`: final model's results.
- `best-results.json` / `best-results-episodes.csv`: saved best model's results.
- `<label>-traces.jsonl`: for each stalled, timed-out or looping episode, the
  decisions that explain it (state fingerprint, action, card-selection prompt,
  limits, selected indexes before and after, and whether anything changed).
- `<label>-relic-stats.csv` / `<label>-card-stats.csv` (run tasks): for each
  relic or card in the final deck, how many runs had it, average copies, and the
  win rate with and without it. Upgraded cards count as the base card.
- `eval_logs/` and `tb_logs/`: evaluation history and TensorBoard logs. Entity
  runs also graph `rollout/gold_mean`, `rollout/deck_size_mean` and `rollout/relics_mean`.

Results include wins, losses and attempts, win rate, distinct seeds, boss reached rate,
Act 1 win rate once the boss is reached, average HP after wins, average/maximum
floor, decisions, reward (total and per term), timeouts, stalls and deaths grouped by room type.
Timeouts, stalls and deaths are counted separately. Fixed-seed deterministic repetitions are correlated;
do not treat them as independent evidence of performance.

A **stall** is a deterministic evaluation that repeats the same state and action,
or the same cycle of up to four, three times in a row. The policy would repeat it
until the decision limit, so the attempt stops there and scores as a timeout in
the reward. `loop_start` and `loop_period` in the episode CSV locate the cycle.
The combat turn is part of the state, so identical turns in a stalemate are not
stalls; `training.max_combat_turns` ends those fights as losses.
Sampled training and evaluation episodes can leave such a cycle, so they only
record it: the entity trainer logs the share of episodes with one as
`rollout/loop_rate`. `Watch-Model.cmd` stops on the same rule in the real game.

Card selections are add-only for policies: a selected card is never offered
again, and confirmation becomes the only legal action once the maximum is
selected. Every final selection is still reachable, and a selection can take at
most `max + 1` decisions.

Run tasks (`act1`, `full_run`) also record the end-of-run state of each episode:
act, gold, max HP, deck size and composition (upgrades, attacks, skills, powers,
curses/statuses), relics, potions held, card removals, rooms visited by type, and
the full deck and relic lists (`;`-separated). The summary JSON averages these
under `averages` and compares gold, deck size, relics, upgrades and floor for
wins, losses, timeouts and stalls under `by_outcome`.

Press Ctrl+C once to save `interrupted_model.zip` and close the workers. A force
kill or system shutdown may only leave periodic checkpoints.

## Limitations

With `model.type: "entity"` the agent sees its deck, relics, potions, offered
cards/relics/potions, shop prices, rest and event options, and a summary of what
lies down each map path. The `flat` baseline still has limited non-combat
information. The live-game watch tool and bridge only run flat models so far.
Configuration makes experiments reproducible; it does not
guarantee a strong Act 1 agent or an exact recreation of the live game. Test the
simulator and observation design further before interpreting results as optimal
play. See `KNOWN_ISSUES.md` and the parity backlog for existing fidelity gaps.


# Live reward comparison

The local dashboard watches a durable queue of three independent SpireFormer
experiments: sparse, shaped, then survival. Only one trainer runs at a time.
Closing the browser or dashboard does not interrupt the separate queue process.
Keep the computer awake for training to continue.

Each model starts fresh and receives 5,000,000 training decisions on random
Ironclad Act 1 runs at ascension 0. The three experiments share the model seed,
seed schedule, architecture, and PPO settings. Different actions and episode
lengths mean they will not see identical trajectories.

Checkpoints are requested every 100,000 decisions and saved at the first completed
PPO rollout after each threshold. With 16 environments and 256 steps, a rollout
contains 4,096 decisions, so actual save steps are recorded in the dashboard.
The training total also rounds up to a completed rollout.

Validation runs every 250,000 decisions on seeds 10000–10049. The best checkpoint
is selected by validation win rate, then average floor. After training, both final
and best checkpoints are evaluated on 200 separate seeds, 20000–20199. Both seed
sets are excluded from random training. One model seed per reward profile is an
initial comparison, not a repeated-seed significance study.

Prepare a new queue (the directory must not already exist):

```powershell
.\.venv\Scripts\python.exe -m sts2_env.training.comparison --prepare --root output/my-comparison
```

Run the queue in one terminal and the dashboard in another:

```powershell
.\.venv\Scripts\python.exe -m sts2_env.training.comparison --root output/my-comparison
.\.venv\Scripts\python.exe -m sts2_env.training.dashboard --root output/my-comparison --port 8770
```

Open http://127.0.0.1:8770/. The dashboard refreshes every five seconds. It includes
validation curves, training metrics, reward contributions, episode statistics,
all resolved configuration variables, GPU telemetry, CSV exports, and live logs.

Completed **game runs** are now visible across the queue and for each reward
profile, alongside decisions per run and completed runs per hour. A game run
is one finished training attempt (win, loss or timeout); a model-training job
is labelled an **experiment**. Evaluation attempts are excluded from these
counts. Counters update at completed rollouts, using the most recent training
record even while the live status shows evaluation, checkpointing or completion.

Decisions per run is the mean length of the last 200 completed attempts (or
all attempts if fewer). It is not total decisions divided by completed runs:
total decisions also includes unfinished attempts in parallel environments.
Runs/hour is cumulative completed training attempts divided by elapsed hours
at the same rollout, including evaluation and optimizer overhead. Both metrics
are available in the metric explorer. Use floor reached and win rate to judge
learning: short runs and high throughput can also reflect early deaths.

Hide profiles, change plot range, smooth training curves, select a metric, or
freeze charts for inspection. Clicking a run card opens its configuration.
Configuration is read-only during training. Queue pause lets the current run
finish, then holds the next run until Resume is clicked.

The comparison table displays latest validation during training and the best
checkpoint's separate final test once available. Raw reward totals use different
scales; compare objective win rates and floor progression instead. Training
metrics average the last 200 completed exploratory episodes. Throughput and ETA
are estimates, including evaluation overhead encountered so far.

All files live under the selected batch directory. `configs/` freezes fully
resolved configurations and reward weights. `runs/` holds models, evaluation
CSVs, TensorBoard events, `metrics.jsonl`, and an atomic `live.json` snapshot.
`comparison.csv` and `comparison.json` summarise completed test results;
`batch.json` records queue state, exit codes, and process IDs. A failed or
interrupted trainer stops the queue. Duplicate queue launches are refused.
To continue a stopped queue, run it again with `--resume`: completed runs are
kept and the interrupted run starts again from scratch. The dashboard never
silently repeats or overwrites work.

`--prepare --plan <file>` queues any list of preset files instead of the three
profiles; `local/config/presets/reward-ablation/README.md` shows the reward
ablation queues. The dashboard's labels only cover the three profiles.

For a short end-to-end diagnostic, add `--smoke` when preparing a separate queue.
It uses the same model on CUDA but tiny rollouts and episode limits. Its results
are diagnostics, not model performance measurements.

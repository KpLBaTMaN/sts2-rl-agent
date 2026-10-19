# Local setup

Real-game watch player added 2026-10-16. Watch-Model.cmd loads the saved experiment
config/model selected by watch.json. Inference uses the 151/157 run policy via
watch adapter; controls are served at localhost:8766. WatchSession.cs applies
seed, character, ascension, optional opening-event bypass, isolated rl-watch save
paths, pausable decision waits and Act 1 completion. UI-only decisions are labelled.
The live seed 42 map has 51 nodes versus 68 in the simulator, with 35 overlapping
coordinates: identical run parity is NOT established. Full trace recorded under
output/watch/. Backup prior to mod installation: backups/before-watch-20261015/.
Live verification: map single-step, exactly one Defend card (block 5, energy 2),
play/pause with stable decision count, next-fight skip completed first combat,
rewards and an intermediate room, paused before the fight at floor 3 (47 decisions,
HP 77/87). Focused Python checks: 21 passed. Mod builds successfully against v0.107.1.

Configurable training added 2026-10-15: edit training.json and run Train-Run.cmd.
New sts2_env/training package supports fixed/pool/random game seeds, separate
model random seed, character/ascension/opening-event controls, checkpointing,
resume, and saved per-episode evaluation results. STS2RunEnv optional act_count
objective stops at final boss victory; default full-run behavior remains intact.
Configured training propagates simulator exceptions. The older upstream full-run
script is stale; use the new runner. New Act 1 models are separate from the
existing live-game combat model. See docs/CONFIGURABLE_TRAINING.md.
Verified short Act 1, combat and resumed training runs. The Act 1 wiring test
trained 1024 steps, evaluated and saved models/checkpoints/results. A resumed
model advanced from 1024 to 1152 steps and recorded 128 new timesteps.

Repository: zhiyue/sts2-rl-agent, commit 1b7e7ce35e608722650763938c153ea8bc370333.
Game: E:/SteamLibrary/steamapps/common/Slay the Spire 2, v0.107.1, Steam build 23811903.
Python: bundled 3.12.14, isolated .venv. Installed training/dev extras, TensorBoard, tqdm, rich.
Hardware: Ryzen 9 7900X and RTX 4090. CPU selected for the small MLP policy; four environments and four PyTorch threads.
Private build tools: C:/Users/Andrew/Documents/Codex/2026-10-15/h/work/tools, .NET SDK 9.0.319, Godot 4.5.1 Mono, ILSpy 9.1.
Save/settings backup: backups/sts2-before-ai.zip (steam saves and mod_configs).

Local training fixes: MaskableEvalCallback, native environment action masks for subprocesses, Monitor, explicit device and thread count, seeded policy.
Bridge fixes for installed game: CardRewardSelection struct, ICombatState types, per-player PlayerTurnPhase.Play, potion enum access, missing model/control namespaces.
Automation requires --sts2-rl (Play-AI.ps1 launches through Steam) or STS2_RL_ENABLED=1. Automated profile directories route to modded/rl-agent/profile{profileId}. Animation-speed prefix uses Harmony's positional argument __0 for compatibility.
Bridge staged under ../bridge-staging. Resource PCK exported separately from C:/Users/Andrew/Documents/Codex/2026-10-15/h/work/bridge-resources; no game assemblies packaged.
Mod localization analyzer disabled for the code-only bridge build.

Validation: 60 focused environment, combat, state adapter and protocol tests passed; pip check passed. 8,192-step training smoke test saved and reloaded models and completed 100 combat evaluations. Initial 500,000-step training launched with output/training.log.

Training completed 507,904 steps in 227 seconds; final policy won 83/100 simulator combats. Saved best policy reloaded successfully and chose a legal action. Steam load check loaded both mods. Headless AI check connected, navigated map, completed the first fight and selected a reward. Game log confirmed isolated rl-agent save paths. Test game exited at its bounded --quit-after limit; test agent was stopped. Play-AI uses --exit-on-disconnect to finish when the game closes.

Project relocated to E:/Coding Projects/sts2-rl-agent on 2026-10-15. Virtual environment editable package and console launchers repaired; save backup copied into backups/.


CUDA repair 2026-10-15: replaced CPU-only torch 2.14.1 with official torch 2.11.0+cu128 from https://download.pytorch.org/whl/cu128. Driver 610.74; RTX 4090 detected. CUDA matrix calculation, old saved model loading and 1024-step Act 1 CUDA training all passed. pip check passed. training.json remains configured for cuda.


Layout tidy 2026-10-18: local-only files moved under local/ (launchers, config,
config/presets, notes, backups, requirements lock). Root .cmd files are shortcuts
into local/launchers. Watch code moved to sts2_env/watch/ (player.py, adapter.py,
control.py, watch.html); WatchSession.cs moved to bridge_mod/Watch/.


Reward profiles added 2026-10-19: local/config/rewards/ holds sparse, shaped
(new default for act1/full_run) and survival profiles; training configs select
one with "reward" and tweak with "reward_overrides". Implemented in
sts2_env/training/rewards.py (RewardTracker in ScenarioEnv). Existing presets
pinned to sparse; new presets/entity-act1-shaped.json. Per-term rewards logged
to TensorBoard (reward/*) and evaluation CSVs. Enemy-damage credit is capped per
enemy because Fogmog's Illusion eye revives endlessly once Fogmog is dead (the
simulator does not end combat when only minions remain).

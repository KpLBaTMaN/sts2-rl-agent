# Watch a trained Act 1 policy in Slay the Spire 2

Close any existing Slay the Spire 2 window, then double-click **Watch-Model.cmd**.
Steam launches the installed game. A local browser control panel opens at
http://127.0.0.1:8766/. Watch the game window and use the browser controls beside it.
It starts paused before the first policy decision. Accept the game's mod notice
if it appears. Leave the launcher running while watching.
Use the game as a viewer while automation is active; manual card/map clicks can
change the state underneath a pending model decision.

## Controls

- **Step once** sends one pending decision (card, end turn, map choice, reward,
  shop or event). Animations and enemy responses to that decision can finish.
- **Play / Pause** continuously send decisions or pause before the next one.
- **Next room** plays forward and pauses before the next room's first decision.
- **Next fight** plays forward until the next combat and pauses before its first
  card decision. From a fight, this means the following fight.
- **Run to boss** plays forward until the next boss combat. Death ends the run.
- **Delay** sets the wait between decisions during ordinary playback.
- **Stop** stops automation and leaves the game open. Close the game when finished.
  Closing the launcher also disconnects and stops watch-mode automation.

Skip controls play decisions, they do not teleport or seek a saved replay.
There is no rewind. Restart Watch-Model.cmd to start the seeded run again.
The model can lose, loop or hit its configured decision limit; its initial Act 1
training did not produce an Act 1 victory. The panel shows action probabilities
to help inspect what it is doing.
The percentages are probabilities assigned to actions, not chances of winning.

## Select the model and configuration

Edit `local/config/watch.json`:

```json
{
  "experiment": "output/experiments/act1-seed42-20261015-221936-d15e69",
  "model": "final",
  "seed": null,
  "start_paused": true,
  "decision_delay": 0.8,
  "port": 8766,
  "game_path": "E:/SteamLibrary/steamapps/common/Slay the Spire 2",
  "steam_path": "C:/Program Files (x86)/Steam/steam.exe"
}
```

`experiment` is a completed training folder. `"latest"` chooses the most recent
non-smoke experiment. `model` is `"final"` or `"best"`. Best checkpoints can be
worse than final models when their sparse evaluation scores tie.

The experiment's saved `config.json` supplies character, ascension, opening-event
setting and Act 1/full-run objective. Editing the current training.json does not
rewrite the configuration attached to a saved model.

`seed: null` uses the saved fixed seed, the first seed of a saved pool, or a newly
chosen seed for a random-seed experiment. An explicit integer or real-game seed
string overrides it. Integer seed 42 is passed to the game as string `"42"`;
both implementations hash that text to seed their named RNG streams.

Watch mode uses CPU inference for the small policy. GPU is still available for
training. It requires the 151-input / 157-action Act 1/full-run model. The old
131-input combat model and Play-AI.cmd remain separate.

## Important: same settings are not an identical simulated run

The Python simulator and installed game are different implementations. Applying
the same seed, character and ascension does **not** prove matching maps, enemies,
card pools, reward order or outcomes. The player reports node counts and coordinate
overlap between the live map and the simulated map. This is a diagnostic, not a
certificate of map equivalence. The current simulator uses approximate encounter
pools, and the isolated live profile can have different unlocks.

For `start_with_neow=false`, the mod bypasses the opening event without granting a
blessing and starts at its map choices. Other game initialization remains native.
Act 1 watch runs stop automation when the Act 1 boss is defeated; they do not
continue into Act 2. Other characters and ascensions are configurable but have
less live validation than the default Ironclad/ascension-0 case.

## Which decisions use the model?

Combat, map routes, card rewards, rest sites, shop options and ordinary event
options are mapped to the same policy action slots as training. The 20 run-state
features use the training encoder's scales. Combat observation is retained for
noncombat phases as it is in the simulator.
The opening-point offset is removed from the Act 1 floor feature so the first
ordinary room has the same floor encoding used during training.

The real game has collection menus and special screens that the simulator skips
or models differently. Gold/reward collection, proceed buttons, card bundles and
the Crystal Sphere screen use explicit bridge handling. The panel labels those
as **UI transition / game-specific screen (not trained)**, rather than claiming
the policy chose them. Multi-card selection uses the policy's ranked choices to
fill the game's required selection count. This adaptation is not a full proof of
simulator/live-action semantic parity.

## Saves, records and installation

Watch-only automation activates with the `--sts2-watch` Steam launch flag. Normal
launches do not start the bot. Saves use the separate directory:

`%APPDATA%/SlayTheSpire2/steam/<account>/modded/rl-watch/profile<id>`

Each session records its model/configuration and JSONL state/action trace under
`output/watch/<timestamp>/`. `summary.json` is written when the agent finishes or
stops. These files let us diagnose transfer failures; they are not game save files
and cannot rewind the live game.

The installed mod is `mods/STS2BridgeMod/STS2BridgeMod.dll` in the Steam game
folder. Its generated `watch-session.json` contains the configuration for the
next launch. The previous DLL and save backup are in `local/backups/before-watch-20261015/`.
The normal training script and saved experiment models are preserved.

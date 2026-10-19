# Parity Gaps to Exact Match

This document tracks the currently confirmed blockers between `sts2_env` and the decompiled game logic under `decompiled/`.

As of 2026-05-18, the correct status is:

- Several gaps called out by earlier audits are now fixed in code.
- The previous version of this document had become stale and is superseded by this rewrite.
- Exact parity is still not guaranteed.

## Recently Fixed

The following items were previously listed as major parity blockers and are now implemented or aligned:

- `sts2_env/core/combat.py`
  - `summon_osty()`
  - `auto_play_from_draw()`
  - `generate_card_to_hand()`
  - `generate_ethereal_cards()`
  - owner-aware Osty lookup / summon / kill for allied player-creatures
- Combat potion support across the RL stack
  - fixed combat action-space layout for potions
  - simulator potion masking / decoding / execution
  - bridge-side potion masking / decoding / execution
- Simulator/bridge observation alignment for pile-summary composition features
  - the simulator now keeps the bridge-only-unavailable composition slots zeroed
- Seeded RNG parity
  - `Rng` now follows the seeded `System.Random` compatibility sequence used by the game
  - named run streams now use the game's deterministic string hash and stream-name derivation
  - player-level `Rewards`, `Shops`, and `Transformations` streams now derive from `run seed + player id`
  - standard and `SpoilsMap` act-map RNG streams now derive from the hashed run seed and map stream names
  - event RNG streams now derive from the hashed run seed plus event id, with player id omitted for shared events
  - event gold/cost rolls and `PunchOff` event HP reductions now use the original exclusive upper-bound RNG semantics
  - selected event random-transform paths now use the event RNG where the decompiled event calls `base.Rng`
  - ancient relic-option rolls for `Darv`, `Nonupeipe`, `Orobas`, `Pael`, `Tanx`, `Tezcatara`, and `Vakuu` now use event RNG instead of the run upfront stream
  - `JungleMazeAdventure` gold variance now uses the original float range roll instead of an integer roll
  - act-map room count rolls now follow the original per-act RNG consumption order
  - combat now routes initial/discard shuffles, random draw-pile insertion, monster move rolls, random targets, generated cards, generated potions, random card selection, random energy costs, and random orb generation through the matching named run RNG streams
  - generated-card paths for `InfernalBlade`, `Distraction`, `WhiteNoise`, `Discovery`, `Metamorphosis`, `Jackpot`, `JackOfAllTrades`, `Splash`, `Quasar`, `Largesse`, `ManifestAuthority`, `BundleOfJoy`, `MadScience`, combat card-generation helpers, combat card-generating potions, `Toolbox`, `ChoicesParadox`, `BigHat`, `OrangeDough`, `Crossbow`, and `VexingPuzzlebox` now use `CombatCardGeneration`
  - `CallOfTheVoidPower` now performs repeated one-card generation with `CombatCardGeneration`, matching the original duplicate-allowed loop
  - stacked `HelloWorldPower` now generates distinct common cards through one `CombatCardGeneration` call, matching the original
  - `SpectrumShiftPower` now generates distinct colorless cards through the same `CombatCardGeneration` helper used by the original
  - card-selection paths for `TrueGrit`, `Thrash`, `NeowsFury`, `SeekerStrike`, `HiddenGem`, `DrainPower`, `Bookmark`, `MummifiedHand`, `PowerCell`, and `JeweledMask` now use `CombatCardSelection`
  - random-target hooks including `JuggernautPower` now use `CombatTargets` instead of Python/global randomness
  - `StampedePower` now auto-plays a random playable attack from hand using the original `Shuffle` stream
  - random potion procurement from combat and combat-related relic hooks now uses `CombatPotionGeneration`
  - `Chaos`, `channel_random_orb()`, `Slither`, `Confused`, and `SneckoOil` now use the matching `CombatOrbs` or `CombatEnergyCosts` stream
  - Niche-backed relic selection for `WarPaint`, `Whetstone`, `RoyalStamp`, `FragrantMushroom`, `SandCastle`, `SereTalon`, and `WarHammer` now uses the `Niche` stream instead of the rewards stream
  - `StoneCracker` now upgrades boss-combat draw-pile cards with `CombatCardSelection`, matching the original draw-pile timing instead of upgrading the persistent deck on room entry
  - stable-shuffled card-pile selections for `Catastrophe`, `Uproar`, `BeatDown`, `SeekerStrike`, `PowerCell`, and `StoneCracker` now sort card candidates before consuming the matching RNG stream
  - `EntropyPower` now requests hand-card selection and transforms the selected cards with `CombatCardSelection` instead of calling a missing no-op helper
  - `LeafyPoultice` now uses the player `Transformations` stream for both immediate and deferred transform flows, while other default run-level transforms keep using `Niche`
  - `ArchaicTooth` setup-attribute deferred rewards now use the supported mapping transform path, avoiding an invalid reward-helper call while preserving upgrades/enchantments through the mapping replacement logic
  - `ScrollBoxes` now presents two generated three-card bundles and adds the selected whole bundle, rather than queuing two independent single-card reward screens
  - `PaelsTooth` now uses a deck-choice flow for selecting up to five stored cards when deck choices are enabled, instead of always removing the first five upgradable cards
  - `WhisperingEarring` now uses `CombatTargets` when its opening auto-play picks an allied player target, instead of always taking the first ally
  - `AlchemicalCoffer` now fills the four newly added potion slots by index, instead of filling pre-existing empty potion slots first
  - `DelicateFrond` now fills combat-start potion slots from the out-of-combat potion pool, and combat potion procurement now honors `Sozu`
  - `BiiigHug`, `BlessedAntler`, and `TeaOfDiscourtesy` now insert generated draw-pile cards at random positions instead of placing them on top
  - `Reave` now places generated `Soul` cards at the bottom of the draw pile, matching the default generated-card insertion position
  - `AggressionPower` now uses an unstable `CombatCardSelection` shuffle of discard-pile attacks, instead of sorting candidates before selection
  - `FurCoat` now derives its marked-room RNG seed from the run RNG seed plus player id and the game's deterministic hash, matching the original room-marking seed
  - draw-pile auto-play now first moves all selected cards out of draw before playing them, matching `AutoPlayFromDrawPile` timing for `Mayhem`, `Cascade`, and related helper flows
  - `Orrery`, `LostCoffer`, `LeadPaperweight`, and `MassiveScroll` card rewards now use non-combat reward pools instead of combat-only filtering where the original uses `CardCreationSource.Other`
  - `SeaGlass` now defaults an unset character to Ironclad, generates its 15 options without upgrade rolls or combat-only filtering, and supports taking multiple cards before skipping
  - `GlassEye` now generates its five fixed-rarity card rewards without combat-only filtering or upgrade rolls, matching the original non-combat uniform reward options
  - `ScrollBoxes` now builds its bundles from non-combat common/uncommon character-card candidates instead of combat-only candidates
  - card reward metadata now distinguishes encounter vs other reward sources and card-pool-modification locks, so `DingyRug`, `PrismaticGem`, and `LastingCandy` no longer modify reward pools the original protects from those hooks
  - `MassiveScroll` now generates options from the original custom multiplayer-only pool instead of the whole owner-character plus colorless card pool
  - merchant purchase hooks now receive actual gold spent, shop card removal is single-use per shop, `LordsParasol` triggers its free card-removal purchase, `MawBank` ignores free purchases, and `Hoarder` blocks merchant card removal while duplicating newly added deck cards
  - `Midas`, `NightTerrors`, and `Vintage` run modifiers now apply their reward / rest-site behavior through the run modifier hook pipeline
  - `BigGameHunter` now regenerates maps with more elite nodes and forces eligible elite card rewards to rare cards while respecting locked reward options
  - `CharacterCards` now expands both card reward pools and merchant character-card pools with the selected extra character pool
  - `CursedRun` now adds a random modifier-eligible curse after each act is entered, and `DeadlyEvents` now updates unknown-room elite odds, treasure odds growth, and Juzu removal from relic pools
  - Fake Merchant relics now have focused coverage, and `FakeHappyFlower` now resets its turn counter after triggering
  - `Circlet` and `DeprecatedRelic` now behave as stackable placeholder relics, while normal relic duplicates are still rejected by canonical id
  - `HappyFlower` now uses the original looping turn counter instead of leaving its saved counter at 3, 6, 9, ...
  - `PaelsTears` now keeps its leftover-energy marker until the next player turn end rewrites it, matching the original timing
  - `SelfFormingClay` now applies its own block-clear power instead of the generic next-turn block power, and no longer triggers that power early on later damage
  - `FiendFire` now stops later hits if the attacker dies during the multi-hit attack, matching the original attack loop
  - Enemy-wide relic effects such as `BagOfMarbles`, `MercuryHourglass`, `CharonsAshes`, `StoneCalendar`, `FestivePopper`, `LetterOpener`, `RedMask`, `TwistedFunnel`, `Metronome`, `ScreamingFlagon`, `LostWisp`, and `MrStruggles` now target only hittable enemies where the original uses `HittableEnemies`
  - Lightning, Dark, and Glass orb damage now uses the original unpowered damage flag, and Dark / Glass target only hittable enemies
  - Hand-written card attack paths now skip dead attackers, and multi-hit loops stop later hits when the attacker dies, matching the original damage / attack commands
  - The damage pipeline now fires block-broken, HP-changed, damage-given, and damage-received hooks in the original order; killed targets no longer receive target `AfterDamageReceived`, and overkill damage no longer inflates the actual unblocked damage passed to hooks
  - Damage-result metadata now exposes block-break / fully-blocked / total-damage details to result-sensitive hooks, fixing `Imbalanced`, `ReaperForm`, `SicEm`, and `HandDrill` parity cases
  - `AfterDamageReceived` result-sensitive powers now receive the same block / fully-blocked context where needed, fixing blocked-hit `Slippery` decrement and non-fully-blocked `VitalSpark` energy behavior
  - `PersonalHive` now creates Dazed cards in the attacker's draw pile and redirects Osty attacks to the pet owner's draw pile, matching the original `AfterDamageReceived` behavior
  - `VitalSpark` now tracks its once-per-turn energy trigger per attacking player instead of once per power, matching the original multiplayer trigger set
  - `EmotionChip` now uses the original previous-turn `!WasFullyBlocked` trigger condition, so non-fully-blocked zero-damage hits can activate it while fully blocked hits do not
  - `Reflect` now runs through the normal `AfterDamageReceived` hook order and reads the active damage result's blocked amount, instead of firing from a separate early blocked-damage shortcut
  - Direct `kill_creature()` calls now emit the same HP-change hook before death processing that the original `CreatureCmd.Kill` emits
  - `TheGambit` now uses the full kill flow after its triggering hit, so death hooks and cleanup run instead of only setting HP to zero
  - `NecroMastery` now triggers from Osty HP loss via `AfterCurrentHpChanged`, including fatal hits and direct Osty kills, matching the original hook
  - `Buffer` now runs in the late after-Osty HP-loss modifier pass, so `TungstenRod` can fully reduce 1 damage before `Buffer` decides whether to consume
  - `HardenedShell` now runs in the late before-Osty HP-loss modifier pass, so its cap applies before damage can be redirected to Osty
  - `SuckPower` now ignores same-side attack results and performs pet-owner result filtering before counting unblocked hits, matching the original `AfterAttack` behavior
  - `Gigantification` now tracks and consumes the actual attack command instead of card-play state, matching the original `BeforeAttack` / `AfterAttack` lifecycle
  - card clone / dupe paths for `DualWield`, `AdaptiveStrike`, `Undeath`, `HeirloomHammer`, `HistoryCourse`, and shared combat clone helpers now allocate new card instance ids without consuming combat RNG, matching the original `CreateClone` / `CreateDupe` behavior
- `sts2_env/gym_env/run_env.py`
  - step exceptions are logged instead of being silently converted into losses
- Explicit card-choice parity fixes
  - Silent: `Acrobatics`, `DaggerThrow`, `Prepared`, `HiddenDaggers`
  - Defect: `Scavenge`, `FlakCannon`
- Additional colorless parity fixes and tests
  - `Alchemize`
  - `BeatDown`
  - `HandOfGreed`
- Status / curse / quest parity fixes and tests
  - status and curse no-op `OnPlay` bodies audited against decompiled models
  - `Beckon`, `BadLuck`, `Decay`, `Doubt`, and `Shame` turn-end-in-hand hooks covered
  - `Debris`, `SporeMind`, and `Enthralled` playable no-op behavior covered
  - random curse generation now uses the decompiled modifier-eligible curse pool and excludes legacy `Pain` / `Parasite`
  - `LanternKey` now forces Act 3 unknown rooms into `WarHistorianRepy`
  - `SpoilsMap` now generates the Act 2 center treasure quest and completes for 600 gold/removal
  - deprecated Act 3 event stubs are covered as disabled / optionless
- Additional Defect / Silent parity tests
  - `Compact`
  - `WhiteNoise`
  - `TheHunt`
- Additional Regent / Necrobinder parity tests
  - `RefineBlade`
  - `SeekingEdge`
  - `TheSmith`
  - `SolarStrike`
  - `SpoilsOfBattle`
  - `WroughtInWar`
  - `KnockoutBlow`
  - `Charge`
  - `Guards`
  - `BigBang`
  - `HiddenCache`
  - `Resonance`
  - `SummonForth`
  - `Bulwark`
  - `Conqueror`
  - `Convergence`
  - `Glimmer`
  - `DecisionsDecisions`
  - `Quasar`
  - `HeirloomHammer`
  - `CrashLanding`
  - `BlackHole`
  - `Furnace`
  - `Orbit`
  - `PaleBlueDot`
  - `Bodyguard`
  - `Reanimate`
  - `Seance`
  - `Afterlife`
  - `GraveWarden`
  - `PullAggro`
  - `LegionOfBone`
  - `Spur`
  - `NecroMastery`
  - `Dirge`
  - `Eidolon`
  - `Protector`
  - `BoneShards`
  - `Rattle`
  - owner-aware `HighFive`
  - `DrainPower`
  - `SculptingStrike`
  - `EndOfDays`
  - `GlimpseBeyond`
  - `Severance`
  - `SoulStorm`
  - `TheScythe`
  - `TimesUp`
  - `BorrowedTime`
  - `CountdownCard`
  - `DanseMacabre`
  - `DeathMarch`

## Tests Covering Previously Listed Gaps

Recent targeted tests now cover several flows that older parity notes incorrectly described as still missing:

- `tests/test_combat_parity.py`
- `tests/test_card_choice_parity.py`
- `tests/test_silent_choice_parity.py`
- `tests/test_defect_choice_parity.py`
- `tests/test_regent_parity.py`
- `tests/test_necrobinder_parity.py`
- `tests/test_combat_potion_action_space_parity.py`
- `tests/test_bridge_state_adapter.py`
- targeted helper coverage in `tests/test_parity_helpers.py`

In particular, these areas now have direct automated coverage:

- Wish / draw-pile choice ordering
- Secret Weapon / Secret Technique draw-pile filtering
- Discovery / Purity / Dredge / Cleanse choice flows
- Nightmare snapshot behavior
- Osty summon helpers
- focused Regent card flows such as `Begone`, `PhotonCut`, `Largesse`, `ManifestAuthority`, `VoidForm`, `RefineBlade`, `SeekingEdge`, `TheSmith`, `SolarStrike`, `SpoilsOfBattle`, `WroughtInWar`, `KnockoutBlow`, `Charge`, `Guards`, `BigBang`, `HiddenCache`, `Resonance`, `SummonForth`, `Bulwark`, `Conqueror`, `Convergence`, `Glimmer`, `DecisionsDecisions`, `Quasar`, `HeirloomHammer`, `CrashLanding`, `BlackHole`, `Furnace`, `Orbit`, and `PaleBlueDot`
- focused Necrobinder card flows such as `CaptureSpirit`, `Sacrifice`, `Transfigure`, `Undeath`, `Bodyguard`, `Reanimate`, `Seance`, `Afterlife`, `GraveWarden`, `PullAggro`, `LegionOfBone`, `Spur`, `NecroMastery`, `Dirge`, `Eidolon`, `Protector`, `BoneShards`, `Rattle`, owner-aware `HighFive`, `DrainPower`, `SculptingStrike`, `EndOfDays`, `GlimpseBeyond`, `Severance`, `SoulStorm`, `TheScythe`, `TimesUp`, `BorrowedTime`, `CountdownCard`, `DanseMacabre`, and `DeathMarch`
- Entropic Brew and combat potion-slot filling
- Combat potion action decoding and bridge mask generation
- Thieving Hopper's original weak-encounter behavior: 79 HP, `EscapeArtist(5)`, fixed steal / flutter / attack / escape rotation, card stealing through `Swipe`, no accidental gold theft, and stolen-card return on death
- Bygone Effigy's original elite move ids and fixed sleep / wake / slash rotation
- Act 1 weak monster move ids for Shrinker Beetle, Nibbit, Leaf Slime S, Twig Slime S, and Twig Slime M now match the decompiled move-state names
- Act 1 normal monster move ids for Cubex Construct, Flyconid, Eye With Teeth, Fogmog, Mawler, Ruby Raiders, and Byrdonis now match the decompiled move-state names; Slithering Strangler now opens with `CONSTRICT` and then rolls `TWACK` / `LASH` like the original
- Act 2 monster move ids and several direct behavior mismatches now match the decompiled models for Tunneler, Bowlbugs, Exoskeletons, Chomper, Hunter-Killer, Ovicopter / Tough Egg, Slumbering Beetle, Spiny Toad, The Obscura / Parafright, Decimillipede, Entomancer, The Insatiable, Knowledge Demon, and Kaiser Crab
- Act 4 weak monster move ids, HP ranges, and direct behaviors now match the decompiled models for Corpse Slug, Seapunk, Sludge Spinner, and Toadpole; the related Corpse Slug and Toadpole encounter compositions now match the original setup
- Act 4 normal move ids and direct behaviors now match the decompiled models for Calcified Cultist, Damp Cultist, Fossil Stalker, Gremlin Merc, Sneaky Gremlin, and Fat Gremlin
- Act 4 normal Punch Construct and Sewer Clam now match the decompiled HP, setup powers, move ids, direct effects, and opening cycles
- Act 4 normal Haunted Ship, Living Fog / Gas Bomb, and Two Tailed Rat now match the decompiled HP, move ids, direct effects, and normal encounter composition
- Act 4 elites now match the decompiled HP, setup powers, move ids, direct effects, and Phantasmal Gardeners encounter composition
- Act 4 bosses now match the decompiled HP, setup powers, move ids, direct effects, death/wake transitions, and boss encounter composition for Waterfall Giant, Soul Fysh, and Lagavulin Matriarch

## Current Confirmed Blockers

### 1. Exact parity still exceeds the current audited surface

The codebase is no longer blocked on the old “core helper missing” category, but broad exact-match claims still require more decompiled-backed tests across:

- colorless and event cards outside the targeted choice-flow subset
- full Regent and Necrobinder regression coverage
- relic interactions across combat, shop, rewards, and rest-site hooks
- broader bridge smoke testing against a live game
- broader random-call boundary audits outside the currently fixed event / combat stream / clone-dupe id routes

This is primarily a coverage gap, not proof of incorrect behavior, but it prevents claiming an exact match.

### 2. Implemented but not yet fully parity-audited cards

There are still implemented cards that need deeper decompiled-backed coverage, but the previously tracked `Compact`, `WhiteNoise`, and `TheHunt` items are now covered by dedicated parity tests.

### 3. Bridge/runtime validation gap

The Python-side bridge adapter and the C# bridge handlers now cover the main
actionable run surfaces: map routing, combat rooms, reward screens, card
rewards, card bundles, Crystal Sphere, rest sites, shops, events, treasure
rooms, boss-relic choices, and terminal run states.

That is still not the same as a proven normal full-run game flow. The current
mod uses the game's AutoSlay automation framework to navigate menus, drain UI
screens, and recover from transitions. It replaces the important gameplay
choices with RL bridge handlers, but it remains an automated run path rather
than the original manual UI path.

The remaining default AutoSlay card-selection screen handlers are expected to
be bypassed by `CardSelectCmd.UseSelector(new RlCardSelector())` for normal
deck upgrade, transform, enchant, simple-grid, deck-card, and choose-a-card
flows. `tests/test_bridge_autoslay_coverage.py` guards that wiring, but this is
still weaker than a live-game smoke test that exercises those paths in the
running client.

Local bridge build validation is currently blocked. On 2026-05-22, this
machine still did not have `dotnet` on `PATH`, so the C# bridge mod could not be
compiled here.

Until a C# build and live-game smoke pass are available, bridge support should
be considered implemented and Python-tested, but not fully field-verified.

### 4. Reachability / semantic audit backlog

The previously ambiguous no-op / quest hook items are now covered:

- deprecated Act 3 event stubs are covered as disabled / optionless
- status and curse `OnPlay` no-ops are covered as unplayable, true no-ops, or hooks handled elsewhere
- quest card hooks for `ByrdonisEgg`, `LanternKey`, and `SpoilsMap` are implemented or covered

This closes the old no-op ambiguity category, but it does not by itself prove full exact parity across the broader unaudited surfaces above.

### 5. Encounter generation order

The game builds every act's rooms at run start from `Rng.UpFront`
(`ActModel.GenerateRooms`): the event order, weak/normal/elite encounter lists,
the boss and the ancient. The simulator shuffles events from `up_front` but
picks weak, normal and elite fights when each room is entered. Each act's boss
is fixed when the act starts (`RunState.pick_act_boss`), from a stream of its
own (`act_N_boss`), so it is known from the act's start as in the game and no
other draw moves. The same seed still gives a different boss than the game.
The A10 second boss is not simulated.

## Standard for Claiming Exact Parity

We should not describe `sts2_env` as an exact match until all of the following are true:

1. The remaining confirmed blockers above are closed.
2. Every gameplay-affecting divergence is either implemented exactly or proven unreachable.
3. The bridge path is compiled and smoke-tested against a live game from
   opening, through map selection, combat, rewards, non-combat rooms, and run
   completion.
4. Full-run replay comparison exists, or an equivalent lifecycle proof covers
   the complete run path instead of only actionable slices.
5. The remaining no-op markers are limited to base-class defaults or explicitly documented unreachable content.

## Repository Pointers

- Combat core: `sts2_env/core/combat.py`
- Card implementations: `sts2_env/cards/`
- Potion implementations: `sts2_env/potions/`
- RL envs: `sts2_env/gym_env/`
- Bridge adapter: `sts2_env/bridge/state_adapter.py`
- Bridge mod: `bridge_mod/RlCombatHandler.cs`
- Bridge AutoSlay wiring guard: `tests/test_bridge_autoslay_coverage.py`
- Decompiled reference: `decompiled/MegaCrit.Sts2.Core.Models.*`

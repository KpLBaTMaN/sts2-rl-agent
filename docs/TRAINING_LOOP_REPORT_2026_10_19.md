# Training decision loop investigation

19 October 2026. Evidence snapshot: 21:35 BST. Diagnosis only; training settings, models, simulator logic and bridge behavior were left unchanged.

**Every saved timeout from the completed sparse and shaped models reproduced as a card selection stall.** Across their final and best checkpoints, 124 of 800 holdout episode evaluations timed out: 119 repeatedly selected and deselected cards, and five repeatedly attempted an invalid extra selection. Separately, real game traces contain reward navigation, repeated End Turn and unplayable Wound loops. Existing working tree changes already address parts of those historical bridge failures.

## Training runs and measured impact

Each completed checkpoint was evaluated on 200 distinct holdout seeds, 20000 through 20199. Best and final checkpoints reuse those seeds, so the rows below are checkpoint evaluations, not 800 different seeds.

| Experiment | Checkpoint steps | Wins | Timeouts | Timeout rate | Original evidence |
| --- | ---: | ---: | ---: | ---: | --- |
| Sparse final | 5,001,216 | 11 | 34 | 17% | [Final summary](C:/Users/40235515/Desktop/Code/sts2-rl-agent/output/reward-comparison-20261019/runs/long-sparse-20261019-093922-da16c5/final-results.json) |
| Sparse best | 3,002,368 | 12 | 34 | 17% | [Best summary](C:/Users/40235515/Desktop/Code/sts2-rl-agent/output/reward-comparison-20261019/runs/long-sparse-20261019-093922-da16c5/best-results.json) |
| Shaped final | 5,001,216 | 1 | 27 | 13.5% | [Final summary](C:/Users/40235515/Desktop/Code/sts2-rl-agent/output/reward-comparison-20261019/runs/long-shaped-20261019-143223-9dfec3/final-results.json) |
| Shaped best | 3,002,368 | 10 | 29 | 14.5% | [Best summary](C:/Users/40235515/Desktop/Code/sts2-rl-agent/output/reward-comparison-20261019/runs/long-shaped-20261019-143223-9dfec3/best-results.json) |

All 124 replays matched the saved seed, floor, remaining HP and 2,000 decisions. Of the 248,000 decisions consumed by these timeout episodes, 235,728 occurred from the first detected cycle state onward, about 95.1%.

The survival run was still training. Its latest completed evaluation at the snapshot was at 1,253,376 steps: 2 wins and 6 timeouts in 50 episodes. These are evaluation statistics, not a completed final result or a causal replay of that checkpoint. The changing records are frozen in [the survival snapshot](C:/Users/40235515/Desktop/Code/sts2-rl-agent/output/loop-audit-20261019/snapshots/survival-latest-evaluation.json); its evaluation is on line 583 of [the captured metrics](C:/Users/40235515/Desktop/Code/sts2-rl-agent/output/loop-audit-20261019/snapshots/survival-metrics.jsonl:583).

The three dashboard smoke runs intentionally used a 16-decision episode limit. Their timeouts alone do not demonstrate a loop. The three long runs used random training seeds; the fixed seed in the current default training configuration does not explain these completed experiments.

## Finding 1 Card selections alternate forever

**Priority: P1. Still reproduces with the current simulator and saved models.**

| Pending selection | Two state cycles across the four checkpoints |
| --- | ---: |
| Choose any number of hand cards to discard | 46 |
| Choose any number of hand cards to exhaust | 35 |
| Choose 2 cards to upgrade | 36 |
| Choose 3 cards to enchant with Sharp | 2 |
| Total | 119 |

These are actual cycles in selection state. Clicking an already selected card deselects it. The policy then selects it again instead of confirming, and combat or the reward screen remains paused.

Exact examples:

- **Discard:** shaped final, seed **20007**, floor 6. Decisions **69 and 70** alternate between selected indexes `{3,4}` and `{4}`; action **4**, selecting or deselecting the fourth card, wins in both states. Decision 71 returns to the state at decision 69. The original timeout is [CSV line 9](C:/Users/40235515/Desktop/Code/sts2-rl-agent/output/reward-comparison-20261019/runs/long-shaped-20261019-143223-9dfec3/final-results-episodes.csv:9). The selected sets and probabilities are in [selection evidence](C:/Users/40235515/Desktop/Code/sts2-rl-agent/output/loop-audit-20261019/selection_examples.json:44); the full replay starts its repeating segment at [decision 69](C:/Users/40235515/Desktop/Code/sts2-rl-agent/output/loop-audit-20261019/long-shaped-20261019-143223-9dfec3-final-results-seed20007-decisions.jsonl:69).
- **Exhaust:** shaped final, seed **20080**, floor 12. Decisions **103–105** repeatedly toggle the third card, a Strike. The original timeout is [CSV line 82](C:/Users/40235515/Desktop/Code/sts2-rl-agent/output/reward-comparison-20261019/runs/long-shaped-20261019-143223-9dfec3/final-results-episodes.csv:82); see [selection evidence](C:/Users/40235515/Desktop/Code/sts2-rl-agent/output/loop-audit-20261019/selection_examples.json:438).
- **Whetstone upgrade:** shaped final, seed **20039**, floor 9. Decisions **156–158** alternate between two selected Angers and one selected Anger, using action **7**. The original timeout is [CSV line 41](C:/Users/40235515/Desktop/Code/sts2-rl-agent/output/reward-comparison-20261019/runs/long-shaped-20261019-143223-9dfec3/final-results-episodes.csv:41); see [selection evidence](C:/Users/40235515/Desktop/Code/sts2-rl-agent/output/loop-audit-20261019/selection_examples.json:241).
- **Gnarled Hammer enchantment:** sparse final, seed **20066**, cycles from decision **193**; sparse best cycles from decision **138** on the same seed. Both original records are at line **68** in their respective episode CSVs. The full final-checkpoint finding is in [replay evidence](C:/Users/40235515/Desktop/Code/sts2-rl-agent/output/loop-audit-20261019/replay_results.json:11051).

The selection semantics are implemented in [PendingCardChoice.toggle](C:/Users/40235515/Desktop/Code/sts2-rl-agent/sts2_env/core/selection.py:61). Combat continues only after confirmation in [CombatState.resolve_pending_choice](C:/Users/40235515/Desktop/Code/sts2-rl-agent/sts2_env/core/combat.py:2504); deck selections use [RunState.resolve_pending_choice](C:/Users/40235515/Desktop/Code/sts2-rl-agent/sts2_env/run/run_state.py:1470).

The observation exposes the selected cards in [observation.py](C:/Users/40235515/Desktop/Code/sts2-rl-agent/sts2_env/entity/observation.py:199), so this is not simply missing selection flags. The saved policies strongly suppress confirmation: in the discard example its probability is approximately **0.000000068**. Evaluation uses deterministic argmax in [EntityAgent.predict](C:/Users/40235515/Desktop/Code/sts2-rl-agent/sts2_env/entity/ppo.py:50), which repeats the same choice whenever that state returns.

## Finding 2 The deck selection mask permits rejected actions

**Priority: P1. Confirmed environment defect; still reproduces.**

Five additional timeouts are single state loops during Whetstone upgrades. After two cards are selected, the mask permits selecting an unselected third card. The selection rejects it because the maximum is two. Nothing changes, and deterministic inference chooses it again.

The clearest example is shaped best, seed **20026**, floor 9, starting at decision **119**. Two Angers at indexes 6 and 8 are selected. Action **1** tries to add the first Strike. `mask_allows_action` is true while `selection_allows_action` is false. This repeats through decision 2000. See [original CSV line 28](C:/Users/40235515/Desktop/Code/sts2-rl-agent/output/reward-comparison-20261019/runs/long-shaped-20261019-143223-9dfec3/best-results-episodes.csv:28) and [the contradictory mask and selection results](C:/Users/40235515/Desktop/Code/sts2-rl-agent/output/loop-audit-20261019/selection_examples.json:599).

The exact mismatch is between:

- [RunManager._actions_run_choice](C:/Users/40235515/Desktop/Code/sts2-rl-agent/sts2_env/run/run_manager.py:866), which lists every card without applying `can_toggle`.
- [STS2RunEnv.action_masks](C:/Users/40235515/Desktop/Code/sts2-rl-agent/sts2_env/gym_env/run_env.py:409), which unmasks all those choices.
- [PendingCardChoice.toggle](C:/Users/40235515/Desktop/Code/sts2-rl-agent/sts2_env/core/selection.py:61), which rejects a new selection when the maximum is reached.

The combat mask already calls `can_toggle` in [action_space.py](C:/Users/40235515/Desktop/Code/sts2-rl-agent/sts2_env/gym_env/action_space.py:75).

The other affected checkpoint and seed pairs are sparse final **20107**, and sparse best **20042**, **20131**, **20138**. Their original CSV lines are **109**, **44**, **133**, **140**, respectively. All five are indexed in [loop_index.csv](C:/Users/40235515/Desktop/Code/sts2-rl-agent/output/loop-audit-20261019/loop_index.csv).

In four isolated experiments covering discard, exhaust, upgrade toggling and the rejected third selection, sending the already legal **Confirm** action cleared the pending selection immediately. No additional cards were selected. The capped Whetstone example returned to the map. These experiments establish a way out of the stall; they do not establish a higher win rate.

## Finding 3 A declined card reward gets reopened

**Historical live failure; related mitigations already exist in the working tree.**

[Watch trace 20261019-195005-504253](C:/Users/40235515/Desktop/Code/sts2-rl-agent/output/watch/20261019-195005-504253/decisions.jsonl:57), lines **57–143**, alternates between opening `CardReward` and choosing `Skip card reward`. There are **43 skip decisions**. Floor 1, HP 66, gold 111 and deck size 10 remain unchanged. Line 94 changes card display order but repeats the same skip behavior. The session summary records that the user stopped it.

The outer navigation is automatic: [WatchAdapter.choices](C:/Users/40235515/Desktop/Code/sts2-rl-agent/sts2_env/watch/adapter.py:108) uses [_pick_reward_screen_option](C:/Users/40235515/Desktop/Code/sts2-rl-agent/sts2_env/bridge/agent_runner.py:389), which prefers a reward over Proceed. If returning from Skip offers that reward again, navigation reopens it and the policy skips again.

An earlier trace also sends Skip against the identical card screen three times at [lines 56–58](C:/Users/40235515/Desktop/Code/sts2-rl-agent/output/watch/20261019-194555-ac7609/decisions.jsonl:56), consistent with a skip that failed to close the screen.

The current working tree retains attempted reward buttons across handler invocations in [RlRewardsScreenHandler.cs](C:/Users/40235515/Desktop/Code/sts2-rl-agent/bridge_mod/RlRewardsScreenHandler.cs:36), and clicks the native Skip option then waits for closure in [RlCardRewardScreenHandler.cs](C:/Users/40235515/Desktop/Code/sts2-rl-agent/bridge_mod/RlCardRewardScreenHandler.cs:148). Those changes predate this investigation. The old traces prove the failure; they do not prove that it persists in the installed bridge.

## Finding 4 End Turn is sent against an unchanged turn

**Historical live failure; transition handling has existing changes.**

[Watch trace 20261019-195259-e12633](C:/Users/40235515/Desktop/Code/sts2-rl-agent/output/watch/20261019-195259-e12633/decisions.jsonl:105), lines **105–184**, contains **80 identical combat states**, excluding request IDs. Every response is End Turn. The state remains round 1, floor 4, HP 41, block 15, energy 0, with a Shrinker Beetle at 34 HP. End Turn has probability 1 because it is the only legal choice.

This trace demonstrates a game transition or bridge execution stall: the policy cannot advance the unchanged state by choosing another legal action. The exact asynchronous failure cannot be established from the decision transcript alone.

The current [CanRequestCombatDecision and EndTurnAndWaitAsync](C:/Users/40235515/Desktop/Code/sts2-rl-agent/bridge_mod/RlCombatHandler.cs:383) check end-turn readiness and transition phases. These are existing working tree changes. A fresh installed-bridge test must establish that a subsequent request belongs to a genuinely new turn or completed combat.

## Finding 5 The policy repeatedly tries to play Wound

**Historical live legality bug; current mask passes an offline replay check.**

[Watch trace 20261019-200635-caa122](C:/Users/40235515/Desktop/Code/sts2-rl-agent/output/watch/20261019-200635-caa122/decisions.jsonl:175), lines **175–223**, repeats the same state **49 times** at the floor 16 Vantom fight. The policy selects action **3**, Wound, despite the card carrying `playable: false` and `Unplayable`. Wound has cost -1; the old cost-based mask treated negative cost as an X cost when energy was available. The bridge rejects the play, preserving the state. Line 224 changes the action, so this observed streak is bounded by the trace.

The existing change in [StateAdapter.compute_action_mask](C:/Users/40235515/Desktop/Code/sts2-rl-agent/sts2_env/bridge/state_adapter.py:269) respects the game's explicit `playable` field. Replaying line 175 with the current adapter excludes both Wounds. The same saved model chooses Defend instead. See [the recorded and current choices](C:/Users/40235515/Desktop/Code/sts2-rl-agent/output/loop-audit-20261019/current_wound_mask.json).

## Why the training stalls survive

Selection toggles pay zero reward in the captured examples, and only the 2,000-decision limit stops them. [RewardTracker.step](C:/Users/40235515/Desktop/Code/sts2-rl-agent/sts2_env/training/rewards.py:160) rewards changes in game state and applies the timeout result at the end; there is no selection cycle penalty. With the saved `gamma=0.995`, a penalty 1,900 decisions away has a discount factor of about **0.000073**. This weakens the signal against early stalling; its contribution to learning has not been isolated experimentally.

Training samples actions at [ppo.py line 164](C:/Users/40235515/Desktop/Code/sts2-rl-agent/sts2_env/entity/ppo.py:164), while evaluation and live playback use argmax. Neither training evaluation nor the watch worker detects repeated state cycles. The [watch worker](C:/Users/40235515/Desktop/Code/sts2-rl-agent/sts2_env/watch/player.py:145) has silence and total-decision limits, but continuous repeated requests avoid its silence timeout. Confirmation also shares the End Turn output in [the model](C:/Users/40235515/Desktop/Code/sts2-rl-agent/sts2_env/entity/model.py:150); separating that decision is worth testing, rather than assuming more training will teach it reliably.

## Fix order and acceptance checks

1. **Correct deck selection legality.** Apply `can_toggle` while preserving original option indexes. Reject masked actions that the resolver cannot execute. Replay all five capped-selection seeds and verify that a third unselected card is masked while deselection and confirmation remain valid.
2. **Make selections finish reliably.** Add a bounded selection protocol or cycle handling that preserves legitimate editing. For a repeated cycle, confirm only when legal or stop with a precise diagnostic. Cover optional discard and exhaust, fixed-count upgrades and enchantments. Reevaluate all four checkpoints on the same 200 holdout seeds; report wins, deaths and stalls separately.
3. **Record selection decisions.** Persist the prompt, selection limits, selected indexes, action, action result and a state fingerprint. Detect both unchanged states and short alternating cycles. Keep transport request IDs outside the fingerprint. Existing episode CSVs locate a timeout but cannot explain its decision history.
4. **Validate existing bridge mitigations in the game.** Verify reward Skip proceeds once, End Turn advances once and Wound is never offered. Preserve the traces above as regression fixtures. The latest saved live session, [20261019-201643-20d6aa](C:/Users/40235515/Desktop/Code/sts2-rl-agent/output/watch/20261019-201643-20d6aa/summary.json), reports Act 1 victory; none of its 226 decisions forms an exact cycle of period one through four repeated three times.
5. **Then tune reward and model behavior.** Compare a bounded selection penalty or explicit confirmation output after environment legality is corrected. Measure the result on held-out seeds before starting another long training comparison.

## Evidence and reproduction

- [Complete index of all 124 timeouts](C:/Users/40235515/Desktop/Code/sts2-rl-agent/output/loop-audit-20261019/loop_index.csv): experiment, checkpoint, seed, original CSV and line, first cycle decision, period, action and reproduction match.
- [Detailed replay records](C:/Users/40235515/Desktop/Code/sts2-rl-agent/output/loop-audit-20261019/replay_results.json), [selection experiments](C:/Users/40235515/Desktop/Code/sts2-rl-agent/output/loop-audit-20261019/selection_examples.json) and [live cycle evidence](C:/Users/40235515/Desktop/Code/sts2-rl-agent/output/loop-audit-20261019/watch_loops.json).
- [Manifest](C:/Users/40235515/Desktop/Code/sts2-rl-agent/output/loop-audit-20261019/manifest.json): capture time, Git revision, model and source hashes, survival snapshot and test result. The working tree already contained unrelated edits when this investigation began.
- Reproduce from the repository root with `.venv\Scripts\python.exe output/loop-audit-20261019/audit_loops.py`; run `diagnose_examples.py` in the same directory for the confirmation and Wound checks. Replays use CPU inference, configured deterministic decisions and original seeds. Predictions are cached only for byte-identical observations and masks within an episode.
- Existing focused tests passed: **84 tests** across `test_bridge_state_adapter.py`, `test_run_deck_choice_flow.py` and `test_reward_profiles.py`. Those tests passing does not invalidate the reproduced stalls; the missing coverage is selection liveness and the deck mask at maximum selection count.

Eight live session directories were inspected, including one empty transcript, with 1,205 recorded decisions across the other seven. Live cycle scanning compared complete states after removing only `request_id`, for periods one through four repeated at least three times. Normal repeated card plays or End Turn actions with changing state were not classified as loops. The 124 causal reproductions cover every saved timeout in the completed models' final and best holdout evaluations; they do not quantify loops across all stochastic training episodes, which lack per-decision transcripts.


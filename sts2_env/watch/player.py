"""Watch a saved policy in the real Steam game with local playback controls."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import secrets
import shutil
import subprocess
import threading
import time
import webbrowser

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from sts2_env.bridge.client import STS2GameClient
from sts2_env.watch.adapter import WatchAdapter
from sts2_env.watch.control import Playback, Stopped
from sts2_env.entity.ppo import EntityAgent, stack_observations, to_tensors
from sts2_env.entity.model import masked_logits
from sts2_env.watch.entity_adapter import EntityWatchAdapter
from sts2_env.training.loops import LoopDetector, fingerprint

ROOT = Path(__file__).resolve().parents[2]
LOG = logging.getLogger(__name__)


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def prepare(config_file):
    settings = json.loads(resolve(config_file).read_text(encoding="utf-8-sig"))
    allowed = {"experiment", "model", "seed", "start_paused", "decision_delay", "port", "game_path", "steam_path"}
    if set(settings) - allowed:
        raise ValueError(f"Unknown watch options: {set(settings) - allowed}")
    if settings.get("experiment", "latest") == "latest":
        candidates = [p.parent for p in (ROOT / "output").rglob("final_model.*")
                      if p.suffix in {".zip", ".pt"} and (p.parent / "config.json").exists()
                      and not p.parent.name.startswith("smoke")]
        if not candidates:
            raise ValueError("No completed training experiments found")
        experiment = max(candidates, key=lambda p: p.stat().st_mtime)
    else:
        experiment = resolve(settings["experiment"])
    training = json.loads((experiment / "config.json").read_text(encoding="utf-8-sig"))
    if training["task"] == "combat":
        raise ValueError("Watch-Model uses Act 1/full-run experiments. Play-AI.cmd runs the old combat model.")
    entity = training.get("model", {}).get("type", "flat") == "entity"
    which = settings.get("model", "final")
    if which not in {"final", "best"}:
        raise ValueError("model must be final or best")
    extension = "pt" if entity else "zip"
    model_path = experiment / (f"final_model.{extension}" if which == "final" else f"best_model/best_model.{extension}")
    if not model_path.exists():
        raise ValueError(f"Saved model missing: {model_path}")
    seed = settings.get("seed")
    if seed is None:
        seed = training["seeds"]["values"][0] if training["seeds"]["mode"] != "random" else secrets.randbelow(2147483647)
    if not isinstance(seed, (str, int)) or isinstance(seed, bool) or not str(seed).strip():
        raise ValueError("seed must be a simulator integer or a real-game seed string")
    seed = str(seed).strip().upper().replace("O", "0").replace("I", "1")
    if not seed.isalnum() or len(seed) > 32:
        raise ValueError("seed must contain at most 32 letters/digits")
    live = dict(seed=seed, character=training["character"], ascension=training["ascension"],
                start_with_neow=training["start_with_neow"], task=training["task"])
    port = settings.get("port", 8766)
    if type(port) is not int or not 1024 <= port <= 65535:
        raise ValueError("port must be 1024–65535")
    if type(settings.get("start_paused", True)) is not bool:
        raise ValueError("start_paused must be true/false")
    if not 0 <= settings.get("decision_delay", 0.8) <= 5:
        raise ValueError("decision_delay must be between 0 and 5")
    return settings, training, experiment, model_path, live


def model_decision(model, adapter, state):
    choices, automatic = adapter.choices(state)
    if automatic:
        wire, label, source = automatic
        return wire, dict(label=label, source=source, probabilities=[])
    n_actions = model.model.config.n_actions if isinstance(model, EntityAgent) else model.action_space.n
    mask = np.zeros(n_actions, dtype=bool)
    for action in choices:
        if action >= len(mask):
            raise ValueError("Model action space does not match the bridge")
        mask[action] = True
    if state["type"] == "card_select":
        return select_cards(model, adapter, state, choices, mask)
    probabilities = policy_probabilities(model, adapter.encode(state), mask)
    action = int(np.argmax(probabilities))
    wire, label = choices[action]
    labels = {a: choice[1] for a, choice in choices.items()}
    return wire, proposal_for(model, action, label, labels, probabilities)


def policy_probabilities(model, obs, mask):
    with torch.no_grad():
        if isinstance(model, EntityAgent):
            logits, _ = model.model(to_tensors(stack_observations([obs]), model.device))
            return torch.softmax(masked_logits(logits, torch.as_tensor(mask[None], device=model.device)), -1).cpu().numpy()[0]
        tensor, _ = model.policy.obs_to_tensor(obs)
        distribution = model.policy.get_distribution(tensor, action_masks=mask)
        return distribution.distribution.probs.cpu().numpy()[0]


def proposal_for(model, action, label, labels, probabilities):
    ranked = sorted(labels, key=lambda a: -probabilities[a])[:5]
    return dict(action=action, label=label, source="Trained SpireFormer policy" if isinstance(model, EntityAgent) else "Trained Act 1 policy",
                probabilities=[dict(label=labels[a], probability=float(probabilities[a])) for a in ranked])


def select_cards(model, adapter, state, choices, mask):
    """Pick cards one at a time as in training: each pick is final and Confirm (action 0) ends it.

    The bridge takes the whole selection in one reply, so the picks are made here
    against an observation that shows the cards chosen so far, then sent together.
    """
    minimum, maximum = state.get("min_select", 0), state.get("max_select", 1)
    multi = minimum != 1 or maximum != 1
    required = min(minimum, len(state.get("cards", [])))
    options = {a: choice[0]["index"] for a, choice in choices.items() if a > 0}
    selected, picks, first = [], [], None
    # Every pass either confirms or adds a card, so this ends within maximum + 1 passes.
    while True:
        step = np.zeros_like(mask)
        step[0] = len(selected) >= required
        if len(selected) < maximum:
            for action, index in options.items():
                step[action] = index not in selected
        if not step.any():
            if first is None:
                raise ValueError("The live card selection offers no legal choice")
            break
        labels = {a: (choices[a][1] if a else "Skip selection" if not selected else "Confirm selection")
                  for a in np.flatnonzero(step)}
        probabilities = policy_probabilities(model, adapter.encode(state, selected), step)
        action = int(np.argmax(probabilities))
        first = first or (action, labels, probabilities)
        picks.append(dict(label=labels[action], probability=float(probabilities[action])))
        if action == 0:
            break
        selected.append(options[action])
        if not multi:
            break
    if not selected:
        wire, label = {"action": "skip"}, "Skip selection"
    elif not multi:
        wire, label = {"action": "choose", "index": selected[0]}, picks[0]["label"]
    else:
        names = [str(state["cards"][i].get("id", i)) for i in selected]
        wire, label = {"action": "choose", "indexes": selected}, f"Select {', '.join(names)} (one at a time, then confirm)"
    action, labels, probabilities = first
    return wire, dict(proposal_for(model, action, label, labels, probabilities), picks=picks)


def decision_key(state, wire):
    """Identify a live decision by its full state and reply; request IDs are transport only."""
    content = {key: value for key, value in state.items() if key != "request_id"}
    reply = {key: value for key, value in wire.items() if key != "request_id"}
    return fingerprint(json.dumps(content, sort_keys=True), json.dumps(reply, sort_keys=True))


def loop_message(period, repeats, state, proposal):
    cycle = "the same decision" if period == 1 else f"the same {period}-decision cycle"
    return (f"The game repeated {cycle} {repeats} times without progress "
            f"({state['type']}: {proposal['label']}). Automation stopped before sending it again; "
            "see decisions.jsonl for the repeating states.")


def compare_map(state, live):
    """Report actual map differences; matching numeric seed is not a parity claim."""
    try:
        from sts2_env.run.run_manager import RunManager
        manager = RunManager(seed=int(live["seed"]), character_id=live["character"], ascension_level=live["ascension"])
        simulated = manager.run_state.map.all_points()
        expected = {(p.row, p.col) for p in simulated}
        actual = {(p["row"], p["col"]) for p in state.get("map_nodes", [])}
        if not actual:
            return "Map comparison unavailable"
        # Node coordinates differ between implementations, so report counts and overlap.
        return (f"Simulator has {len(expected)} map nodes; real game has {len(actual)}. "
                f"{len(expected & actual)} coordinates overlap. Exact map/encounter parity is not established.")
    except (ValueError, TypeError):
        return "Real-game seed string selected; no integer simulator map comparison is available."


def worker(playback, model, live, directory, max_decisions):
    adapter = EntityWatchAdapter() if isinstance(model, EntityAgent) else WatchAdapter(model.observation_space.shape[0])
    transcript = (directory / "decisions.jsonl").open("a", encoding="utf-8")
    client = STS2GameClient(timeout=2, reconnect_attempts=60, reconnect_delay=2)
    connected = False
    try:
        client.connect()
        connected = True
        with playback.condition:
            playback.status = "Connected; waiting for a decision"
        map_checked = False
        last_response = time.monotonic()
        # Continuous repeats never trip the silence timeout, so stop on exact cycles instead.
        loops = LoopDetector()
        while not playback.stopped:
            try:
                state = client.receive_state()
            except TimeoutError:
                if time.monotonic() - last_response > 90:
                    raise ValueError("The game has not produced a new decision for 90 seconds; inspect the game log.")
                continue
            if state.get("type") in {"run_complete", "game_over"}:
                with playback.condition:
                    playback.finished, playback.paused = True, True
                    playback.status = f"Finished: {state.get('result', state.get('message', state['type']))}"
                break
            context = state.get("run_state", {})
            if context:
                if str(context.get("seed")) != live["seed"] or context.get("ascension") != live["ascension"]:
                    raise ValueError("Live seed/difficulty do not match the saved watch configuration")
                if str(context.get("character", "")).replace("_", "").upper() != live["character"].replace("_", "").upper():
                    raise ValueError("Live character does not match the saved watch configuration")
            if not map_checked and state.get("map_nodes"):
                with playback.condition:
                    playback.context["map_comparison"] = compare_map(state, live)
                map_checked = True
            if playback.count >= max_decisions:
                raise ValueError("Configured decision limit reached. Automation has stopped.")
            wire, proposal = model_decision(model, adapter, state)
            key = decision_key(state, wire)
            period = loops.push(key)
            if period:
                transcript.write(json.dumps(dict(state=state, proposal=proposal, sent=None, fingerprint=key,
                                                 stopped=f"loop period {period}")) + "\n")
                transcript.flush()
                raise ValueError(loop_message(period, loops.repeats, state, proposal))
            playback.gate(state, proposal)
            # Control messages consume the client's implicit correlation ID too.
            # Attach the game request ID explicitly to the actual decision.
            wire = dict(wire, request_id=state["request_id"])
            client.send_action({"action": "watch_speed", "multiplier": 5 if playback.skip else 1})
            client.send_action(wire)
            last_response = time.monotonic()
            transcript.write(json.dumps(dict(state=state, proposal=proposal, sent=wire, fingerprint=key)) + "\n")
            transcript.flush()
    except Stopped:
        with playback.condition:
            playback.status = "Stopped by user"
    except Exception as exc:
        LOG.exception("Watch stopped")
        with playback.condition:
            playback.error, playback.status = str(exc), "Stopped on error"
    finally:
        if connected:
            try:
                client.send_action({"action": "watch_stop"})
            except Exception:
                pass
        client.disconnect()
        transcript.close()
        with playback.condition:
            playback.finished = True
        (directory / "summary.json").write_text(json.dumps(playback.snapshot(), indent=2), encoding="utf-8")


def make_server(playback, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send_json(self, value, status=200):
            data = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/api/state":
                return self.send_json(playback.snapshot())
            if self.path != "/":
                return self.send_error(404)
            data = (Path(__file__).parent / "watch.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path != "/api/control":
                return self.send_error(404)
            if self.headers.get("Origin") not in (None, f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
                return self.send_error(403)
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 2048:
                    raise ValueError("Invalid request size")
                playback.control(json.loads(self.rfile.read(size)))
                self.send_json({"ok": True})
            except (ValueError, TypeError, KeyError) as exc:
                self.send_json({"error": str(exc)}, 400)
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="local/config/watch.json")
    parser.add_argument("--no-launch", action="store_true", help="Connect to an already launched watch-mode game")
    parser.add_argument("--headless", action="store_true", help="Diagnostic game check without graphics")
    parser.add_argument("--quit-after", type=int, default=0, help="Diagnostic game frame limit")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    settings, training, experiment, model_path, live = prepare(args.config)
    torch.set_num_threads(4)
    if training.get("model", {}).get("type", "flat") == "entity":
        model, _ = EntityAgent.load(model_path, device="cpu")
        if model.model.config.n_actions != 157:
            raise ValueError("This viewer requires a 157-action Act 1/full-run SpireFormer")
    else:
        model = MaskablePPO.load(model_path, device="cpu")
        if model.observation_space.shape != (151,) or model.action_space.n != 157:
            raise ValueError("This viewer requires a 151-input, 157-action Act 1/full-run model")
    game = Path(settings.get("game_path", "E:/SteamLibrary/steamapps/common/Slay the Spire 2"))
    steam = Path(settings.get("steam_path", "C:/Program Files (x86)/Steam/steam.exe"))
    if not game.is_dir() or not steam.is_file():
        raise ValueError("Check game_path and steam_path in local/config/watch.json")
    if not args.no_launch:
        import ctypes
        # Avoid starting a second game or replacing another live session's settings.
        running = subprocess.run(["C:/Windows/System32/tasklist.exe", "/FI", "IMAGENAME eq SlayTheSpire2.exe", "/FO", "CSV"], capture_output=True, text=True)
        if '"SlayTheSpire2.exe"' in running.stdout:
            raise ValueError("Close Slay the Spire 2 before starting Watch-Model.cmd")
    directory = ROOT / "output/watch" / (datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + f"-{secrets.token_hex(3)}")
    directory.mkdir(parents=True)
    session = dict(live=live, model=str(model_path), experiment=str(experiment), training=training)
    (directory / "session.json").write_text(json.dumps(session, indent=2), encoding="utf-8")
    playback = Playback(settings.get("start_paused", True), settings.get("decision_delay", 0.8))
    playback.context = dict(model=str(model_path), experiment=str(experiment), **live,
                            records=str(directory), map_comparison="Waiting for live map")
    port = settings.get("port", 8766)
    server = make_server(playback, port)
    if not args.no_launch:
        mod_config = game / "mods/STS2BridgeMod/watch-session.json"
        if not mod_config.parent.exists():
            raise ValueError("The watch-enabled STS2BridgeMod must be installed first")
        mod_config.write_text(json.dumps(live, indent=2), encoding="utf-8")
        command = [str(steam), "-applaunch", "2868840", "--sts2-watch"]
        if args.headless:
            command += ["--headless"]
        if args.quit_after:
            command += ["--quit-after", str(args.quit_after)]
        subprocess.Popen(command, creationflags=subprocess.CREATE_NO_WINDOW)
    threading.Thread(target=worker, args=(playback, model, live, directory, training["training"]["max_episode_steps"]), daemon=True).start()
    url = f"http://127.0.0.1:{port}/"
    print(f"Watch controls: {url}\nModel: {model_path}\nSeed: {live['seed']}\nRecords: {directory}", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        playback.control({"action": "stop"})
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

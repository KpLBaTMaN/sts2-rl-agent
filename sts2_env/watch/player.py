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
        candidates = [p.parent for p in (ROOT / "output/experiments").glob("*/final_model.zip")
                      if not p.parent.name.startswith("smoke")]
        if not candidates:
            raise ValueError("No completed training experiments found")
        experiment = max(candidates, key=lambda p: p.stat().st_mtime)
    else:
        experiment = resolve(settings["experiment"])
    training = json.loads((experiment / "config.json").read_text(encoding="utf-8-sig"))
    if training["task"] == "combat":
        raise ValueError("Watch-Model uses Act 1/full-run experiments. Play-AI.cmd runs the old combat model.")
    if training.get("model", {}).get("type", "flat") == "entity":
        raise ValueError("Watch-Model cannot drive the real game with SpireFormer (entity) models yet; "
                         "pick a flat experiment or watch it in the simulator.")
    which = settings.get("model", "final")
    if which not in {"final", "best"}:
        raise ValueError("model must be final or best")
    model_path = experiment / ("final_model.zip" if which == "final" else "best_model/best_model.zip")
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
    mask = np.zeros(model.action_space.n, dtype=bool)
    for action in choices:
        if action >= len(mask):
            raise ValueError("Model action space does not match the bridge")
        mask[action] = True
    obs = adapter.encode(state)
    with torch.no_grad():
        tensor, _ = model.policy.obs_to_tensor(obs)
        distribution = model.policy.get_distribution(tensor, action_masks=mask)
        probabilities = distribution.distribution.probs.cpu().numpy()[0]
    action = int(np.argmax(probabilities))
    wire, label = choices[action]
    # A live multi-select screen sends the complete required set in one UI decision.
    if state["type"] == "card_select" and state.get("min_select", 0) > 1:
        count = min(state["min_select"], state.get("max_select", 1), len(state.get("cards", [])))
        ranking = sorted((a for a in choices if a > 0), key=lambda a: -probabilities[a])
        indexes = [choices[a][0]["index"] for a in ranking[:count]]
        wire = {"action": "choose", "indexes": indexes}
        label = f"Select cards {indexes} (ranked by policy)"
    ranked = sorted(choices, key=lambda a: -probabilities[a])[:5]
    return wire, dict(action=action, label=label, source="Trained Act 1 policy",
                      probabilities=[dict(label=choices[a][1], probability=float(probabilities[a])) for a in ranked])


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
    adapter = WatchAdapter(model.observation_space.shape[0])
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
            playback.gate(state, proposal)
            # Control messages consume the client's implicit correlation ID too.
            # Attach the game request ID explicitly to the actual decision.
            wire = dict(wire, request_id=state["request_id"])
            client.send_action({"action": "watch_speed", "multiplier": 5 if playback.skip else 1})
            client.send_action(wire)
            last_response = time.monotonic()
            transcript.write(json.dumps(dict(state=state, proposal=proposal, sent=wire)) + "\n")
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

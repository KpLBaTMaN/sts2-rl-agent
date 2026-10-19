"""Thread-safe playback gate. A step releases one wire decision, never a run."""
from __future__ import annotations

from collections import deque
import threading
import time


class Stopped(Exception):
    pass


class Playback:
    def __init__(self, paused=True, delay=0.8):
        self.condition = threading.Condition()
        self.paused = paused
        self.delay = delay
        self.credits = 0
        self.stopped = False
        self.skip = None
        self.origin = None
        self.state = {}
        self.proposal = None
        self.history = deque(maxlen=100)
        self.count = 0
        self.status = "Waiting for the game"
        self.error = None
        self.finished = False
        self.context = {}

    @staticmethod
    def room(state):
        run = state.get("run_state", {})
        return (run.get("act_index", 0), run.get("floor", state.get("floor", 0)))

    def control(self, command):
        with self.condition:
            action = command.get("action")
            if action == "play":
                self.paused, self.skip, self.credits = False, None, 0
            elif action == "pause":
                self.paused, self.skip, self.credits = True, None, 0
            elif action == "step":
                self.paused, self.skip = True, None
                self.credits += 1
            elif action in {"next_room", "next_combat", "boss"}:
                self.skip, self.origin = action, self.room(self.state)
                self.paused, self.credits = False, 0
            elif action == "delay":
                value = float(command["value"])
                if not 0 <= value <= 5:
                    raise ValueError("Decision delay must be between 0 and 5 seconds")
                self.delay = value
            elif action == "stop":
                self.stopped = True
            else:
                raise ValueError("Unknown playback command")
            self.condition.notify_all()

    def gate(self, state, proposal):
        with self.condition:
            room = self.room(state)
            hit = (self.skip == "next_room" and room != self.origin or
                   self.skip == "next_combat" and state["type"] == "combat_action" and room != self.origin or
                   self.skip == "boss" and state["type"] == "combat_action" and state.get("run_state", {}).get("room") == "Boss")
            if hit:
                self.skip, self.paused, self.credits = None, True, 0
            self.state, self.proposal = state, proposal
            self.status = "Paused before decision" if self.paused else "Playing"
            deadline = time.monotonic() + (0 if self.skip else self.delay)
            while True:
                if self.stopped:
                    raise Stopped()
                if self.credits:
                    self.credits -= 1
                    break
                if not self.paused and time.monotonic() >= deadline:
                    break
                self.condition.wait(timeout=0.1)
            self.count += 1
            self.history.appendleft({"decision": self.count, "floor": room[1],
                                     "type": state["type"], "label": proposal["label"], "source": proposal["source"]})
            self.status = "Waiting for game to finish this decision"

    def snapshot(self):
        with self.condition:
            return dict(paused=self.paused, stopped=self.stopped, finished=self.finished,
                        status=self.status, error=self.error, decisions=self.count,
                        skip=self.skip, delay=self.delay, state=self.state,
                        proposal=self.proposal, history=list(self.history), context=self.context)

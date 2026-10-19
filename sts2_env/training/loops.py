"""Detect decision loops: one state repeated, or a short cycle of states repeated.

A deterministic policy that sees the same state again makes the same choice, so
an exact cycle never ends by itself.  Keys are compared, not interpreted: the
caller decides what identifies a state (never transport request IDs).
"""
from __future__ import annotations

from collections import deque
import hashlib

MAX_PERIOD = 4
REPEATS = 3


def fingerprint(*parts: bytes | str) -> str:
    """Short stable hash of state parts."""
    digest = hashlib.blake2b(digest_size=8)
    for part in parts:
        digest.update(part if isinstance(part, bytes) else part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


class LoopDetector:
    """Report a period once the last ``repeats`` cycles of keys are identical."""

    def __init__(self, max_period: int = MAX_PERIOD, repeats: int = REPEATS):
        if max_period < 1 or repeats < 2:
            raise ValueError("max_period must be at least 1 and repeats at least 2")
        self.max_period = max_period
        self.repeats = repeats
        self.keys: deque = deque(maxlen=max_period * repeats)

    def reset(self) -> None:
        self.keys.clear()

    def push(self, key) -> int | None:
        """Add one decision's key; return the shortest repeating period, if any."""
        keys = self.keys
        keys.append(key)
        for period in range(1, self.max_period + 1):
            span = period * self.repeats
            if len(keys) >= span and all(keys[-i] == keys[-i - period] for i in range(1, span - period + 1)):
                return period
        return None

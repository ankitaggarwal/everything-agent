"""The 'simple' memory adapter -- a JSON facts file + recent turns.

Zero dependencies, easy to read. Long-term FACTS persist to a JSON file; recent
TURNS live in memory for the session. Good enough until personalization matters
(ROADMAP Phase 4) -- then swap in the Mem0 adapter behind this same port.
"""
from __future__ import annotations

import json
import logging
import os
from collections import deque

from ..core.ports import Memory

log = logging.getLogger("everything_agent.memory")


class SimpleMemory(Memory):
    def __init__(self, config, ctx):
        self.path = config.get("path", "memory.json")
        self.facts = {}
        self.turns = deque(maxlen=config.get("max_turns", 6))
        if os.path.exists(self.path):
            try:
                with open(self.path) as f:
                    self.facts = json.load(f)
            except Exception:
                log.warning("Could not read %s -- starting empty", self.path)

    def _save(self):
        try:
            with open(self.path, "w") as f:
                json.dump(self.facts, f, indent=2)
        except Exception:
            log.exception("Could not save memory to %s", self.path)

    def remember(self, key, value):
        self.facts[key] = value
        self._save()

    def add_turn(self, user, reply):
        self.turns.append((user, reply))

    def context(self):
        lines = []
        if self.facts:
            lines.append("What you know about the user:")
            for k, v in self.facts.items():
                lines.append(f"  - {k}: {v}")
        if self.turns:
            lines.append("Recent conversation:")
            for user, reply in self.turns:
                lines.append(f"  user: {user}")
                lines.append(f"  you: {reply}")
        return "\n".join(lines)

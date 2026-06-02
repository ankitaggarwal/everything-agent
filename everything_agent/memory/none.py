"""The 'none' memory adapter -- remembers nothing.

This is the point of having a port: opting OUT of a capability is just another
adapter, not a special case sprinkled through the code. With memory.backend:
none the agent runs identically, minus recall.
"""
from __future__ import annotations

from ..core.ports import Memory


class NoMemory(Memory):
    def __init__(self, config, ctx):
        pass

    def remember(self, key, value):
        pass

    def add_turn(self, user, reply):
        pass

    def context(self):
        return ""

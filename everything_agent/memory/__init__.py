"""MEMORY port -- what the robot remembers about you.

Adapters:
  none   -> remembers nothing (a real no-op). This is how you run WITHOUT memory.
  simple -> a small JSON facts file + recent turns. Default; zero dependencies.
  mem0   -> (future) the Mem0 memory layer for auto-extraction + semantic recall.
            Add memory/mem0.py + one line below, then set memory.backend: mem0.

Because every adapter implements the same Memory port, switching is one config
line and nothing else in the codebase changes.
"""
from __future__ import annotations

from ..core.plugins import load

BACKENDS = {
    "none": "everything_agent.memory.none:NoMemory",
    "simple": "everything_agent.memory.simple:SimpleMemory",
    # "mem0": "everything_agent.memory.mem0:Mem0Memory",
}


def build(config, ctx):
    return load(BACKENDS, config, ctx, default="simple")

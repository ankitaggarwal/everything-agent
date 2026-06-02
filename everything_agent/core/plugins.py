"""The registry loader -- how config picks an adapter, lazily.

Each port package (robot/, memory/, hearing/stt/, ...) defines a BACKENDS map:

    BACKENDS = {
        "mock":     "everything_agent.robot.mock:MockRobot",
        "reachy_mini": "everything_agent.robot.reachy_mini:ReachyMiniRobot",
    }

`load()` reads `config["backend"]`, looks up the dotted "module:Class" path, and
imports it ONLY THEN. That laziness is the whole point: choosing the `mock`
backend never imports torch, the Cartesia SDK, etc. -- so users install only the
dependencies for the backends they actually turn on.

Adding a backend = write one adapter file + add one line to that BACKENDS map.
No change to the core, the loop, or this file.

(Future: this is also where you'd scan Python entry points to discover backends
shipped as separate pip packages -- without changing any adapter.)
"""
from __future__ import annotations

from importlib import import_module
from typing import Dict


def load(backends: Dict[str, str], config: dict, ctx, default: str = "mock"):
    name = (config or {}).get("backend", default)
    if name not in backends:
        raise ValueError(
            f"Unknown backend {name!r}. Available: {sorted(backends)}. "
            f"Add it to the BACKENDS map for this port."
        )
    module_path, cls_name = backends[name].split(":")
    cls = getattr(import_module(module_path), cls_name)   # lazy import
    return cls(config or {}, ctx)

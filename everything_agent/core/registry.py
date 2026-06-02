"""Loads the modules named in config and aggregates what they contribute.

The registry is the bridge between config.yaml and the running robot: it builds
the enabled modules, calls setup(), and gives the rest of the system simple
combined views -- all actions, all perceptions, all idle ticks.
"""
from __future__ import annotations

import logging
from typing import List

from .module import Action, Module

log = logging.getLogger("everything_agent.registry")


def _module_classes() -> dict:
    """name -> Module class. REGISTER NEW MODULES HERE (one line each).

    Imports are local so one broken module can't crash the whole registry.
    """
    from ..modules.idle.idle import IdleModule
    from ..modules.system_time.system_time import SystemTimeModule
    from ..modules.emotions.emotions import EmotionsModule
    return {
        "idle": IdleModule,
        "system_time": SystemTimeModule,
        "emotions": EmotionsModule,
        # "telegram": TelegramModule,   # Phase 3 (see ROADMAP.md)
    }


class Registry:
    def __init__(self, robot, providers: dict, memory, config: dict):
        self.robot = robot
        self.providers = providers
        self.memory = memory
        self.config = config
        self.modules: List[Module] = []

    def load(self) -> None:
        available = _module_classes()
        for name in self.config.get("modules", []):
            cls = available.get(name)
            if cls is None:
                log.warning("Unknown module %r in config -- skipping", name)
                continue
            module = cls()
            module.setup(self.robot, self.providers, self.memory, self.config)
            self.modules.append(module)
            log.info("Loaded module: %s", module.name)

    def all_actions(self) -> List[Action]:
        actions: List[Action] = []
        for m in self.modules:
            actions.extend(m.actions())
        return actions

    async def gather_perceptions(self) -> str:
        parts = []
        for m in self.modules:
            try:
                p = await m.perceive()
            except Exception:
                log.exception("perceive() failed for %s", m.name)
                continue
            if p:
                parts.append(f"[{m.name}] {p}")
        return "\n".join(parts)

    async def run_ticks(self) -> None:
        for m in self.modules:
            try:
                await m.tick()
            except Exception:
                log.exception("tick() failed for %s", m.name)

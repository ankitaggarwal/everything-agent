"""The Approval gate -- the robot asks before doing anything risky.

Any Action marked `sensitive=True` (ordering food, sending a message, buying,
deleting) is routed through `confirm()` first. This is a first-class design
choice, not an afterthought -- especially important for an open-source robot
that can spend money or message people on your behalf.

Backends:
  interactive (default) : asks you in the terminal y/n.
  auto                  : always approves -- ONLY for headless demos/tests.

Later you can add a "voice" backend that asks out loud and listens for "yes".
"""
from __future__ import annotations

import asyncio
import logging

log = logging.getLogger("everything_agent.approval")


class Approval:
    def __init__(self, config: dict):
        self.auto = (config or {}).get("auto_approve", False)

    async def confirm(self, description: str) -> bool:
        """Return True if the user approves the described action."""
        if self.auto:
            log.info("[approval:auto] auto-approving: %s", description)
            return True
        # Ask in the terminal without blocking the event loop.
        prompt = f"\n[approve?] {description}  (y/n) "
        try:
            answer = await asyncio.to_thread(input, prompt)
        except EOFError:
            return False
        return answer.strip().lower() in ("y", "yes")

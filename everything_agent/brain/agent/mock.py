"""Mock agent: no LLM. Finds ONE local action whose name matches the words and
runs it (respecting the approval gate). Demo-only -- it exists so the whole
pipeline works today, on Python 3.9, with no SDK. Real reasoning and argument
filling is the claude_sdk adapter's job.

Reads the module Actions and the Approval gate from the shared context.
"""
from __future__ import annotations

import logging
import re

from ...core.ports import AgentBrain

log = logging.getLogger("everything_agent.brain.agent")


class MockAgent(AgentBrain):
    def __init__(self, config, ctx):
        self.actions = ctx.actions
        self.approval = ctx.approval

    async def run(self, text: str, memory_context: str = "") -> str:
        spoken = set(re.findall(r"[a-z]+", text.lower()))
        for action in self.actions:
            keywords = [w for w in action.name.split("_") if len(w) > 2]
            if any(w in spoken for w in keywords):
                if action.sensitive and not await self.approval.confirm(action.name):
                    return "Okay, I won't."
                log.info("[agent:mock] calling action %r", action.name)
                return await action.handler({})
        return ("(mock agent) I'd reason about this and pick a tool with the real "
                "Agent SDK. Available: " + ", ".join(a.name for a in self.actions))

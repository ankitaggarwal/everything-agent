"""Mock router: crude keyword rules, no API key. Lets the whole robot run out of
the box. The real routing logic is the anthropic adapter's job.
"""
from __future__ import annotations

from ...core.ports import Decision, Router

_INSTANT_HINTS = ("hello", "hi ", "hey", "joke", "how are you", "thank")
_AGENT_HINTS = ("time", "date", "order", "send", "print", "research",
                "remind", "play", "weather", "turn on", "turn off", "message")


class MockRouter(Router):
    def __init__(self, config, ctx):
        self.config = config

    async def decide(self, text: str, memory_context: str = "") -> Decision:
        if not text or not text.strip():
            return Decision("ignore")
        t = text.lower()
        if any(h in t for h in _AGENT_HINTS):
            return Decision("agent")
        if any(h in t for h in _INSTANT_HINTS):
            return Decision("instant", "Hi! I'm here. (mock instant reply)")
        return Decision("agent")   # default: let the agent + tools try

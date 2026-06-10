"""Haiku agent brain -- a lightweight conversational responder.

A stopgap between the keyword `mock` agent and the full `claude_sdk` tool brain:
it answers with a real Claude Haiku completion, so the robot actually converses.
No tools/MCP yet (that's claude_sdk, which needs `claude login` on the robot).
Uses the same ANTHROPIC_API_KEY as the router. Replies are kept short because
they're spoken aloud.
"""
from __future__ import annotations

import logging

from ...core.ports import AgentBrain

log = logging.getLogger("everything_agent.brain.agent")


class HaikuAgent(AgentBrain):
    def __init__(self, config, ctx):
        self.model = (config or {}).get("model", "claude-haiku-4-5")
        self.timeout = float((config or {}).get("timeout", 30.0))
        from ...persona import DEFAULT_PERSONA
        # No tools here (that's claude_sdk), so drop the movement instructions but
        # keep the same voice/character for a consistent personality.
        self.persona = (config or {}).get("personality") or DEFAULT_PERSONA
        self._client = None   # reused across turns (keeps the HTTP pool warm)

    def _get_client(self):
        if self._client is None:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(timeout=self.timeout)
        return self._client

    async def run(self, text: str, memory_context: str = "") -> str:
        client = self._get_client()
        system = self.persona + ("\n\nWhat you know:\n" + memory_context if memory_context else "")
        try:
            msg = await client.messages.create(
                model=self.model, max_tokens=200, system=system,
                messages=[{"role": "user", "content": text}],
            )
            reply = "".join(
                b.text for b in msg.content if getattr(b, "type", None) == "text"
            ).strip()
            return reply or "Hmm, I'm not sure what to say to that."
        except Exception as e:  # noqa: BLE001
            log.warning("haiku agent failed: %s", e)
            return "Sorry, my brain hiccuped for a second."

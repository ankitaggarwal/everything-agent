"""Anthropic router: one tiny Haiku call decides the path. Needs the `anthropic`
package and ANTHROPIC_API_KEY. This is the real Tier-1 brain (ROADMAP Phase 1).
"""
from __future__ import annotations

import json
import logging

from ...core.ports import Decision, Router

log = logging.getLogger("everything_agent.brain.router")

_SYSTEM = (
    "You are the fast router for an always-on desk robot. Read the user's "
    "utterance and decide how to handle it. Reply with ONLY a JSON object: "
    '{"action": "instant"|"agent"|"ignore", "reply": "<text if instant>"}. '
    "Use 'instant' for greetings, small talk, simple facts, or direct robot "
    "commands you can answer in one line. Use 'agent' when it needs tools, "
    "multiple steps, external services, or real-world actions. Use 'ignore' "
    "for empty or irrelevant input."
)


class AnthropicRouter(Router):
    def __init__(self, config, ctx):
        self.model = config.get("model", "claude-haiku-4-5")
        # A hung routing call stalls the whole voice loop, so keep it short.
        self.timeout = float(config.get("timeout", 10.0))
        self._client = None   # reused across turns (keeps the HTTP pool warm)

    def _get_client(self):
        if self._client is None:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(timeout=self.timeout)
        return self._client

    async def decide(self, text: str, memory_context: str = "") -> Decision:
        if not text or not text.strip():
            return Decision("ignore")
        client = self._get_client()
        user = (memory_context + "\n\nUser: " + text) if memory_context else text
        msg = await client.messages.create(
            model=self.model, max_tokens=200, system=_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        raw = msg.content[0].text if msg.content else ""
        try:
            return self._parse(raw)
        except Exception:
            log.warning("Router returned non-JSON (%r) -- escalating to agent", raw[:60])
            return Decision("agent")

    @staticmethod
    def _parse(raw: str) -> Decision:
        # Haiku sometimes wraps the JSON in prose or ```json fences; pull out the
        # first {...} block before parsing.
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end > start:
            raw = raw[start:end + 1]
        data = json.loads(raw)
        return Decision(data.get("action", "agent"), data.get("reply", ""))
